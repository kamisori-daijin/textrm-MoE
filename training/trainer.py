import os
import math
import glob
import re
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from tqdm import tqdm

from ema.ema import EMA


def train(
    model,
    train_loader,
    val_loader,
    tokenizer,
    epochs=20,
    lr=5e-5,
    warmup_steps=1000,
    n_supervision_steps=4,
    gradient_accumulation_steps=2,
    ema_decay=0.9995,
    aux_loss_coef=0.1,
    save_path="textrm-model.safetensors",
):

    # Set up learning rate schedule and optimizer
    lr_schedule = optim.linear_schedule(0, lr, steps=warmup_steps)
    optimizer = optim.AdamW(
        learning_rate=lr_schedule, betas=[0.9, 0.95], weight_decay=0.1
    )

    latest_model_path = "textrm_latest.safetensors"  
    latest_optim_path = "textrm_latest_optim.safetensors"
 
    # Resume from latest checkpoint if available
    base_name, _ = os.path.splitext(save_path)
    checkpoint_pattern = f"{base_name}_epoch*_val*.safetensors"
    checkpoint_files = glob.glob(checkpoint_pattern)

    chosen_model_path = None
    start_epoch = 0

    if checkpoint_files:
        try:
            
            checkpoint_files.sort(key=lambda x: int(re.search(r'_epoch(\d+)_', x).group(1)))
            chosen_model_path = checkpoint_files[-1]
            start_epoch = int(re.search(r'_epoch(\d+)_', chosen_model_path).group(1))
            print(f"📦 Found saved epoch checkpoint: {chosen_model_path}")
        except Exception:
            chosen_model_path = None

    
    if not chosen_model_path and os.path.exists(latest_model_path):
        chosen_model_path = latest_model_path
        print(f"🔄 Epoch checkpoint not found. Falling back to latest: {latest_model_path}")


    if chosen_model_path and os.path.exists(chosen_model_path):
        print(f"🔄 Restoring model weights from: {chosen_model_path}")
        try:
            model.load_weights(chosen_model_path)
            print(f"✅ Model weights successfully restored. (Resuming from Epoch {start_epoch + 1})")
        except Exception as e:
            print(f"⚠️ Failed to restore model weights: {e}")

  
    chosen_optim_path = None
    if chosen_model_path and chosen_model_path != latest_model_path:
        potential_optim = chosen_model_path.replace(".safetensors", f"_optim.safetensors")
        if os.path.exists(potential_optim):
            chosen_optim_path = potential_optim

    if not chosen_optim_path and os.path.exists(latest_optim_path):
        chosen_optim_path = latest_optim_path

    if chosen_optim_path and os.path.exists(chosen_optim_path):
        print(f"🔄 Restoring optimizer states from: {chosen_optim_path}")
        try:
            loaded_flat_state = mx.load(chosen_optim_path)
            if isinstance(loaded_flat_state, dict):
                optimizer.state = tree_unflatten(loaded_flat_state)
                print("✅ Optimizer states successfully restored.")
            else:
                print("⚠️ Loaded state was not a valid dictionary structure.")
        except Exception as e:
            print(f"⚠️ Failed to restore optimizer state: {e}")

    # Initialize EMA
    ema = EMA(model, decay=ema_decay)

    # Loss function operating on pure trainable parameters
    def loss_fn(params, x, y):
        model.update(params)
        main, aux = model(
            x, y, n_supervision_steps=n_supervision_steps, training=True
        )
        loss = main + aux_loss_coef * aux
        return loss.reshape([]), aux

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # 1. Compile ONLY a single micro-batch step (Keeps Metal graph compact)
    @mx.compile
    def _accumulate_step(trainable_params, accumulated_grads, x, y, scale):
        (loss, aux), grads = loss_and_grad_fn(trainable_params, x, y)
        
        # Accumulate scaled gradients inside a small compiled graph
        next_accumulated_grads = tree_map(
            lambda g, ag: ag + (g * scale), grads, accumulated_grads
        )
        return next_accumulated_grads, loss, aux

    # 2. Compile optimizer and EMA updates together (Separated from loss graph)
    @mx.compile
    def _update_step(trainable_params, optimizer_state, ema_shadow, accumulated_grads):
        model.update(trainable_params)
        optimizer.state.update(optimizer_state)
        
        # Clip and apply gradients
        grads, _ = optim.clip_grad_norm(accumulated_grads, 1.0)
        optimizer.update(model, grads)
        
        # Update EMA shadow
        next_params = model.trainable_parameters()
        def _ema_update(s, p):
            return ema_decay * s + (1.0 - ema_decay) * p
        next_ema_shadow = tree_map(_ema_update, ema_shadow, next_params)
        
        return model.trainable_parameters(), optimizer.state, next_ema_shadow

    best_val_loss = float("inf")

   
    for epoch in range(start_epoch, epochs):
        model.train()

        train_loader_inst = train_loader()
        total_steps = len(train_loader_inst) // gradient_accumulation_steps
        pbar = tqdm(total=total_steps, desc=f"Epoch {epoch + 1}/{epochs}", unit="step")
        current_batches = []

        for i, (input_ids, targets) in enumerate(train_loader_inst):
            current_batches.append((input_ids.astype(mx.int32), targets.astype(mx.int32)))

            if len(current_batches) == gradient_accumulation_steps:
                
                # Fetch initial raw states from objects
                trainable_params = model.trainable_parameters()
                optim_state = optimizer.state
                ema_shadow = ema.shadow
                
                # Safely initialize accumulated gradients tensor matching params structure
                accumulated_grads = tree_map(lambda p: mx.zeros_like(p), trainable_params)
                
                param_dtype = tree_flatten(trainable_params)[0][1].dtype
                total_loss = mx.array(0.0, dtype=param_dtype)
                total_aux = mx.array(0.0, dtype=param_dtype)
                
                n = len(current_batches)
                scale = mx.array(1.0 / n, dtype=param_dtype)

                # Process sequentially in Python to break up Metal kernel sizes
                for x, y in current_batches:
                    accumulated_grads, loss, aux = _accumulate_step(
                        trainable_params, accumulated_grads, x, y, scale
                    )
                    total_loss = total_loss + loss * scale
                    total_aux = total_aux + aux * scale

                # Apply weight updates using the separate compiled update function
                trainable_params, optim_state, ema_shadow = _update_step(
                    trainable_params, optim_state, ema_shadow, accumulated_grads
                )
                
                # Push updated structural dicts back into active objects
                model.update(trainable_params)
                optimizer.state.update(optim_state)
                ema.shadow = ema_shadow

                # Explicitly evaluate everything at once to clear lazy computation paths
                mx.eval(model.parameters(), optimizer.state, ema.shadow, total_loss, total_aux)

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{total_loss.item():.4f}",
                        "aux": f"{total_aux.item():.6f}",
                        "lr": f"{optimizer.learning_rate.item():.6f}",
                    }
                )

                current_batches = []
                mx.clear_cache()

        pbar.close()
        
      
        print(f"💾 Saving latest epoch {epoch + 1} checkpoints for session recovery...")
        model.save_weights(latest_model_path)
        
       
        flat_latest_optim = dict(tree_flatten(optimizer.state, destination={}))
        mx.save_safetensors(latest_optim_path, flat_latest_optim)
        
        # Apply EMA weights for evaluation
        ema.apply_shadow()

        # Validation phase
        val_loss, val_steps = 0.0, 0
        for v_input, v_target in val_loader():
            v_main, v_aux = model(
                v_input,
                v_target,
                n_supervision_steps=n_supervision_steps,
                training=False,
            )
            v_step_loss = v_main + aux_loss_coef * v_aux
            val_loss += v_step_loss.item()
            val_steps += 1

        val_loss /= max(val_steps, 1)
        print(f"Val Loss: {val_loss:.4f}")

        # Save checkpoint
        base, ext = os.path.splitext(save_path)
        if not ext:
            ext = ".safetensors"

        
        checkpoint_name = f"{base}_epoch{epoch + 1:03d}_val{val_loss:.4f}{ext}"
        model.save_weights(checkpoint_name)
        print(f"Checkpoint saved: {checkpoint_name}")

       
        optim_checkpoint_name = f"{base}_epoch{epoch + 1:03d}_val{val_loss:.4f}_optim{ext}"
        flat_optim_state = dict(tree_flatten(optimizer.state, destination={}))
        mx.save_safetensors(optim_checkpoint_name, flat_optim_state)
        print(f"Optimizer state saved: {optim_checkpoint_name}")

        # Generation sample
        test_prompt = "Write a polite refusal email"
        test_ids = mx.array([tokenizer.encode(test_prompt)])
        generated = model.generate(test_ids, max_new_tokens=50)
        print(f"Sample: {tokenizer.decode(generated[0].tolist())[:150]}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights(save_path)
            print(f"Best model updated: {val_loss:.4f}")

        # Restore original online weights
        ema.restore()

    return model