import os
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
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
    aux_loss_coef=0.01,
    save_path="textrm-model.safetensors",
):
    # 1. Configure schedule and optimizer
    lr_schedule = optim.linear_schedule(0, lr, steps=warmup_steps)
    optimizer = optim.AdamW(
        learning_rate=lr_schedule, betas=[0.9, 0.95], weight_decay=0.1
    )

    # Automatically resume optimizer states via direct property assignment conforming to official specs
    latest_optim_path = "textrm_latest_optim.safetensors"
    if os.path.exists(latest_optim_path):
        print(f"🔄 Restoring optimizer states from checkpoint: {latest_optim_path}")
        try:
            loaded_optim_state = mx.load(latest_optim_path)
            optimizer.state = loaded_optim_state
            print("✅ Optimizer states (step count, momentum tables, etc.) successfully restored.")
        except Exception as e:
            print(f"⚠️ Failed to restore optimizer state (initializing from default settings): {e}")

    # Initialize Exponential Moving Average (EMA)
    ema = EMA(model, decay=ema_decay)

    # 2. Purely functional loss routine completely stripped of internal state updates
    def loss_fn(params, x, y):
        main, aux = model(
            x, y, n_supervision_steps=n_supervision_steps, training=True
        )
        loss = main + aux_loss_coef * aux
        return loss.reshape([]), aux

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # 3. Consolidate accumulation, updates, and EMA updates into a single monolithic graph.
    @mx.compile
    def _full_train_step(trainable_params, optimizer_state, ema_shadow, current_batches_x, current_batches_y):
        # Synchronize transient objects to capture state transformations mathematically within the graph
        model.update(trainable_params)
        
        accumulated_grads = tree_map(lambda p: mx.zeros_like(p), trainable_params)
        total_loss = mx.array(0.0, dtype=mx.float32)
        total_aux = mx.array(0.0, dtype=mx.float32)
        
        n = len(current_batches_x)
        scale = 1.0 / n

        for x, y in zip(current_batches_x, current_batches_y):
            (loss, aux), grads = loss_and_grad_fn(trainable_params, x, y)
            
            accumulated_grads = tree_map(
                lambda g, ag: ag + (g * scale), grads, accumulated_grads
            )
            total_loss += loss * scale
            total_aux += aux * scale

        grads, _ = optim.clip_grad_norm(accumulated_grads, 1.0)
        
        # Purely functional updates calculated based on the decoupled `optimizer_state` and parameter tables
        new_params = optimizer.apply_gradients(grads, trainable_params)
        
        def _ema_update(s, p):
            return ema_decay * s + (1.0 - ema_decay) * p
        next_ema_shadow = tree_map(_ema_update, ema_shadow, new_params)
        
        return new_params, optimizer.state, next_ema_shadow, total_loss, total_aux

    best_val_loss = float("inf")
    latest_model_path = "textrm_latest.safetensors"

    # Main training loop
    for epoch in range(epochs):
        model.train()

        train_loader_inst = train_loader()
        # Compute exact tqdm total steps handling fractional macro-batches seamlessly
        total_steps = math.ceil(len(train_loader_inst) / gradient_accumulation_steps)
        pbar = tqdm(total=total_steps, desc=f"Epoch {epoch + 1}/{epochs}", unit="step")
        current_batches_x = []
        current_batches_y = []

        # Encapsulated helper to dispatch steps uniformly and mitigate dry-run copy pastes
        def dispatch_train_step(batches_x, batches_y):
            trainable_params = model.trainable_parameters()
            optim_state = optimizer.state
            ema_shadow = ema.shadow
            
            trainable_params, optim_state, ema_shadow, total_loss, total_aux = _full_train_step(
                trainable_params, optim_state, ema_shadow, batches_x, batches_y
            )
            
            # Update stateful tracking containers outside compilation scopes
            model.update(trainable_params)
            optimizer.state = optim_state
            ema.shadow = ema_shadow

            # Realize deferred evaluation pipelines immediately to flush GPU allocation caches safely
            mx.eval(trainable_params, optimizer.state, ema.shadow, total_loss, total_aux)
            mx.clear_cache()
            return total_loss.item(), total_aux.item()

        for i, (input_ids, targets) in enumerate(train_loader_inst):
            current_batches_x.append(input_ids.astype(mx.int32))
            current_batches_y.append(targets.astype(mx.int32))

            if len(current_batches_x) == gradient_accumulation_steps:
                loss_val, aux_val = dispatch_train_step(current_batches_x, current_batches_y)
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{loss_val:.4f}",
                        "aux": f"{aux_val:.6f}",
                        "lr": f"{optimizer.learning_rate.item():.6f}",
                    }
                )
                current_batches_x = []
                current_batches_y = []

        # Salvage and compute unaligned tail-end micro-batches left over at epoch completion bounds
        if current_batches_x:
            loss_val, aux_val = dispatch_train_step(current_batches_x, current_batches_y)
            pbar.update(1)
            pbar.set_postfix(
                {
                    "loss": f"{loss_val:.4f}",
                    "aux": f"{aux_val:.6f}",
                    "lr": f"{optimizer.learning_rate.item():.6f}",
                    }
            )
            current_batches_x = []
            current_batches_y = []

        pbar.close()

        # Persist standard recovery states safely at checked block-level execution boundaries
        print(f"💾 Saving latest epoch {epoch + 1} checkpoints for session recovery...")
        model.save_weights(latest_model_path)
        mx.save_safetensors(latest_optim_path, optimizer.state)
        
        # Deploy active EMA state tables onto the network tracking instances during evaluations
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

        # Persist standard training checkpoint files
        base, ext = os.path.splitext(save_path)
        if not ext:
            ext = ".safetensors"

        checkpoint_name = f"{base}_epoch{epoch + 1:03d}_val{val_loss:.4f}{ext}"
        model.save_weights(checkpoint_name)
        print(f"Checkpoint saved: {checkpoint_name}")

        # Sample generation task executing on unconstrained context tracking loops
        test_prompt = "Write a polite refusal email"
        test_ids = mx.array([tokenizer.encode(test_prompt)])
        generated = model.generate(test_ids, max_new_tokens=50)
        print(f"Sample: {tokenizer.decode(generated.tolist())[:150]}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights(save_path)
            print(f"Best model updated: {val_loss:.4f}")

        # Revert online baseline targets back to online operational targets
        ema.restore()

    return model
