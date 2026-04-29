import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
from tqdm import tqdm
from ema.ema import EMA

def train(
    model,
    train_loader,
    val_loader,
    tokenizer,
    epochs=20,
    lr=1e-4,
    warmup_steps=1000,
    n_supervision_steps=4,
    gradient_accumulation_steps=1,
    ema_decay=0.999,
    aux_loss_coef=0.01,
    save_path="textrm-model.safetensors",
):
    
    lr_schedule = optim.linear_schedule(0, lr, steps=warmup_steps)
    optimizer = optim.AdamW(learning_rate=lr_schedule, betas=(0.9, 0.95), weight_decay=0.1)
    
    # Initialize EMA 
    
    ema = EMA(model, decay=ema_decay)

    # 2. Defining the loss function (integrating aux_loss calculated on the Model side)
    def loss_fn(model, input_ids, targets):
        main_loss, aux_loss = model(
            input_ids, targets, n_supervision_steps=n_supervision_steps, training=True
        )
        
        return main_loss + aux_loss_coef * aux_loss

    # 3. Gradient calculation and compilation
    grad_fn = nn.value_and_grad(model, loss_fn)

    @mx.compile
    def train_step(batch_list):
        # Complete gradient accumulation within the compile timeframe.
        acc_grads = None
        acc_loss = mx.array(0.0)
        n = len(batch_list)
        
        for input_ids, targets in batch_list:
            loss, grads = grad_fn(model, input_ids, targets)
            acc_loss = acc_loss + (loss / n)
            if acc_grads is None:
                acc_grads = grads
            else:
                # Add gradients for all parameters
                acc_grads = tree_map(lambda g1, g2: g1 + g2, acc_grads, grads)
        
        
        acc_grads, _ = optim.clip_grad_norm(acc_grads, 1.0)
        
        optimizer.update(model, acc_grads)
        return acc_loss

    # ============================================================================
    # Training Loop
    # ============================================================================
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(desc=f"Epoch {epoch + 1}/{epochs}", unit="step")
        
        current_batches = []
        for i, (input_ids, targets) in enumerate(train_loader()):
            current_batches.append((input_ids, targets))
            
            if len(current_batches) == gradient_accumulation_steps:
                
                loss = train_step(current_batches)
                
                
                ema.update()
                
               
                if i % (10 * gradient_accumulation_steps) == 0:
                    mx.eval(model.parameters(), optimizer.state, loss)
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{optimizer.learning_rate.item():.6f}"
                    })
                
                pbar.update(1)
                current_batches = []

        # ============================================================================
        # 4. Validation (Apply EMA)
        # ============================================================================
        pbar.close()
        ema.apply_shadow() 
        
        val_loss, val_steps = 0.0, 0
        for v_input, v_target in val_loader():
            
            v_main, v_aux = model(v_input, v_target, n_supervision_steps=n_supervision_steps, training=False)
            v_step_loss = v_main + aux_loss_coef * v_aux
            val_loss += v_step_loss.item()
            val_steps += 1
            
        val_loss /= max(val_steps, 1)
        print(f"Val Loss: {val_loss:.4f}")

       
       
        base, ext = os.path.splitext(save_path)
        
        if not ext:
            ext = ".safetensors"
                    
        
        checkpoint_name = f"{base}_epoch{epoch + 1:03d}_val{val_loss:.4f}{ext}"
                
       
        model.save_weights(checkpoint_name)
        print(f"Checkpoint saved: {checkpoint_name}")
        
        
        test_prompt = "Write a polite refusal email"
        test_ids = mx.array([tokenizer.encode(test_prompt)])
        generated = model.generate(test_ids, max_new_tokens=50)
        print(f"Sample: {tokenizer.decode(generated[0].tolist())[:150]}\n")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            model.save_weights(save_path)
            print(f"🌟 Best model updated: {val_loss:.4f}")
        
        
        ema.restore()

    return model