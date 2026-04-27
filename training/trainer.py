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
    """
    Main training loop optimized for MLX.
    """
    
    # 1. Learning Rate Schedule Function
    
    def lr_schedule(step):
        warmup_val = mx.where(
            step < warmup_steps, 
            step / float(warmup_steps), 
            1.0
        )
        return lr * warmup_val

    # 2. Setup Optimizer (AdamW)
    optimizer = optim.AdamW(
        learning_rate=lr_schedule,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # 3. Setup EMA
    ema = EMA(model, decay=ema_decay)

    # 4. Pure Loss Function for MLX Autograd
    def loss_fn(model, input_ids, targets):
        main_loss, total_aux_loss = model(
            input_ids, targets, n_supervision_steps=n_supervision_steps
        )
        total_loss = main_loss + aux_loss_coef * total_aux_loss
        return total_loss

    # Create the gradient function
    grad_fn = nn.value_and_grad(model, loss_fn)

    # 5. Compiled Train Step

    @mx.compile
    def train_step(model, optimizer, batch_list):
        
        accumulated_grads = None
        total_loss = 0.0
        
        
        for input_ids, targets in batch_list:
            loss, grads = grad_fn(model, input_ids, targets)
            loss = loss / len(batch_list)
            
            
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(
                    lambda g1, g2: g1 + g2, accumulated_grads, grads
                )
            total_loss += loss

        
        optimizer.update(model, accumulated_grads)
        
        return total_loss

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}")
        
        
        current_batches = []
        
        for i, (input_ids, targets) in enumerate(train_loader):
            current_batches.append((input_ids, targets))
            
            
            if len(current_batches) == gradient_accumulation_steps or (i + 1) == len(train_loader):
                
                loss = train_step(model, optimizer, current_batches)
                
                
                ema.update()
                
                
                mx.eval(model.parameters(), optimizer.state)
                
                
                pbar.update(len(current_batches))
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{lr_schedule(optimizer.state['step']).item():.6f}",
                })
                
               
                current_batches = []
                global_step += 1

        pbar.close()

        # 6. Validation Logic
        ema.apply_shadow()
        model.eval()
        val_loss = 0.0
        
        for input_ids, targets in tqdm(val_loader, desc="Validation"):
            v_main_loss, v_aux_loss = model(
                input_ids, targets, n_supervision_steps=n_supervision_steps
            )
            step_val_loss = v_main_loss + aux_loss_coef * v_aux_loss
            val_loss += step_val_loss.item()
            
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}")

        # 7. Checkpoint Management
        base, ext = os.path.splitext(save_path)
        ckpt_path = f"{base}_epoch{epoch + 1:03d}_val{val_loss:.4f}.safetensors"
        model.save_weights(ckpt_path)
        
        # 8. Sample Generation
        test_prompt = "Write a polite refusal email"
        test_ids = mx.array([tokenizer.encode(test_prompt)])
        generated = model.generate(test_ids, max_new_tokens=50)
        
        generated_list = generated[0].tolist()
        print(f"Sample: {tokenizer.decode(generated_list)[:150]}...\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights(save_path)
            print(f"Best model updated: {val_loss:.4f}")

        # Restore original weights from EMA
        ema.restore()

    return model
