import os
import torch
from tqdm import tqdm
from ema.ema import EMA

def train(
    model,
    train_loader,
    val_loader,
    tokenizer,
    device,
    epochs=20,
    lr=1e-4,
    warmup_steps=1000,
    n_supervision_steps=4,
    gradient_accumulation_steps=1,
    ema_decay=0.999,
    aux_loss_coef=0.01,
    save_path="textrm-model.pt",
):
    """
    Main training loop. 
    Numerical values are passed from config.py via train.py.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    ema = EMA(model, decay=ema_decay)

    # Scheduler setup
    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    global_step = 0
    best_val_loss = float("inf")

    # Important: Reset gradients before the epoch loop
    optimizer.zero_grad()

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        
        for i, (input_ids, targets) in enumerate(pbar):
            input_ids, targets = input_ids.to(device), targets.to(device)

            # Forward pass
            main_loss = model(input_ids, targets, n_supervision_steps=n_supervision_steps)
            
            # Sum MoE auxiliary losses for all layers
            total_aux_loss = sum(layer.moe.aux_loss for layer in model.net.layers)
            
            # Combine losses and scale by accumulation steps
            loss = (main_loss + aux_loss_coef * total_aux_loss) / gradient_accumulation_steps
            loss.backward()

            # Gradient Accumulation Step
            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                ema.update()
                optimizer.zero_grad() # Crucial: reset after step
                global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                "aux": f"{total_aux_loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}",
            })

        # Validation Logic
        ema.apply_shadow()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, targets in tqdm(val_loader, desc="Validation"):
                input_ids, targets = input_ids.to(device), targets.to(device)
                v_main_loss = model(input_ids, targets, n_supervision_steps=n_supervision_steps)
                v_aux_loss = sum(layer.moe.aux_loss for layer in model.net.layers)
                val_loss += (v_main_loss + aux_loss_coef * v_aux_loss).item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}")

        # Checkpoint Management
        base, ext = os.path.splitext(save_path)
        ckpt_path = f"{base}_epoch{epoch + 1:03d}_val{val_loss:.4f}{ext if ext else '.pt'}"
        torch.save({
            "model_state_dict": model.state_dict(),
            "ema_shadow": ema.shadow,
            "epoch": epoch,
            "val_loss": val_loss,
        }, ckpt_path)

        # Sample Generation
        test_prompt = "Write a polite refusal email"
        test_ids = torch.tensor([tokenizer.encode(test_prompt)], device=device)
        generated = model.generate(test_ids, max_new_tokens=50)
        print(f"Sample: {tokenizer.decode(generated[0].tolist())[:150]}...\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "ema_shadow": ema.shadow,
                "epoch": epoch,
                "val_loss": val_loss,
            }, save_path)
            print(f"Best model updated: {val_loss:.4f}")

        ema.restore()

    return model
