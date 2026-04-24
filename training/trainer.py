import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    aux_loss_coef=0.01, # Coefficient for MoE load balancing
    save_path="textrm-model.pt",
):
    """
    Enhanced training loop for textrm-MoE with deep supervision, 
    EMA, and MoE auxiliary loss integration.
    """

    model = model.to(device)
    # High weight decay for stability in small, recursive models
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    ema = EMA(model, decay=ema_decay)

    # Learning rate scheduler with linear warmup
    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad()

        for i, (input_ids, targets) in enumerate(pbar):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # 1. Forward pass: Calculates Cross-Entropy + Halting Loss
            main_loss = model(input_ids, targets, n_supervision_steps=n_supervision_steps)
            
            # 2. Collect Auxiliary Losses from all MoE layers to ensure load balancing
            # This forces the router to utilize all experts instead of just one.
            total_aux_loss = sum(layer.moe.aux_loss for layer in model.net.layers)
            
            # 3. Combine losses
            loss = main_loss + aux_loss_coef * total_aux_loss
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                # Gradient clipping to prevent explosion in recursive layers
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                ema.update()
                optimizer.zero_grad()

                global_step += 1

            # Update progress bar with both primary and auxiliary loss values
            pbar.set_postfix(
                {
                    "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                    "aux": f"{total_aux_loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                }
            )

        # Validation phase
        ema.apply_shadow()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, targets in tqdm(val_loader, desc="Validation"):
                input_ids = input_ids.to(device)
                targets = targets.to(device)
                # Validation loss also considers MoE balance
                v_main_loss = model(input_ids, targets, n_supervision_steps=n_supervision_steps)
                v_aux_loss = sum(layer.moe.aux_loss for layer in model.net.layers)
                val_loss += (v_main_loss + aux_loss_coef * v_aux_loss).item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}")

        # Checkpoint Saving
        base, ext = os.path.splitext(save_path)
        if ext == "":
            ext = ".pt"
        ckpt_path = f"{base}_epoch{epoch + 1:03d}_val{val_loss:.4f}{ext}"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "ema_shadow": ema.shadow,
                "epoch": epoch,
                "val_loss": val_loss,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")

        # Generate sample to monitor text quality during training
        # We use a static prompt to observe evolution of reasoning/style
        test_prompt = "Write a polite refusal email"
        test_ids = torch.tensor([tokenizer.encode(test_prompt)], device=device)
        generated = model.generate(test_ids, max_new_tokens=50)
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"Sample Generation: {generated_text[:200]}...\n")

        # Persistence of the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "ema_shadow": ema.shadow,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                save_path,
            )
            print(f"New best model saved (val_loss={val_loss:.4f})")

        ema.restore()

    return model
