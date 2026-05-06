import os

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
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
    optimizer = optim.AdamW(
        learning_rate=lr_schedule, betas=(0.9, 0.95), weight_decay=0.1
    )

    ema = EMA(model, decay=ema_decay)
    state = [model.state, optimizer.state]

    @mx.compile
    def _full_update_step(model_state, optimizer_state, x, y):
        model.update(model_state)
        optimizer.state.update(optimizer_state)

        params = model.trainable_parameters()

        def loss_fn(params):
            model.update(params)
            main, aux = model(
                x, y, n_supervision_steps=n_supervision_steps, training=True
            )
            loss = main + aux_loss_coef * aux
            return loss.reshape([]), aux

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        (loss, aux), grads = loss_and_grad_fn(params)

        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)

        return model.state, optimizer.state, loss, aux

    def train_step(batch_list):
        current_model_state = model.state
        current_optim_state = optimizer.state

        param_dtype = tree_flatten(model.parameters())[0][1].dtype
        total_loss = mx.array(0.0, dtype=param_dtype)
        total_aux = mx.array(0.0, dtype=param_dtype)

        n = len(batch_list)
        for x, y in batch_list:
            x = x.astype(mx.int32)
            y = y.astype(mx.int32)

            current_model_state, current_optim_state, l, a = _full_update_step(
                current_model_state, current_optim_state, x, y
            )
            total_loss = total_loss + l / n
            total_aux = total_aux + a / n

        model.update(current_model_state)
        optimizer.state.update(current_optim_state)

        return total_loss, total_aux

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()

        train_loader_inst = train_loader()
        total_steps = len(train_loader_inst) // gradient_accumulation_steps
        pbar = tqdm(total=total_steps, desc=f"Epoch {epoch + 1}/{epochs}", unit="step")
        current_batches = []

        for i, (input_ids, targets) in enumerate(train_loader_inst):
            current_batches.append((input_ids, targets))

            if len(current_batches) == gradient_accumulation_steps:
                loss, aux = train_step(current_batches)
                mx.eval(state, loss, aux)

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "aux": f"{aux.item():.6f}",
                        "lr": f"{optimizer.learning_rate.item():.6f}",
                    }
                )

                current_batches = []
                mx.clear_cache()

        pbar.close()
        ema.apply_shadow()

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
            print(f"Best model updated: {val_loss:.4f}")

        ema.restore()

    return model
