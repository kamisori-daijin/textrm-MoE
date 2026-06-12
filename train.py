import os
import mlx.core as mx
from dataset.dataset import get_binary_datasets
from models.config import config
from training.instantiate import model, tokenizer
from training.trainer import train

if __name__ == "__main__":
    # 1. Initialize and load binary datasets via dataset factories
    train_loader_factory, val_loader_factory = get_binary_datasets(
        tokenizer=tokenizer,
        max_length=config["max_seq_len"],
        max_documents=config["max_train_samples"] + config["max_val_samples"], 
        max_train_sequences=config["max_train_sequences"], 
        batch_size=config["batch_size"],
    )


    print("Dataset loaded. Now training model...")

    save_path = "best_model.safetensors"
    latest_model_path = "textrm_latest.safetensors"

    # 2. Fault-tolerant model recovery: automatically resume core weights if an epoch checkpoint exists
    if os.path.exists(latest_model_path):
        print(f"🔄 Attempting session recovery: loading model weights from {latest_model_path}...")
        try:
            model.load_weights(latest_model_path)
            print("✅ Core model weights successfully restored.")
        except Exception as e:
            print(f"⚠️ Failed to restore model weights (initializing from default settings): {e}")

    # 3. Launch the monolithic optimized MLX training orchestration
    model = train(
        model=model,
        train_loader=train_loader_factory,
        val_loader=val_loader_factory,
        tokenizer=tokenizer,
        epochs=config["epochs"],
        lr=config["lr"],
        warmup_steps=config["warmup_steps"],
        n_supervision_steps=config["n_supervision_steps"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        save_path=save_path,
    )

    print("\nTraining complete!")

    # 4. Critical Resolution: Enforce a hard reload of the best EMA weights from disk.
    # This discards noisy final-step training weights, allocating the genuine "best model" for final export and sampling.
    if os.path.exists(save_path):
        print(f"🌟 Loading the highest accuracy EMA weights for evaluation and final save: {save_path}")
        model.load_weights(save_path)
        print("✅ Highest accuracy EMA model successfully restored.")

    # Save final weights (this represents the authentic, optimized best EMA model)
    final_path = "final_model.safetensors"
    model.save_weights(final_path)
    print(f"Saved final model to {final_path}")

    # 5. Final multi-domain evaluation and qualitative sampling phase (executed using clean EMA parameters)
    prompts = [
        "Explain why the sky looks blue during the day:",
        "The following is a Python function for binary search:\ndef binary_search(arr, target):",
        "Question: If a cube has 6 faces, how many faces do 3 cubes have in total? Answer:",
        "A formal email to a professor requesting an extension on a deadline:",
    ]

    print("\n=== Final Qualitative Evaluation ===\n")

    model.eval()

    for prompt in prompts:
        # Guarantee dynamic inference tracking by setting target type indices to standard int32 arrays
        prompt_ids = mx.array([tokenizer.encode(prompt)], dtype=mx.int32)
        generated = model.generate(prompt_ids, max_new_tokens=150, temperature=0.8)
        
        # Explicitly extract the first batch element [0] to safely flatten the array prior to decoding
        full_text = tokenizer.decode(generated[0].tolist())

        print(f'Prompt: "{prompt}"')
        print(f"Generated: {full_text}\n")
        print("-" * 50 + "\n")
