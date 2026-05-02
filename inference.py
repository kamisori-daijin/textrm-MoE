import mlx.core as mx
from transformers import AutoTokenizer
from models.trm_model import TinyRecursiveModel
from models.config import config

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("./textrm-2.0-tokenizer")
tokenizer.pad_token = tokenizer.eos_token

# Model
model = TinyRecursiveModel(
    vocab_size=config["vocab_size"],
    dim=config["dim"],
    n_heads=config["n_heads"],
    n_layers=config["n_layers"],
    mlp_ratio=config["mlp_ratio"],
    max_seq_len=config["max_seq_len"],
    n_latent_recursions=config["n_latent_recursions"],
    n_improvement_cycles=config["n_improvement_cycles"],
)

# Load trained weights
# In MLX, we use load_weights
try:
    model.load_weights("final_model.safetensors")
    print("Loaded weights from final_model.safetensors")
except:
    try:
        model.load_weights("best_model.safetensors")
        print("Loaded weights from best_model.safetensors")
    except:
        print("Warning: Could not load weights. Using randomized weights.")

def generate_email(prompt, max_new_tokens=100, temperature=0.7):
    prompt_ids = mx.array([tokenizer.encode(prompt)])
    generated = model.generate(prompt_ids, max_new_tokens=max_new_tokens, temperature=temperature)
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


if __name__ == "__main__":
    prompts = [
        "Write a polite refusal email",
        "Write a professional business email.",
        "Write a short formal email.",
        "Write a polite refusal email to a client regarding a budget request.",
    ]

    print("\n=== Generated Emails ===\n")
    for prompt in prompts:
        email = generate_email(prompt)
        print(f'Prompt: "{prompt}"')
        print(f"Email: {email}\n")
        print("-" * 50 + "\n")
