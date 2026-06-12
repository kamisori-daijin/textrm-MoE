import os
from mlx.utils import tree_flatten
from transformers import AutoTokenizer

from models.config import config
from models.trm_model import TinyRecursiveModel

# 1. Initialize and Expand Tokenizer Custom Vocabularies
model_id = "TinyLlama/TinyLlama_v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Inject reasoning-specific control tags directly to govern system behaviors smoothly
special_tokens_dict = {
    "additional_special_tokens": [
        "<user>",
        "<think>",
        "</think>",
        "<generate>",
        "</generate>",
    ]
}
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token = tokenizer.eos_token

# Persist custom tokenizer configurations locally for seamless downstream distribution
save_dir = "./textrm-2.0-tokenizer"
os.makedirs(save_dir, exist_ok=True)
tokenizer.save_pretrained(save_dir)

print(f"Vocab size (Original): {tokenizer.vocab_size}")
print(f"Vocab size (Expanded): {len(tokenizer)}")

# 2. Instantiate TinyRecursiveModel Conforming to the Streamlined Dynamic Configurations
# CRITICAL FIX: Completely omitted `max_seq_len` as the core attention architecture has evolved to a fully fluid design.
model = TinyRecursiveModel(
    vocab_size=len(tokenizer), 
    dim=config["dim"],
    n_heads=config["n_heads"],
    n_layers=config["n_layers"],
    mlp_ratio=config["mlp_ratio"],
    n_latent_recursions=config["n_latent_recursions"],
    n_improvement_cycles=config["n_improvement_cycles"],
    num_experts=config["num_experts"],
)

# 3. Compute Operational Metrics and Architecture Properties
# Use `tree_flatten` dynamically over `model.trainable_parameters()` to gauge baseline parameters cleanly
n_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
first_param = tree_flatten(model.trainable_parameters())[0][1]

print(f"Model parameters: {n_params:,} ({n_params / 1e6:.2f}M)")
print(f"Model dtype: {first_param.dtype}")
print(
    f"Effective depth per supervision step: {config['n_improvement_cycles'] * (config['n_latent_recursions'] + 1) * config['n_layers']}"
)
