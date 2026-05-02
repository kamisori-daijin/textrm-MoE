from transformers import AutoTokenizer
import os
from models.trm_model import TinyRecursiveModel
from models.config import config
from mlx.utils import tree_flatten 


# Tokenizer
model_id = "TinyLlama/TinyLlama_v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

special_tokens_dict = {
    'additional_special_tokens': ['<user>', '<think>', '</think>', '<generate>', '</generate>']
}
tokenizer.add_special_tokens(special_tokens_dict)

tokenizer.pad_token = tokenizer.eos_token



save_dir = "./textrm-2.0-tokenizer"
os.makedirs(save_dir, exist_ok=True)
# Save tokenizer
tokenizer.save_pretrained(save_dir)

print(f"Vocab size (Original): {tokenizer.vocab_size}")
print(f"Vocab size (Added): {len(tokenizer)}")

# Model
model = TinyRecursiveModel(
    vocab_size=config['vocab_size'],
    dim=config['dim'],
    n_heads=config['n_heads'],
    n_layers=config['n_layers'],
    mlp_ratio=config['mlp_ratio'],
    max_seq_len=config['max_seq_len'],
    n_latent_recursions=config['n_latent_recursions'],
    n_improvement_cycles=config['n_improvement_cycles'],
)

# Count parameters
n_params = sum(v.size for _, v in tree_flatten(model.parameters()))

print(f'Model parameters: {n_params:,} ({n_params/1e6:.2f}M)')
print(f'Effective depth per supervision step: {config["n_improvement_cycles"] * (config["n_latent_recursions"] + 1) * config["n_layers"]}')
     
