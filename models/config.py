config = {
    "vocab_size": 32005,  # TinyLlama(32k) + 5 Special Tokens
    "dim": 1024,  # Hidden dimension
    "n_heads": 16,  # Attention heads
    "n_layers": 4,  # Only 4 layers
    "mlp_ratio": 4,
    "max_seq_len": 512,  
    "n_latent_recursions": 6,  
    "n_improvement_cycles": 3,  
    # Training
    "batch_size": 16,  # Increased from 1 to improve GPU utilization
    "gradient_accumulation_steps": 4, # Total effective batch = 64
    "epochs": 20,
    "lr": 1e-4,
    "warmup_steps": 500,
    "n_supervision_steps": 3,  # Deep supervision steps during training
    "max_train_samples": 100000,  # Reduced for memory and speed
    "max_val_samples": 1000,
}
