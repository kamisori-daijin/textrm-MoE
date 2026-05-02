# MLX Tiny Recursive Models with Mixture of Experts (textrm-MoE)

An efficient reimplementation of [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) using the [MLX](https://github.com/ml-explore/mlx) framework, enhanced with **Mixture of Experts (MoE)** and optimized for Apple Silicon.

## Key Features

- **MLX Native**: Built for high-performance inference and training on Apple Silicon.
- **Recursive Latent Reasoning**: Implements the TRM architecture where a single "tiny" network is reused across latent recursions (`n`) and improvement cycles (`T`).
- **Mixture of Experts (MoE)**:
    - Integrated `MoELayer` with Top-k routing.
    - Persistent **Shared Expert** for capturing general knowledge alongside specialized experts.
    - Auxiliary loss for expert load balancing.
- **Adaptive Computation**: Includes a `Halt Head` to learn optimal early-exit or accuracy-based termination.
- **Efficient Binary Data Pipeline**: 
    - Automated pre-tokenization and export to binary format (`.bin`).
    - High-speed data loading using `np.memmap` for zero-copy memory access.
- **Deep Supervision**: Multi-step intermediate losses ensure stable convergence of recursive layers.
- **Modern Architecture**: Uses RoPE (Rotary Positional Embeddings), RMSNorm, and SwiGLU experts.

## Usage

### 1. Setup the Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure the Model

Adjust hyperparameters in `models/config.py`. The defaults are tuned for performance on M-series MacBooks.

```python
config = {
    "vocab_size": 32005,
    "dim": 1024,
    "n_heads": 16,
    "n_layers": 4,
    "n_latent_recursions": 5,
    "n_improvement_cycles": 2,
    "num_experts": 8,
    "max_seq_len": 512,
}
```

### 3. Training

Launch the training script. It will automatically download the required datasets (Cosmopedia, FineWeb, etc.), prepare binary caches, and begin training with EMA (Exponential Moving Average) weights.

```bash
python train.py
```

### 4. Inference

Run generation tests using the trained weights (default: `final_model.safetensors`):

```bash
python inference.py
```

## Dataset & Special Tokens

The model uses a TinyLlama-based tokenizer .

Training data is automatically packed and masked .

## Acknowledgments

- [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) - Original research.
- [gmarchetti2020/TRM-Experiments](https://github.com/gmarchetti2020/TRM-Experiments) - Training insights.                      
- [stockeh/mlx-trm](https://github.com/stockeh/mlx-trm) - Project structure inspiration. 
- [ml-explore/mlx](https://github.com/ml-explore/mlx) - The backbone framework.
- [chaowei312/dsan6650_final](https://github.com/chaowei312/dsan6650_final) - MoE System

---
Created by Kamisori-daijin
