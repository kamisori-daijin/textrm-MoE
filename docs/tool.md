# Project Structure Documentation

This document describes the `textrm-MoE` repository structure and explains the role of each major file and folder.

## Root files

- `README.md`
  - Main repository overview, feature summary, and usage guidance.

- `requirements.txt`
  - Lists runtime dependencies for the project.

- `LICENSE.txt`
  - Project license agreement.

- `convert_to_safetensors.py`
  - Utility script likely intended to convert saved model weights into the `safetensors` format.

- `inference.py`
  - A simple inference script that loads the tokenizer and model, attempts to load weights from `final_model.safetensors` or `best_model.safetensors`, and generates text samples.

- `train.py`
  - Main training entrypoint that loads datasets, trains the model, saves checkpoints, and performs generation tests after training.

## `models/`

Contains the core model implementation and architecture components.

- `models/config.py`
  - Defines the default model and training configuration dictionary.
  - Also contains a helper function for exporting text datasets into binary token files using Hugging Face streaming datasets.

- `models/moe.py`
  - Implements the Mixture of Experts (MoE) layer.
  - Includes expert network definitions, router logic, top-k routing, shared expert support, and auxiliary routing loss.

- `models/trm_build.py`
  - Builds the transformer block and attention components used by the model.
  - Contains RMS normalization, causal self-attention with RoPE, and `TransformerBlock` that integrates the MoE layer.

- `models/trm_model.py`
  - Defines `TinyRecursiveModel`, the main recursive Transformer architecture.
  - Implements latent recursion, deep recursion, training-time supervision, halt prediction, and autoregressive generation.

## `dataset/`

Handles dataset preparation and data loading.

- `dataset/dataset.py`
  - Defines `MLXBinaryDataLoader`, a binary token data loader backed by `numpy.memmap`.
  - Provides `get_binary_datasets()` to prepare training and validation binary dataset files and return loader factories.

- `dataset/prepare_binary_dataset.py`
  - Exports text data from Hugging Face datasets into a binary `.bin` format.
  - Writes tokenized examples as `uint16` and supports a maximum sample count.

## `training/`

Contains training orchestration and model/ tokenizer initialization.

- `training/instantiate.py`
  - Instantiates the tokenizer and the `TinyRecursiveModel` with configuration values.
  - Saves a custom tokenizer directory and prints model parameter statistics.

- `training/trainer.py`
  - Implements the training loop, optimizer setup, EMA shadow weights, and gradient accumulation.
  - Uses compiled MLX steps to keep the Metal graph compact and separate update logic from loss computation.

## `ema/`

Contains exponential moving average support.

- `ema/ema.py`
  - Defines `EMA`, which maintains a shadow copy of model parameters and supports apply/restore semantics.

## Important architecture notes

- The model is built on the `mlx` framework and targets efficient recursive Transformer training.
- A custom Mixture of Experts layer (`models/moe.py`) provides top-k routing and a persistent shared expert.
- Training uses deep supervision and an auxiliary halt-loss mechanism to improve recursive inference.
- Dataset preparation is handled by converting text datasets into binary token sequences for fast `np.memmap` loading.

## Usage flow

1. `training/instantiate.py` sets up the tokenizer and model.
2. `train.py` uses `dataset/dataset.py` and `training/trainer.py` to train the model.
3. `inference.py` loads the trained model for generation.
4. `convert_to_safetensors.py` is available for converting saved weights if needed.
