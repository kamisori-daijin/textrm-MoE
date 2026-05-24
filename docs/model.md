# Model Architecture Documentation

This document explains the `textrm-MoE` model architecture and its main components.

## Overview

The model combines a recursive Transformer architecture with a Mixture of Experts (MoE) layer. It is implemented in the `mlx` framework and is designed for efficient latent recursion, deep supervision, and autoregressive generation.

The main model class is `models.trm_model.TinyRecursiveModel`.

## Architecture diagrams

### Overall model flow

```text
input_ids ---> token_emb + pos_emb ---> x
                       |                \
                       v                 \
                y_init, z_init (learned)  \
                       |                  \
                       v                   \
                   deep_recursion(x, y, z) ---> output_head ---> logits
                                            ---> halt_head ---> halt_score
```

### Recursive cycle

```text
deep_recursion:
  for improvement_cycle in 1..n_improvement_cycles:
    latent_recursion:
      for recursion_step in 1..n_latent_recursions:
        combined = combine_xyz([x, y, z])
        z, aux = net(combined)
      combined_yz = combine_yz([y, z])
      y, aux = net(combined_yz)
```

### Structural block

```text
TransformerBlock:
  x -> norm1 -> CausalSelfAttention -> + residual -> x_attn
  x_attn -> norm2 -> MoELayer -> + residual -> output

MoELayer:
  input -> shared_expert -> shared_out
        -> router -> top-k expert routing -> expert_out
        -> sum(shared_out, expert_out)
```

## Core components

### `TinyRecursiveModel`

This is the primary model class.

Key responsibilities:
- token and positional embedding
- recurrent latent state initialization
- repeated latent recursion and improvement cycles
- deep supervision during training
- output projection and halt prediction
- autoregressive generation

Main fields:
- `token_emb`: token embedding layer
- `pos_emb`: positional embedding layer
- `net`: `TinyRecursiveNetwork` that contains the structural Transformer blocks
- `combine_xyz`: linear layer for merging input and latent states
- `combine_yz`: linear layer for refining latent state representation
- `output_head`: linear layer projecting hidden states to vocabulary logits
- `halt_head`: linear layer predicting a halt confidence score
- `y_init`, `z_init`: learnable initial latent states

### Forward computation

`__call__(input_ids, targets=None, n_supervision_steps=4, training=True)`

- Embeds tokens and position encodings.
- Broadcasts the learnable initial hidden states `y_init` and `z_init` across the batch and sequence length.
- If `targets` is absent, runs `deep_recursion` in inference mode and returns logits.
- If `targets` is present, performs repeated supervision steps and computes:
  - cross-entropy loss over logits
  - auxiliary halt loss using confidence from `halt_head`
  - averaged main and auxiliary losses over `n_supervision_steps`

### Recursive mechanisms

#### `latent_recursion(x, y, z, training=True)`

- Repeatedly refines latent states `y` and `z` for `n_latent_recursions` iterations.
- Combines current input `x`, latent state `y`, and latent state `z` through `combine_xyz`.
- Passes the combined state through `self.net`, accumulating auxiliary losses from the MoE blocks.
- Refines `y` again by combining `y` and `z` with `combine_yz`.

#### `deep_recursion(x, y, z, training=True)`

- Runs multiple improvement cycles (`n_improvement_cycles`).
- In inference mode, executes latent recursion repeatedly and returns final logits and halt logits.
- In training mode, stops gradient on earlier cycles to prevent overly deep gradient graphs while still learning long-term refinement.

## Structural network

### `TinyRecursiveNetwork`

This module stacks `TransformerBlock` layers and applies RMS normalization to the final output.

- `layers`: list of `TransformerBlock` objects
- `norm`: RMS normalization layer

Output:
- normalized hidden state
- summed auxiliary loss from each layer

### `TransformerBlock`

A single structural block with two sub-layers:
1. Causal self-attention with RMS normalization
2. MoE feed-forward layer with residual connection

Returned values:
- output hidden state after residual updates
- auxiliary loss from the MoE layer

## Attention and MoE details

### `CausalSelfAttention`

Implements multi-head causal self-attention with Rotary Positional Embedding (RoPE).

- Projects input into query, key, and value tensors.
- Applies RoPE to queries and keys.
- Uses a causal mask to prevent future token attention.
- Computes attention weights and applies them to values.
- Projects the concatenated output back to the model dimension.

### `MoELayer`

Implements a mixture-of-experts block with optional shared expert support.

Key features:
- `num_experts`: number of specialized expert networks
- `top_k`: number of experts selected per token
- `shared_expert`: optional always-active expert for global representation
- `router`: dense projection from token features to expert logits
- auxiliary routing loss to encourage expert utilization

Routing process:
- Convert input to flattened token features.
- Compute expert selection logits and optionally add noise during training.
- Derive soft expert probabilities using softmax.
- Select top-k expert indices per token.
- Normalize selected expert probabilities.
- Compute expert outputs and combine them with selected weights.
- Add the shared expert output if enabled.

### `Expert`

A single expert network with SwiGLU activation.

- `w1`, `w2`, `w3`: linear layers
- Output formula: `w2(silu(w1(x)) * w3(x))`

## Configuration and parameterization

Model hyperparameters are defined in `models/config.py` and include:
- `vocab_size`
- `dim`
- `n_heads`
- `n_layers`
- `mlp_ratio`
- `num_experts`
- `max_seq_len`
- `n_latent_recursions`
- `n_improvement_cycles`

These values determine the width, depth, and recursion behavior of the model.

## Training vs inference

- Training uses `TinyRecursiveModel.__call__` with targets and `n_supervision_steps`.
- Deep recursion is repeated and previous recursions are stopped from backpropagating gradients until the final cycle.
- The model computes both a main cross-entropy loss and a halt prediction loss.
- Inference uses `generate()`, which autoregressively extends text by repeatedly running the recursive model on the current prefix.

### `generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=40)`

- Generates new tokens one step at a time.
- Recomputes latent recursion for each prefix window.
- Applies top-k filtering to the logits before sampling.
- Concatenates the sampled token to the generated sequence.

## Remarks

- The architecture is intentionally recursive: the same structural network is reused across latent recursions and improvement cycles.
- The MoE layer adds conditional capacity by selecting expert sub-networks dynamically for each token.
- The model uses learned initial latent states to bootstrap the recursive reasoning process.
- The halt head provides an auxiliary signal that encourages the model to judge prediction quality at each cycle.
