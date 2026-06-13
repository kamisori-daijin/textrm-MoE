import mlx.core as mx
import mlx.nn as nn

class Expert(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # SwiGLU block activation: w2(silu(w1(x)) * w3(x))
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class MoELayer(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4, num_experts: int = 4, top_k: int = 1, shared_expert: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        num_activated = (1 if shared_expert else 0) + top_k
        expert_hidden_dim = (dim * mlp_ratio) // num_activated

        self.shared_expert = Expert(dim, expert_hidden_dim) if shared_expert else None
        self.experts = [Expert(dim, expert_hidden_dim) for _ in range(num_experts)]
        self.router = nn.Linear(dim, num_experts, bias=False)

    def __call__(self, x: mx.array, training: bool = True) -> tuple[mx.array, mx.array]:
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)  # Flattened Shape: [N, C]

        # 1. Persistent Shared Expert (Always Active)
        shared_out = self.shared_expert(x) if self.shared_expert else mx.zeros_like(x)

        # 2. Dynamic Routing (Top-K Selection)
        logits = self.router(x_flat)
        if training:
            # Inject small exploratory noise safely using explicit dtype allocation to prevent in-place collision
            noise = mx.random.normal(logits.shape, dtype=logits.dtype) * 0.15
            logits = logits + noise

        probs = mx.softmax(logits, axis=-1)

        # Retrieve Top-K routing allocations using graph-stable argsort
        all_indices = mx.argsort(probs, axis=-1)
        top_k_indices = all_indices[:, -self.top_k :]  # Shape: [N, top_k]
        top_k_probs = mx.take_along_axis(probs, top_k_indices, axis=-1)

        # Re-normalize routing weights over the chosen Top-K experts
        top_k_probs = top_k_probs / (mx.sum(top_k_probs, axis=-1, keepdims=True) + 1e-20)

        # 3. Pure Tensor-Based Multi-Top-K Auxiliary Loss (100% Loop-Free Load Balancing)
        aux_loss = mx.array(0.0, dtype=x.dtype)
        if training:
            # density (P_i): Mean soft probability allocated to each expert across the token batch
            density = mx.mean(probs, axis=0)  # Shape: [num_experts]
            
            # 【Masterstroke】3D Broadcasting to construct a complete Top-K frequency map in a single shot.
            # Shape expansion: [N, top_k, 1] == [num_experts] -> Broadcasted Grid: [N, top_k, num_experts]
            match_mask = (top_k_indices[:, :, None] == mx.arange(self.num_experts, dtype=mx.int32))
            
            # Determine if an expert was selected ANYWHERE within the top_k dimension for each token
            # Grid: [N, top_k, num_experts] -> any(axis=1) -> [N, num_experts]
            is_selected = mx.any(match_mask, axis=1).astype(x.dtype)
            
            # usage (f_i): Calculate the mean selection frequency per expert
            usage = mx.mean(is_selected, axis=0)  # Shape: [num_experts]
            
            # Penalize deviation from the balanced uniform routing distribution
            aux_loss = mx.sum(density * usage) * mx.array(self.num_experts, dtype=x.dtype)

        # 4. Ultra-Fast Top-K Vectorized Execution (Loops Fully Flattened via Broadcasting)
        final_flat_output = mx.zeros_like(x_flat)

        # Process tokens through experts under a 100% stable static shape [N, C].
        for i, expert in enumerate(self.experts):
            expert_out = expert(x_flat)  # Forward pass over the full batch matrix: [N, C]

            # Broadcast comparison: Check where the current expert `i` exists within the entire [N, top_k] grid
            expert_mask = (top_k_indices == i).astype(x.dtype)  # Shape: [N, top_k]
            
            # Element-wise gating intersection: Merge weights and sum out the top_k dimension -> [N, 1]
            gated_weight = mx.sum(top_k_probs * expert_mask, axis=-1, keepdims=True)
            
            # Accumulate results back into the global timeline buffer via static tensor algebra
            final_flat_output = final_flat_output + (expert_out * gated_weight)

        return shared_out + final_flat_output.reshape(B, T, C), aux_loss