import mlx.core as mx
import mlx.nn as nn

class Expert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class MoELayer(nn.Module):
    def __init__(self, dim, mlp_ratio=4, num_experts=4, top_k=1, shared_expert=True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        num_activated = (1 if shared_expert else 0) + top_k
        expert_hidden_dim = (dim * mlp_ratio) // num_activated

        self.shared_expert = Expert(dim, expert_hidden_dim) if shared_expert else None
        self.experts = [Expert(dim, expert_hidden_dim) for _ in range(num_experts)]
        self.router = nn.Linear(dim, num_experts, bias=False)

    def __call__(self, x, training: bool = True):
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)
        N = x_flat.shape[0]  # Total number of tokens (B * T)

        # 1. Shared Expert processing (executed for all tokens)
        if self.shared_expert:
            shared_out = self.shared_expert(x)
        else:
            shared_out = mx.zeros_like(x)

        # 2. Dynamic Routing (Gate computation)
        logits = self.router(x_flat)
        if training:
            noise = mx.random.normal(logits.shape) * 0.01
            logits = logits + noise

        probs = mx.softmax(logits, axis=-1)

        # Retrieve top_k indices and their corresponding probabilities
        all_indices = mx.argsort(probs, axis=-1)
        top_k_indices = all_indices[:, -self.top_k :]  # Shape: [N, top_k]
        top_k_probs = mx.take_along_axis(probs, top_k_indices, axis=-1)
        top_k_probs = top_k_probs / mx.sum(top_k_probs, axis=-1, keepdims=True)

        # 3. Auxiliary Loss computation
        aux_loss = mx.array(0.0, dtype=x.dtype)
        if training:
            density = mx.mean(probs, axis=0)
            target_indices = top_k_indices[:, -1]
            one_hot_usage = target_indices[:, None] == mx.arange(self.num_experts)
            usage = mx.mean(one_hot_usage.astype(x.dtype), axis=0)
            aux_loss = mx.sum(density * usage) * mx.array(self.num_experts, dtype=x.dtype)

        # 4. Optimized Dynamic Expert Execution via Token Gathering
        final_flat_output = mx.zeros_like(x_flat)

        # Iterate through each top_k slot (from k=0 to top_k-1)
        for k in range(self.top_k):
            # Target expert IDs and their routing weights for the current slot
            expert_ids = top_k_indices[:, k]  # Shape: [N]
            weights = top_k_probs[:, k : k + 1]  # Shape: [N, 1]

            for idx, expert in enumerate(self.experts):
                # Binary mask indicating tokens assigned to the current expert [N]
                mask = (expert_ids == idx)
                
                # Early exit: Skip the expert completely if no tokens are assigned
                if not mx.any(mask).item():
                    continue

                # CRITICAL: Gather only the assigned token embeddings.
                # MLX index-slicing ([mask]) compiles efficiently into a gather operation,
                # shrinking the tensor to [num_routed_tokens, C].
                expert_in = x_flat[mask]
                
                # Execute the expert forward pass using only the gathered tokens (massive FLOPs reduction)
                expert_out = expert(expert_in)
                
                # Apply the corresponding routing weights
                expert_out = expert_out * weights[mask]
                
                # Scatter and accumulate results back into the original token coordinates
                final_flat_output[mask] += expert_out

        return shared_out + final_flat_output.reshape(B, T, C), aux_loss