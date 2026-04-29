import mlx.core as mx
import mlx.nn as nn

class Expert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x):
        # SwiGLU: w2(silu(w1(x)) * w3(x))
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))

class MoELayer(nn.Module):
    def __init__(self, dim, mlp_ratio=4, num_experts=4, top_k=1, shared_expert=True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        num_activated = (1 if shared_expert else 0) + top_k
        expert_hidden_dim = (dim * mlp_ratio) // num_activated

        self.shared_expert = Expert(dim, expert_hidden_dim) if shared_expert else None
        self.experts = [
            Expert(dim, expert_hidden_dim) for _ in range(num_experts)
        ]
        
        self.router = nn.Linear(dim, num_experts, bias=False)

    def __call__(self, x, training: bool = True):
        B, T, C = x.shape
        # Flatten for routing: [B*T, C]
        x_flat = x.reshape(-1, C)
        
        # 1. Persistent Shared Expert
        if self.shared_expert:
            shared_out = self.shared_expert(x)
        else:
            shared_out = mx.zeros_like(x)
            
        # 2. Dynamic Routing
        logits = self.router(x_flat)
        if training:
            # Expert Diversity Noise
            noise = mx.random.normal(logits.shape) * 0.01
            logits = logits + noise

        probs = mx.softmax(logits, axis=-1)
        
        
        all_indices = mx.argsort(probs, axis=-1)
        top_k_indices = all_indices[:, -self.top_k:] # [B*T, top_k]
        
        
        top_k_probs = mx.take_along_axis(probs, top_k_indices, axis=-1)
        
        
        top_k_probs = top_k_probs / mx.sum(top_k_probs, axis=-1, keepdims=True)
        
        # 3. Auxiliary Loss 
        aux_loss = mx.array(0.0)
        if training:
            # density
            density = mx.mean(probs, axis=0)
            # usage: Use one_hot to create a differentiable path
            # Usage calculation represented by the one with the highest probability (the last row) of top_k
            usage = mx.mean(mx.one_hot(top_k_indices[:, -1], self.num_experts), axis=0)
            aux_loss = mx.sum(density * usage) * self.num_experts

        # 4. Expert Execution 
        final_flat_output = mx.zeros_like(x_flat)
        
        # For each expert, extract and calculate only the target tokens
        for i, expert in enumerate(self.experts):
            
            # mask shape: [B*T]
            mask = mx.any(top_k_indices == i, axis=-1)
            
            if mx.any(mask):
                # Extract only the corresponding tokens (Sparse Execution)
                selected_x = x_flat[mask]
                expert_out = expert(selected_x)
                
            
                for k in range(self.top_k):
                    
                    k_mask = (top_k_indices[mask, k] == i)
                    if mx.any(k_mask):
                        
                        w = top_k_probs[mask, k:k+1][k_mask]
                        
                        final_flat_output[mask] += mx.where(
                            mx.expand_dims(k_mask, -1),
                            expert_out * w,
                            mx.zeros_like(expert_out)
                        )

        return shared_out + final_flat_output.reshape(B, T, C), aux_loss
