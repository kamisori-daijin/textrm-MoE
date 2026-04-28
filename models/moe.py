import mlx.core as mx
import mlx.nn as nn

class Expert(nn.Module):
    """
    Individual SwiGLU Expert.
    Acts as a specialized processing unit within the MoE layer.
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # MLX Linear layers default to bias=True, so explicitly set to False
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x):
        # Use nn.silu instead of F.silu, and construct SwiGLU by multiplication
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))

class MoELayer(nn.Module):
    """
    Autonomous MoE Layer with Load Balancing.
    Designed for recursive reasoning without manual domain tagging.
    """
    def __init__(self, dim: int, mlp_ratio: int = 4, num_experts: int = 4, top_k: int = 1, shared_expert: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Calculate expert dimension to keep total parameters efficient
        num_activated = (1 if shared_expert else 0) + top_k
        expert_hidden_dim = (dim * mlp_ratio) // num_activated

        self.shared_expert = Expert(dim, expert_hidden_dim) if shared_expert else None
        
        # In MLX, standard Python lists replace ModuleList
        self.experts = [
            Expert(dim, expert_hidden_dim) for _ in range(num_experts)
        ]
        
        # Router predicts which experts to use for each token
        self.router = nn.Linear(dim, num_experts, bias=False)

    def __call__(self, x, training: bool = True):
        """
        Forward pass with dynamic routing and load balancing.
        Returns a tuple of (output, aux_loss).
        """
        B, T, C = x.shape
        
        # 1. Persistent Shared Expert (captures universal patterns)
        shared_output = self.shared_expert(x) if self.shared_expert else mx.zeros_like(x)
            
        # 2. Dynamic Routing Logic
        logits = self.router(x) # [B, T, num_experts]
        
        # Add slight noise during training to encourage exploration
        if training:
            noise = mx.random.normal(logits.shape) * 0.01
            logits = logits + noise

        probs = mx.softmax(logits, axis=-1)
        
        # Get top_k indices and values
        # In MLX, we use argsort to get indices, then take the last k
        indices_full = mx.argsort(probs, axis=-1)
        top_k_indices = indices_full[..., -self.top_k:]
        
        # Gather the corresponding probabilities
        top_k_probs = mx.take_along_axis(probs, top_k_indices, axis=-1)
        
        # Normalize top-k weights
        top_k_probs = top_k_probs / mx.sum(top_k_probs, axis=-1, keepdims=True)
        
        # 3. Auxiliary Loss Calculation (Load Balancing)
        aux_loss = mx.array(0.0)
        if training:
            # Importance (P): average probability assigned to each expert
            P = mx.mean(probs.reshape(-1, self.num_experts), axis=0)
            
            # Load (f): fraction of tokens routed to each expert (using top-1 for simplicity)
            top1_indices = top_k_indices[..., 0].reshape(-1)
            # Efficiently calculate fraction for each expert
            f = mx.array([mx.mean(top1_indices == i) for i in range(self.num_experts)])
            
            # Calculate Switch Transformer style loss
            aux_loss = self.num_experts * mx.sum(f * P)

        # 4. Expert Execution
        # We can optimize this by only running experts on relevant tokens, 
        # but for a small model, we can use a simpler approach.
        combined_expert_outputs = mx.zeros_like(x)
        
        for k in range(self.top_k):
            k_indices = top_k_indices[:, :, k] # [B, T]
            k_probs = top_k_probs[:, :, k:k+1] # [B, T, 1]
            
            for i, expert in enumerate(self.experts):
                # Mask where this specific expert is selected [B, T, 1]
                expert_mask = mx.expand_dims(k_indices == i, -1)
                
                # Compute only for selected tokens (though MLX might still compute all)
                expert_out = expert(x)
                combined_expert_outputs += expert_out * expert_mask * k_probs
                
        return shared_output + combined_expert_outputs, aux_loss
