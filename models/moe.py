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

    def __call__(self, x):
        """
        Forward pass with dynamic routing and load balancing.
        Returns a tuple of (output, aux_loss) following MLX conventions.
        """
        B, T, C = x.shape
        final_output = mx.zeros_like(x)
        
        # 1. Persistent Shared Expert (captures universal patterns)
        if self.shared_expert:
            final_output += self.shared_expert(x)
            
        # 2. Dynamic Routing Logic
        logits = self.router(x) # [B, T, num_experts]
        
        # Add slight noise during training to encourage exploration
        if self.training:
            noise = mx.random.normal(logits.shape) * 0.01
            logits = logits + noise

        probs = mx.softmax(logits, axis=-1)
        
        # Get top_k (MLX topk returns both values and indices)
        top_k_probs, top_k_indices = mx.topk(probs, self.top_k, axis=-1)
        
        # Normalize top-k weights
        top_k_probs = top_k_probs / mx.sum(top_k_probs, axis=-1, keepdims=True)
        
        # 3. Auxiliary Loss Calculation (Load Balancing)
        aux_loss = mx.array(0.0)
        if self.training:
            # density: average probability assigned to each expert
            density = mx.mean(probs, axis=(0, 1))
            
            # usage: frequency of an expert being selected as top-k
            # Simulate one_hot in MLX using equality check
            expanded_indices = mx.expand_dims(top_k_indices, -1) # [B, T, top_k, 1]
            expert_range = mx.arange(self.num_experts) # [num_experts]
            
            # Create a one_hot-like mask via broadcasting
            usage_mask = (expanded_indices == expert_range).astype(mx.float32)
            usage = mx.mean(usage_mask, axis=(0, 1, 2))
            
            # Calculate loss
            aux_loss = mx.sum(density * usage) * self.num_experts

        # 4. Expert Execution (MLX optimized version)
        # Avoid complex slicing in the loop and blend all expert results
        combined_expert_outputs = mx.zeros((B, T, C))
        
        # Loop through each slot of top_k
        for k in range(self.top_k):
            k_indices = top_k_indices[:, :, k] # [B, T]
            k_probs = top_k_probs[:, :, k:k+1] # [B, T, 1]
            
            for i, expert in enumerate(self.experts):
                # Mask where this specific expert is selected [B, T, 1]
                expert_mask = mx.expand_dims(k_indices == i, -1).astype(mx.float32)
                
                # Compute and apply mask and routing probability
                expert_out = expert(x)
                combined_expert_outputs += expert_out * expert_mask * k_probs
                
        final_output += combined_expert_outputs
                        
        return final_output, aux_loss
