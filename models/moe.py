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
        
            
            target_indices = top_k_indices[:, -1]
            
            
            one_hot_usage = (target_indices[:, None] == mx.arange(self.num_experts))
            
           
            usage = mx.mean(one_hot_usage.astype(mx.float32), axis=0)
            
            
            aux_loss = mx.sum(density * usage) * self.num_experts


        # 4. Expert Execution 
        final_flat_output = mx.zeros_like(x_flat)
        
        # For each expert, extract and calculate only the target tokens
      
        
        for i, expert in enumerate(self.experts):
            
            #is_current_expert = mx.any(top_k_indices == i, axis=-1, keepdims=True)
            
            expert_out = expert(x_flat)
            
            for k in range(self.top_k):
                
                k_mask = (top_k_indices[:, k:k+1] == i)
                
                
                w = top_k_probs[:, k:k+1]
                safe_w = w * k_mask
                
                final_flat_output = final_flat_output + (expert_out * safe_w)
                
        return shared_out + final_flat_output.reshape(B, T, C), aux_loss
