import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """
    Individual SwiGLU Expert.
    Acts as a specialized processing unit within the MoE layer.
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoELayer(nn.Module):
    """
    Autonomous MoE Layer with Load Balancing.
    Designed for recursive reasoning without manual domain tagging.
    """
    def __init__(self, dim, mlp_ratio=4, num_experts=4, top_k=1, shared_expert=True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Calculate expert dimension to keep total parameters efficient
        num_activated = (1 if shared_expert else 0) + top_k
        expert_hidden_dim = (dim * mlp_ratio) // num_activated

        self.shared_expert = Expert(dim, expert_hidden_dim) if shared_expert else None
        self.experts = nn.ModuleList([
            Expert(dim, expert_hidden_dim) for _ in range(num_experts)
        ])
        
        # Router predicts which experts to use for each token
        self.router = nn.Linear(dim, num_experts, bias=False)
        
        # Placeholder for auxiliary loss (collected during training)
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        """
        Forward pass with dynamic routing and load balancing.
        """
        B, T, C = x.shape
        final_output = torch.zeros_like(x)
        
        # 1. Persistent Shared Expert (captures universal patterns)
        if self.shared_expert:
            final_output += self.shared_expert(x)
            
        # 2. Dynamic Routing Logic
        logits = self.router(x) # [B, T, num_experts]
        
        # Add slight noise during training to encourage exploration (Expert Diversity)
        if self.training:
            noise = torch.randn_like(logits) * 0.01
            logits = logits + noise

        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        
        # Normalize top-k weights to sum to 1
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # 3. Auxiliary Loss Calculation (Load Balancing)
        # Prevents the "Winner-Take-All" problem where only one expert is trained.
        if self.training:
            # density: how much probability mass is assigned to each expert
            density = probs.mean(dim=(0, 1)) 
            # usage: how often an expert is actually selected as top-k
            usage = F.one_hot(top_k_indices.view(-1), self.num_experts).float().mean(0)
            # Ideal case: density and usage are uniform across all experts
            self.aux_loss = (density * usage).sum() * self.num_experts

        # 4. Expert Execution
        # We iterate through experts and apply them to their assigned tokens
        for i, expert in enumerate(self.experts):
            # Create a mask for tokens assigned to this expert
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                # Apply expert only to relevant tokens
                expert_out = expert(x[mask])
                
                # Find which of the top-k slots this expert occupies for each token
                # This is needed to apply the correct routing weight (prob)
                for k in range(self.top_k):
                    k_mask = (top_k_indices[:, :, k] == i)
                    if k_mask.any():
                        # Extract the specific probability for this expert in the top-k
                        w = top_k_probs[k_mask][:, k:k+1]
                        # Add weighted output back to the final result
                        # Using a temporary buffer for scatter-add like behavior
                        final_output[k_mask] += expert(x[k_mask]) * w
                        
        return final_output
