import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """
    Individual SwiGLU Expert.
    Using SwiGLU activation (Silu(W1x) * W3x) followed by W2 projection.
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
    DeepSeek-style Mixture of Experts Layer.
    Combines a persistent shared expert with dynamically routed experts.
    """
    def __init__(self, dim, mlp_ratio=4, num_experts=4, top_k=1, shared_expert=True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Keep compute constant by scaling down expert size
        # Total activated capacity = (shared + top_k) * expert_hidden_dim
        num_activated = (1 if shared_expert else 0) + top_k
        expert_hidden_dim = (dim * mlp_ratio) // num_activated

        # Shared Expert: Always active for all tokens to capture common patterns
        self.shared_expert = Expert(dim, expert_hidden_dim) if shared_expert else None
        
        # Routed Experts: Selected by the router based on token characteristics
        self.experts = nn.ModuleList([
            Expert(dim, expert_hidden_dim) for _ in range(num_experts)
        ])
        
        # Router: Learns to assign tokens to the most relevant experts
        self.router = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x, expert_idx=None):
        """
        Args:
            x: Input tensor [Batch, Seq_len, Dim]
            expert_idx: Optional forced expert index for domain-specific training
        """
        final_output = torch.zeros_like(x)
        
        # 1. Execute Shared Expert
        if self.shared_expert:
            final_output += self.shared_expert(x)
            
        # 2. Execute Routed Experts
        if expert_idx is not None:
            # Training Mode: Force selection of specific experts (e.g., Python expert)
            if isinstance(expert_idx, int):
                final_output += self.experts[expert_idx](x)
            else:
                # Handle heterogeneous batches (multiple domains in one batch)
                for b in range(x.shape[0]):
                    final_output[b] += self.experts[expert_idx[b]](x[b])
        else:
            # Inference Mode: Dynamic routing based on top-k probabilities
            logits = self.router(x)
            probs = F.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
            
            # Renormalize top-k probabilities
            top_k_probs /= top_k_probs.sum(dim=-1, keepdim=True)
            
            # Simple weighted sum of selected experts
            for k in range(self.top_k):
                idx = top_k_indices[:, :, k]
                for e in range(self.num_experts):
                    mask = (idx == e).unsqueeze(-1)
                    if mask.any():
                        # Weighted expert output
                        final_output += mask * self.experts[e](x) * top_k_probs[:, :, k:k+1]
                        
        return final_output
