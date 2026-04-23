import torch
import torch.nn as nn
import torch.nn.functional as F
from models import trm_build
from models.trm_build import RMSNorm, TransformerBlock

class TinyRecursiveNetwork(nn.Module):
    """
    Core recursive unit. 
    Processes the latent state z using a small stack of Transformer blocks.
    """
    def __init__(self, dim, n_heads=8, n_layers=2, mlp_ratio=4, max_seq_len=512, num_experts=4):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_ratio, max_seq_len, num_experts)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)

    def forward(self, x, expert_idx=None):
        for layer in self.layers:
            # Pass expert_idx down to each TransformerBlock -> MoELayer
            x = layer(x, expert_idx=expert_idx)
        return self.norm(x)

class TinyRecursiveModel(nn.Module):
    """
    Tiny Recursive Model with MoE (textrm-MoE).
    Refines latent representations through multiple cycles of recursive reasoning.
    """
    def __init__(
        self,
        vocab_size,
        dim=256,
        n_heads=8,
        n_layers=3,
        mlp_ratio=4,
        max_seq_len=256,
        n_latent_recursions=6,
        n_improvement_cycles=3,
        num_experts=4
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_latent_recursions = n_latent_recursions
        self.n_improvement_cycles = n_improvement_cycles

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # Updated to support multiple experts
        self.net = TinyRecursiveNetwork(dim, n_heads, n_layers, mlp_ratio, max_seq_len, num_experts)

        self.combine_xyz = nn.Linear(dim * 3, dim, bias=False)
        self.combine_yz = nn.Linear(dim * 2, dim, bias=False)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)
        self.halt_head = nn.Linear(dim, 1, bias=False)

        self.y_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_embeddings(self, input_ids):
        B, T = input_ids.shape
        T = min(T, self.max_seq_len)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        return self.token_emb(input_ids[:, :T]) + self.pos_emb(pos)

    def latent_recursion(self, x, y, z, expert_idx=None):
        """Update z 'n' times, then refine y once."""
        # 1. Update latent state z
        for _ in range(self.n_latent_recursions):
            combined = self.combine_xyz(torch.cat([x, y, z], dim=-1))
            z = self.net(combined, expert_idx=expert_idx)

        # 2. Refine prediction y
        combined_yz = self.combine_yz(torch.cat([y, z], dim=-1))
        y = self.net(combined_yz, expert_idx=expert_idx)

        return y, z

    def deep_recursion(self, x, y, z, use_grad=True, expert_idx=None):
        """Improve y and z through T cycles."""
        if not use_grad:
            with torch.no_grad():
                for _ in range(self.n_improvement_cycles):
                    y, z = self.latent_recursion(x, y, z, expert_idx=expert_idx)
            return y.detach(), z.detach()

        with torch.no_grad():
            for _ in range(self.n_improvement_cycles - 1):
                y, z = self.latent_recursion(x, y, z, expert_idx=expert_idx)

        y, z = self.latent_recursion(x, y, z, expert_idx=expert_idx)
        return y, z, self.output_head(y), self.halt_head(y.mean(dim=1))

    def forward(self, input_ids, targets=None, n_supervision_steps=4, expert_idx=None):
        """Main forward pass with support for expert routing/forcing."""
        B, T = input_ids.shape
        T = min(T, self.max_seq_len)
        input_ids = input_ids[:, :T]

        x = self.get_embeddings(input_ids)
        y = self.y_init.expand(B, T, -1).clone()
        z = self.z_init.expand(B, T, -1).clone()

        if targets is None:
            # Inference: Recursive reasoning
            y, z = self.deep_recursion(x, y, z, use_grad=False, expert_idx=expert_idx)
            return self.output_head(y)

        # Training with Deep Supervision
        targets = targets[:, :T]
        total_loss = 0.0

        for _ in range(n_supervision_steps):
            y, z, logits, halt_logit = self.deep_recursion(x, y, z, use_grad=True, expert_idx=expert_idx)

            ce_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100
            )
            
            # Simple ACT-based halting loss
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = (targets != -100)
                correct = ((preds == targets) & mask).float().sum() / mask.float().sum().clamp(min=1)
            
            halt_loss = F.binary_cross_entropy_with_logits(
                halt_logit.squeeze(-1),
                correct.expand(B)
            )
            total_loss = total_loss + ce_loss + 0.1 * halt_loss

        return total_loss / n_supervision_steps
