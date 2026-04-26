import mlx.core as mx
import mlx.nn as nn
from models.trm_build import RMSNorm, TransformerBlock


class TinyRecursiveNetwork(nn.Module):
    def __init__(self, dim, n_heads=8, n_layers=2, mlp_ratio=4, max_seq_len=512, num_experts=8):
        super().__init__()
        self.layers = [
            TransformerBlock(dim, n_heads, mlp_ratio, max_seq_len, num_experts)
            for _ in range(n_layers)
        ]
        
        self.norm = RMSNorm(dim)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TinyRecursiveModel(nn.Module):
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
        num_experts=4,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_latent_recursions = n_latent_recursions
        self.n_improvement_cycles = n_improvement_cycles

        # 1. Embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # 2. Single recursive core
        self.net = TinyRecursiveNetwork(dim, n_heads, n_layers, mlp_ratio, max_seq_len, num_experts)

        # 3. Projections for xyz interaction
        self.combine_xyz = nn.Linear(dim * 3, dim, bias=False)
        self.combine_yz = nn.Linear(dim * 2, dim, bias=False)

        
        self.output_head = nn.Linear(dim, vocab_size, bias=False)
        self.halt_head = nn.Linear(dim, 1, bias=False)

        
        self.y_init = mx.random.normal((1, 1, dim)) * 0.02
        self.z_init = mx.random.normal((1, 1, dim)) * 0.02

        
        self._init_weights()

    def _init_weights(self):
        
        def init_linear_or_emb(path, m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                
                m.weight = mx.random.normal(m.weight.shape) * 0.02
                
        
        self.apply_to_modules(init_linear_or_emb)


    def get_embeddings(self, input_ids):
        B, T = input_ids.shape
        T = min(T, self.max_seq_len)
        
        
        pos = mx.arange(T)[None, :]  # shape: (1, T)
        
        return self.token_emb(input_ids[:, :T]) + self.pos_emb(pos)

    def latent_recursion(self, x, y, z):
        """Update z recursively, then update prediction y."""
        for _ in range(self.n_latent_recursions):
            combined = self.combine_xyz(mx.concatenate([x, y, z], axis=-1))
            z = self.net(combined)
        
        combined_yz = self.combine_yz(mx.concatenate([y, z], axis=-1))
        y = self.net(combined_yz)
        
        return y, z


    def deep_recursion(self, x, y, z, use_grad=True):
        """Perform T cycles of improvement."""
        if not use_grad:
            for _ in range(self.n_improvement_cycles):
                y, z = self.latent_recursion(x, y, z)
            y, z = self.latent_recursion(x, y, z)
            return y, z, self.output_head(y), self.halt_head(mx.mean(y, axis=1))


    import mlx.core as mx
    import mlx.nn as nn
    
    def __call__(self, input_ids, targets=None, n_supervision_steps=4):
        """Forward pass with Deep Supervision and MoE routing."""
        B, T = input_ids.shape
        T = min(T, self.max_seq_len)
        input_ids = input_ids[:, :T]

        x = self.get_embeddings(input_ids)
        
       
        y = mx.broadcast_to(self.y_init, (B, T, self.dim))
        z = mx.broadcast_to(self.z_init, (B, T, self.dim))

        if targets is None:
            # Inference Mode
            y, z = self.deep_recursion(x, y, z, use_grad=False)
            return self.output_head(y)

        # Training Mode with Deep Supervision
        targets = targets[:, :T]
        total_loss = 0.0

        for _ in range(n_supervision_steps):
            y, z, logits, halt_logit = self.deep_recursion(x, y, z, use_grad=True)

            ce_loss_raw = nn.losses.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                reduction="none"
            )
            mask = (targets.reshape(-1) != -100)
            ce_loss = mx.mean(ce_loss_raw * mask)

           
            preds = mx.argmax(logits, axis=-1)
            correct = mx.sum((preds == targets) * mask) / mx.maximum(mx.sum(mask), 1)

            
            halt_loss = nn.losses.binary_cross_entropy(
                mx.squeeze(halt_logit, -1),
                mx.broadcast_to(correct, (B,)),
                with_logits=True
            )

            total_loss = total_loss + ce_loss + 0.1 * halt_loss

        return total_loss / n_supervision_steps


    
    
    def generate(self, input_ids, max_new_tokens=50, temperature=0.8, top_k=40):
        self.eval() 
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -(self.max_seq_len - 1):]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                sorted_logits = mx.sort(logits, axis=-1)
                k_th_value = sorted_logits[:, [-top_k]]
                logits = mx.where(logits < k_th_value, float('-inf'), logits)
                probs = mx.softmax(logits, axis=-1)
                
                next_token = mx.random.categorical(probs, axis=-1)[:, None]
                input_ids = mx.concatenate([input_ids, next_token], axis=1)
        return input_ids

