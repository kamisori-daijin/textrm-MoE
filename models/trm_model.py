import math
import mlx.core as mx
import mlx.nn as nn
from models.trm_build import RMSNorm, TransformerBlock

class TinyRecursiveNetwork(nn.Module):
    def __init__(self, dim, n_heads=8, n_layers=2, mlp_ratio=4, num_experts=8):
        super().__init__()
        self.layers = [
            TransformerBlock(dim, n_heads, mlp_ratio, num_experts)
            for _ in range(n_layers)
        ]
        self.norm = RMSNorm(dim)

    def __call__(self, x, apply_rope: bool = True, training: bool = True):
        total_aux_loss = mx.array(0.0, dtype=x.dtype)
        for layer in self.layers:
            x, aux_loss = layer(x, apply_rope=apply_rope, training=training)
            total_aux_loss = total_aux_loss + aux_loss
        return self.norm(x), total_aux_loss


class TinyRecursiveModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim=256,
        n_heads=8,
        n_layers=3,
        mlp_ratio=4,
        n_latent_recursions=6,
        n_improvement_cycles=3,
        num_experts=4,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.n_latent_recursions = n_latent_recursions
        self.n_improvement_cycles = n_improvement_cycles

        # Unconstrained token embedding without sequence length limitations
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.net = TinyRecursiveNetwork(
            dim, n_heads, n_layers, mlp_ratio, num_experts
        )

        self.combine_xyz = nn.Linear(dim * 3, dim, bias=False)
        self.combine_yz = nn.Linear(dim * 2, dim, bias=False)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)
        self.halt_head = nn.Linear(dim, 1, bias=False)

        # Scale learnable initial states dynamically based on hidden dimensions
        init_scale = math.sqrt(1.0 / dim)
        self.y_init = mx.random.normal((1, 1, dim)) * init_scale
        self.z_init = mx.random.normal((1, 1, dim)) * init_scale

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using dynamic Xavier/Glorot scaling to prevent numerical instability"""
        def init_linear_or_emb(path, m):
            if isinstance(m, nn.Linear):
                # Compute scale factor dynamically based on fan_in dimension
                fan_in = m.weight.shape[-1]
                scale = math.sqrt(1.0 / fan_in)
                m.weight = mx.random.normal(m.weight.shape) * scale
            elif isinstance(m, nn.Embedding):
                # Standard normal initialization for embedding matrices conformant with typical MLX setups
                m.weight = mx.random.normal(m.weight.shape)

        self.apply_to_modules(init_linear_or_emb)

    def latent_recursion(self, x, y, z, training: bool = True):
        total_aux_loss = mx.array(0.0, dtype=y.dtype)
        
        for i in range(self.n_latent_recursions):
            combined = self.combine_xyz(mx.concatenate([x, y, z], axis=-1))
            
            # Apply RoPE rotation strictly on the first step; subsequent loops operate on stable spatial constraints
            apply_rope = (i == 0)
            
            z, aux = self.net(combined, apply_rope=apply_rope, training=training)
            total_aux_loss = total_aux_loss + aux

        combined_yz = self.combine_yz(mx.concatenate([y, z], axis=-1))
        y, aux = self.net(combined_yz, apply_rope=False, training=training)
        total_aux_loss = total_aux_loss + aux
        return y, z, total_aux_loss

    def deep_recursion(self, x, y, z, training: bool = True):
        total_aux_loss = mx.array(0.0, dtype=y.dtype)

        if not training:
            for _ in range(self.n_improvement_cycles):
                y, z, aux = self.latent_recursion(x, y, z, training=False)
            return y, z, self.output_head(y), self.halt_head(mx.mean(y, axis=1)), total_aux_loss

        for _ in range(self.n_improvement_cycles - 1):
            y, z, aux = self.latent_recursion(x, y, z, training=training)
            y = mx.stop_gradient(y)
            z = mx.stop_gradient(z)
            total_aux_loss = total_aux_loss + aux

        y, z, aux = self.latent_recursion(x, y, z, training=training)
        total_aux_loss = total_aux_loss + aux

        return y, z, self.output_head(y), self.halt_head(mx.mean(y, axis=1)), total_aux_loss

    def __call__(self, input_ids, targets=None, n_supervision_steps=4, training: bool = True):
        B, T = input_ids.shape
        x = self.token_emb(input_ids)

        y = mx.broadcast_to(self.y_init, (B, T, self.dim))
        z = mx.broadcast_to(self.z_init, (B, T, self.dim))

        if targets is None:
            y, z, logits, _, _ = self.deep_recursion(x, y, z, training=False)
            return logits

        param_dtype = self.token_emb.weight.dtype
        total_main_loss = mx.array(0.0, dtype=param_dtype)
        total_aux_loss = mx.array(0.0, dtype=param_dtype)

        for _ in range(n_supervision_steps):
            y, z, logits, halt_logit, step_aux = self.deep_recursion(x, y, z, training=training)

            ce_loss = mx.mean(nn.losses.cross_entropy(logits, targets))

            preds = mx.argmax(logits, axis=-1)
            mask = targets != -100
            correct = mx.sum((preds == targets) * mask) / mx.maximum(mx.sum(mask), 1)

            target_halt = mx.stop_gradient(mx.broadcast_to(correct, (B,)))
            halt_loss = mx.mean(
                nn.losses.binary_cross_entropy(mx.squeeze(halt_logit, -1), target_halt, with_logits=True)
            )

            total_main_loss = total_main_loss + ce_loss + 0.1 * halt_loss
            total_aux_loss = total_aux_loss + step_aux

            y, z = mx.stop_gradient(y), mx.stop_gradient(z)

        return total_main_loss / n_supervision_steps, total_aux_loss / n_supervision_steps
        
    def generate(self, input_ids, max_new_tokens=50, temperature=0.8, top_k=40):
        """Dynamic text generation via standard `mx.topk` filtering over unconstrained context windows"""
        B, T = input_ids.shape
        generated = input_ids
        
        for _ in range(max_new_tokens):
            _, curr_T = generated.shape

            x = self.token_emb(generated)
            y = mx.broadcast_to(self.y_init, (B, curr_T, self.dim))
            z = mx.broadcast_to(self.z_init, (B, curr_T, self.dim))

            y, z, logits, _, _ = self.deep_recursion(x, y, z, training=False)
            next_token_logits = logits[:, -1, :] / temperature
            
            if top_k is not None and top_k > 0:
                k = min(top_k, next_token_logits.shape[-1])
                top_values = mx.topk(next_token_logits, k, axis=-1)
                thresh = mx.min(top_values, axis=-1, keepdims=True)
                next_token_logits = mx.where(next_token_logits < thresh, float("-inf"), next_token_logits)

            next_token = mx.random.categorical(next_token_logits, num_samples=1)
            generated = mx.concatenate([generated, next_token], axis=-1)
                    
            mx.eval(generated)

        return generated