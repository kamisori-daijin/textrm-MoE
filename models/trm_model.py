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

    def __call__(self, x, training: bool = True):
        total_aux_loss = mx.array(0.0)
        for layer in self.layers:
            x, aux_loss = layer(x, training=training)
            total_aux_loss = total_aux_loss + aux_loss
        return self.norm(x), total_aux_loss

class TinyRecursiveModel(nn.Module):
    def __init__(self, vocab_size, dim=256, n_heads=8, n_layers=3, mlp_ratio=4, 
                 max_seq_len=256, n_latent_recursions=6, n_improvement_cycles=3, num_experts=4):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_latent_recursions = n_latent_recursions
        self.n_improvement_cycles = n_improvement_cycles

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.net = TinyRecursiveNetwork(dim, n_heads, n_layers, mlp_ratio, max_seq_len, num_experts)
        
        self.combine_xyz = nn.Linear(dim * 3, dim, bias=False)
        self.combine_yz = nn.Linear(dim * 2, dim, bias=False)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)
        self.halt_head = nn.Linear(dim, 1, bias=False)

        # Learnable Initial State 
        self.y_init = mx.random.normal((1, 1, dim)) * 0.02
        self.z_init = mx.random.normal((1, 1, dim)) * 0.02
        
        self._init_weights()

    def _init_weights(self):
        def init_linear_or_emb(path, m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight = mx.random.normal(m.weight.shape) * 0.02
        self.apply_to_modules(init_linear_or_emb)

    def latent_recursion(self, x, y, z, training: bool = True):
        total_aux_loss = mx.array(0.0)
            
          
        def one_step(x_in, y_in, z_in):
            
            combined = self.combine_xyz(mx.concatenate([x_in, y_in, z_in], axis=-1))
            new_z, aux_z = self.net(combined, training=training)
                
                
            combined_yz = self.combine_yz(mx.concatenate([y_in, new_z], axis=-1))
            new_y, aux_y = self.net(combined_yz, training=training)
                
            return new_y, new_z, aux_z + aux_y
    
         
        for _ in range(self.n_latent_recursions):
               
            y, z, aux = mx.checkpoint(one_step)(x, y, z)
            total_aux_loss = total_aux_loss + aux
                
        return y, z, total_aux_loss

    def deep_recursion(self, x, y, z, training: bool = True):
        total_aux_loss = mx.array(0.0)
        
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
        T = min(T, self.max_seq_len)
        x = self.token_emb(input_ids[:, :T]) + self.pos_emb(mx.arange(T)[None, :])
        
        y = mx.broadcast_to(self.y_init, (B, T, self.dim))
        z = mx.broadcast_to(self.z_init, (B, T, self.dim))

        if targets is None:
            y, z, logits, _, _ = self.deep_recursion(x, y, z, training=False)
            return logits

        total_main_loss = mx.array(0.0)
        total_aux_loss = mx.array(0.0)
        targets = targets[:, :T]

        for _ in range(n_supervision_steps):
            y, z, logits, halt_logit, step_aux = self.deep_recursion(x, y, z, training=training)
            
            # Cross Entropy
            ce_loss = mx.mean(nn.losses.cross_entropy(logits, targets))
            
            # Accuracy-based Halt 
            preds = mx.argmax(logits, axis=-1)
            mask = (targets != -100)
            correct = mx.sum((preds == targets) * mask) / mx.maximum(mx.sum(mask), 1)
            
            target_halt = mx.stop_gradient(mx.broadcast_to(correct, (B,)))
            
            halt_loss = mx.mean(nn.losses.binary_cross_entropy(
                mx.squeeze(halt_logit, -1), target_halt, with_logits=True
            ))

            total_main_loss = total_main_loss + ce_loss + 0.1 * halt_loss
            total_aux_loss = total_aux_loss + step_aux
            
            
            y, z = mx.stop_gradient(y), mx.stop_gradient(z)

        return total_main_loss / n_supervision_steps, total_aux_loss / n_supervision_steps