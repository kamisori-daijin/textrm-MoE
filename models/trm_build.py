import mlx.core as mx
import mlx.nn as nn
from models.moe import MoELayer

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.rms_norm = nn.RMSNorm(dims=dim, eps=eps)

    def __call__(self, x):
        return self.rms_norm(x)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE (Fully Dynamic & Recursion-Safe)"""
    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = nn.RoPE(dims=self.head_dim, traditional=True)

    def __call__(self, x, apply_rope: bool = True):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Reshape and transpose to [B, H, T, D] format
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE rotation strictly on the initial recursion step to prevent spatial coordinate distortion
        if apply_rope:
            q = self.rope(q)
            k = self.rope(k)

        # Invoke the optimized fast SDPA kernel conforming to official MLX specifications
        scale = self.head_dim ** -0.5
        y = mx.fast.scaled_dot_product_attention(
            q, k, v, 
            scale=scale, 
            mask="causal"
        )
        
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, num_experts=8):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        
        self.moe = MoELayer(
            dim=dim, 
            mlp_ratio=mlp_ratio, 
            num_experts=num_experts, 
            top_k=1, 
            shared_expert=True
        )

    def __call__(self, x, apply_rope: bool = True, training: bool = True):
        # Forward the positional embedding condition down to the self-attention layer
        x = x + self.attn(self.norm1(x), apply_rope=apply_rope)
        moe_out, aux_loss = self.moe(self.norm2(x), training=training)
        x = x + moe_out
        return x, aux_loss
