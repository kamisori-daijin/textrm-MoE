import mlx.core as mx
import mlx.nn as nn
from models.moe import MoELayer
import math

# ============================================================================
# Model Architecture
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        
        self.rms_norm = nn.RMSNorm(dims=dim, eps=eps)

    def __call__(self, x):
        
        return self.rms_norm(x)


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE"""
    def __init__(self, dim, n_heads, max_seq_len=512):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        
       
        self.rope = nn.RoPE(dims=self.head_dim, traditional=True)

        # 2. Creating a Causal Mask
        mask = mx.triu(mx.ones((max_seq_len, max_seq_len)), k=1).astype(mx.bool_)
        self.mask = mask

    def __call__(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # 3. Convert the shape to [B, T, n_heads, head_dim]
    
        q = q.reshape(B, T, self.n_heads, self.head_dim)
        k = k.reshape(B, T, self.n_heads, self.head_dim)
        v = v.reshape(B, T, self.n_heads, self.head_dim)

        # 4. Applying the official RoPE (calculating and multiplying Cos/Sin in one step)
        q = self.rope(q)
        k = self.rope(k)

        # 5. Transpose dimensions for matrix calculations [B, n_heads, T, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Attention calculation
        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(self.head_dim))
        
        # Mask processing
        current_mask = self.mask[:T, :T]
        att = mx.where(current_mask, float('-inf'), att)
        att = mx.softmax(att, axis=-1)

        y = att @ v
        
        # Return to the original shape [B, T, C] and output.
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    """
    Single transformer block using MoE instead of a static MLP.
    Designed for recursive reasoning where routing happens at each step.
    """
    def __init__(self, dim, n_heads, mlp_ratio=4, max_seq_len=512, num_experts=8):
        super().__init__()
        
        self.norm1 = nn.RMSNorm(dims=dim)
        
        self.attn = CausalSelfAttention(dim, n_heads, max_seq_len)
        
        self.norm2 = nn.RMSNorm(dims=dim)
        
        self.moe = MoELayer(
            dim=dim, 
            mlp_ratio=mlp_ratio, 
            num_experts=num_experts, 
            top_k=1, 
            shared_expert=True
        )

    def __call__(self, x, training: bool = True):
        """
        Standard pre-norm residual connection.
        Returns a tuple of (output, aux_loss).
        """
        x = x + self.attn(self.norm1(x))
        
        # MoE sub-layer
        moe_out, aux_loss = self.moe(self.norm2(x), training=training)
        x = x + moe_out
        
        return x, aux_loss
