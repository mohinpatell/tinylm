"""GPT model. All transformer components implemented from scratch."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Implements the attention mechanism from scratch:
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    The causal mask prevents attending to future positions.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Q, K, V projections for all heads in one matrix multiply
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = nn.Dropout(config.dropout)

        # causal mask: lower triangular matrix
        # register as buffer so it moves to GPU with the model but isn't a parameter
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer('mask', mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape

        # compute Q, K, V for all heads at once
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)  # each is (B, T, C)

        # reshape into heads: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # apply causal mask: set future positions to -inf so softmax gives 0
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        out = attn @ v

        # concatenate heads: (B, n_head, T, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)


if __name__ == '__main__':
    config = GPTConfig()
    attn = CausalSelfAttention(config)

    x = torch.randn(2, 16, config.n_embd)
    out = attn(x)
    print(f'Input:  {x.shape}')
    print(f'Output: {out.shape}')
    print(f'Params: {sum(p.numel() for p in attn.parameters()):,}')
