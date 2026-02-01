"""GPT model built from scratch."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer('mask', mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)

        # reshape into (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # append to KV cache if we have one
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
        new_kv_cache = (k, v)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # only mask during training/prefill, not cached generation
        if kv_cache is None:
            attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out), new_kv_cache


class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x, kv_cache=None):
        attn_out, new_kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv_cache


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # scale residual projections to keep variance stable in deep nets
        for block in self.blocks:
            torch.nn.init.normal_(block.attn.out_proj.weight, mean=0.0,
                                  std=0.02 / math.sqrt(2 * self.config.n_layer))
            torch.nn.init.normal_(block.ffn.fc2.weight, mean=0.0,
                                  std=0.02 / math.sqrt(2 * self.config.n_layer))

    def forward(self, idx, targets=None, kv_caches=None):
        B, T = idx.shape

        # position offset when using KV cache
        if kv_caches is not None and kv_caches[0] is not None:
            pos_offset = kv_caches[0][0].shape[2]  # length of cached sequence
        else:
            pos_offset = 0

        assert pos_offset + T <= self.config.block_size, \
            f'seq len {pos_offset + T} exceeds block_size {self.config.block_size}'

        pos = torch.arange(pos_offset, pos_offset + T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, kv_cache=cache)
            new_kv_caches.append(new_cache)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, new_kv_caches


if __name__ == '__main__':
    config = GPTConfig()
    model = GPT(config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'GPT model: {n_params:,} parameters')

    # test forward pass
    idx = torch.randint(0, config.vocab_size, (2, 32))
    targets = torch.randint(0, config.vocab_size, (2, 32))
    logits, loss, caches = model(idx, targets)
    print(f'Input:  {idx.shape}')
    print(f'Logits: {logits.shape}')
    print(f'Loss:   {loss.item():.4f}')

    # test KV cache generation
    single = torch.randint(0, config.vocab_size, (1, 1))
    logits1, _, kv = model(single)
    logits2, _, kv = model(single, kv_caches=kv)
    print(f'KV cache: first token -> cache k shape = {kv[0][0].shape}')
    print(f'KV cache working!')
