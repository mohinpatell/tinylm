"""Tests against PyTorch reference implementations."""

import torch
import torch.nn as nn
import math
from model import CausalSelfAttention, GPT
from config import GPTConfig


def test_causal_attention():
    """Check our attention matches nn.MultiheadAttention."""
    torch.manual_seed(42)

    config = GPTConfig(n_embd=64, n_head=4, block_size=32, dropout=0.0)
    our_attn = CausalSelfAttention(config)

    # build a comparable pytorch MHA
    pt_attn = nn.MultiheadAttention(64, 4, dropout=0.0, batch_first=True)

    # copy our weights into pytorch's format
    pt_attn.in_proj_weight.data = our_attn.qkv_proj.weight.data.clone()
    pt_attn.in_proj_bias.data = our_attn.qkv_proj.bias.data.clone()
    pt_attn.out_proj.weight.data = our_attn.out_proj.weight.data.clone()
    pt_attn.out_proj.bias.data = our_attn.out_proj.bias.data.clone()

    x = torch.randn(2, 16, 64)

    # our output
    our_attn.eval()
    our_out, _ = our_attn(x)

    # pytorch output (need causal mask)
    causal_mask = torch.triu(torch.ones(16, 16, dtype=torch.bool), diagonal=1)
    pt_attn.eval()
    pt_out, _ = pt_attn(x, x, x, attn_mask=causal_mask)

    match = torch.allclose(our_out, pt_out, atol=1e-5)
    max_diff = (our_out - pt_out).abs().max().item()
    print(f'Causal attention match: {match} (max diff: {max_diff:.2e})')
    return match


def test_forward_backward():
    """Full forward + backward pass."""
    torch.manual_seed(42)
    config = GPTConfig(n_embd=64, n_head=4, n_layer=2, block_size=32)
    model = GPT(config)

    idx = torch.randint(0, config.vocab_size, (4, 32))
    targets = torch.randint(0, config.vocab_size, (4, 32))

    logits, loss, _ = model(idx, targets)
    loss.backward()

    # check shapes
    assert logits.shape == (4, 32, config.vocab_size)
    assert loss.shape == ()

    # check that gradients exist
    for name, p in model.named_parameters():
        assert p.grad is not None, f'no gradient for {name}'

    print(f'Forward/backward pass OK (loss={loss.item():.4f})')
    return True


def test_weight_tying():
    """Embedding and output head should be the same tensor."""
    config = GPTConfig()
    model = GPT(config)

    assert model.tok_emb.weight is model.head.weight, 'weights not tied!'
    print(f'Weight tying OK (embedding and head share weights)')
    return True


def test_causal_masking():
    """Changing future tokens shouldn't affect earlier positions."""
    torch.manual_seed(42)
    config = GPTConfig(n_embd=64, n_head=4, n_layer=2, block_size=32, dropout=0.0)
    model = GPT(config)
    model.eval()

    idx1 = torch.randint(0, config.vocab_size, (1, 16))
    idx2 = idx1.clone()
    idx2[0, 10:] = torch.randint(0, config.vocab_size, (6,))  # change tokens 10-15

    with torch.no_grad():
        logits1, _, _ = model(idx1)
        logits2, _, _ = model(idx2)

    # positions 0-9 should be identical (can't see tokens 10+)
    early_match = torch.allclose(logits1[0, :10], logits2[0, :10], atol=1e-5)
    # positions 10+ should differ
    late_differ = not torch.allclose(logits1[0, 10:], logits2[0, 10:], atol=1e-5)

    print(f'Causal masking: early positions unaffected={early_match}, later positions changed={late_differ}')
    return early_match and late_differ


if __name__ == '__main__':
    results = []
    results.append(test_causal_attention())
    results.append(test_forward_backward())
    results.append(test_weight_tying())
    results.append(test_causal_masking())

    print(f'\n{sum(results)}/{len(results)} tests passed')
