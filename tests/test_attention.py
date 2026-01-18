import torch

from src.models.attention import AttentionConfig, CausalSelfAttention


def test_attention_shapes_and_causality():
    torch.manual_seed(0)

    cfg = AttentionConfig(d_model=32, n_heads=4, dropout=0.0, bias=True)
    attn = CausalSelfAttention(cfg).eval()

    B, T, C = 2, 6, cfg.d_model
    x = torch.randn(B, T, C)

    y, a = attn(x, return_attn=True)

    assert y.shape == (B, T, C)
    assert a.shape == (B, cfg.n_heads, T, T)

    # Causality: attention above diagonal should be ~0 after softmax + mask
    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    leaked = a[..., upper].abs().max().item()
    assert leaked < 1e-6