import torch

from src.models.block import BlockConfig, TransformerBlock


def test_transformer_block_shapes_and_attn_return():
    torch.manual_seed(0)
    cfg = BlockConfig(d_model=32, n_heads=4, d_ff=128, dropout=0.0)
    block = TransformerBlock(cfg).eval()

    B, T, C = 2, 7, cfg.d_model
    x = torch.randn(B, T, C)

    y = block(x)
    assert y.shape == (B, T, C)

    y2, att = block(x, return_attn=True)
    assert y2.shape == (B, T, C)
    assert att.shape == (B, cfg.n_heads, T, T)

    # causality check: future attention should be ~0
    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    leaked = att[..., upper].abs().max().item()
    assert leaked < 1e-6