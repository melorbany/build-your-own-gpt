import torch

from src.models.gpt import GPT, GPTConfig


def test_gpt_forward_shapes_and_loss():
    torch.manual_seed(0)
    cfg = GPTConfig(
        vocab_size=100,
        block_size=16,
        n_layers=2,
        d_model=32,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
    )
    model = GPT(cfg).eval()

    B, T = 2, 10
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    logits = model(input_ids)
    assert logits.shape == (B, T, cfg.vocab_size)

    logits2, loss = model(input_ids, targets=targets)
    assert logits2.shape == (B, T, cfg.vocab_size)
    assert loss.ndim == 0

    logits3, loss3, attn_list = model(input_ids, targets=targets, return_attn=True)
    assert logits3.shape == (B, T, cfg.vocab_size)
    assert loss3.ndim == 0
    assert len(attn_list) == cfg.n_layers
    assert attn_list[0].shape == (B, cfg.n_heads, T, T)