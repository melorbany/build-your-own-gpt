from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import torch


def simple_tokenize(text: str) -> list[str]:
    # Minimal, deterministic tokenizer for debugging only.
    # Later you can swap this with your real tokenizer without changing the QKV inspection logic.
    return text.strip().split()


def build_vocab(tokens: list[str]) -> dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for t in tokens:
        if t not in vocab:
            vocab[t] = len(vocab)
    return vocab


def _find_qkv_linear(attn_module):
    # Common names used in GPT-like implementations
    if hasattr(attn_module, "qkv"):
        return attn_module.qkv
    if hasattr(attn_module, "c_attn"):
        return attn_module.c_attn
    raise AttributeError("Could not find QKV projection on attention module (expected .qkv or .c_attn)")


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect Q,K,V for a text input using CausalSelfAttention.")
    p.add_argument("--text", type=str, default="", help="Text to inspect. If empty, read from stdin.")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)

    p.add_argument("--head", type=int, default=0, help="Which head to print.")
    p.add_argument("--preview", type=int, default=8, help="How many values of each vector to print.")
    p.add_argument("--show-attn", action="store_true", help="Also print attention probabilities per token (chosen head).")

    args = p.parse_args()

    text = args.text if args.text else sys.stdin.read().strip()
    if not text:
        print("No text provided. Use --text or pipe stdin.", file=sys.stderr)
        return 2

    if args.d_model % args.n_heads != 0:
        print("Error: d-model must be divisible by n-heads", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Import attention only (Phase 5.1)
    try:
        from src.models.attention import AttentionConfig, CausalSelfAttention
    except Exception as e:
        print("Import failed: expected src/models/attention.py with AttentionConfig, CausalSelfAttention", file=sys.stderr)
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2

    tokens = simple_tokenize(text)
    vocab = build_vocab(tokens)
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]

    T = len(ids)
    if T == 0:
        print("Tokenization produced zero tokens.", file=sys.stderr)
        return 2

    # Token embeddings (debug-only)
    # Later: replace this embedding with your real model's embedding lookup.
    emb = torch.nn.Embedding(num_embeddings=len(vocab), embedding_dim=args.d_model).to(device)

    x_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1,T)
    x = emb(x_ids)  # (1,T,C)

    cfg = AttentionConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        dropout=0.0,
        bias=True,
    )
    attn = CausalSelfAttention(cfg).to(device).eval()

    # Compute Q,K,V from the module's qkv projection directly
    qkv_linear = _find_qkv_linear(attn)

    with torch.no_grad():
        qkv = qkv_linear(x)  # (1,T,3C)
        B, T, threeC = qkv.shape
        C = threeC // 3
        q, k, v = qkv.split(C, dim=-1)  # each (1,T,C)

    nh = args.n_heads
    hs = args.d_model // nh

    # reshape to (T, nh, hs)
    q = q.view(1, T, nh, hs)[0]
    k = k.view(1, T, nh, hs)[0]
    v = v.view(1, T, nh, hs)[0]

    head = max(0, min(args.head, nh - 1))
    n = max(1, min(args.preview, hs))

    print("== Input ==")
    print(f"text: {text!r}")
    print(f"T={T}  d_model={args.d_model}  n_heads={nh}  head_dim={hs}")
    print("\nTokens:")
    for i, (t, id_) in enumerate(zip(tokens, ids)):
        print(f"  {i:02d}: {t!r} -> id={id_}")

    print(f"\n== Vector previews (head={head}) ==")
    for i, t in enumerate(tokens):
        emb_prev = x[0, i, :n].detach().cpu().tolist()
        q_prev = q[i, head, :n].detach().cpu().tolist()
        k_prev = k[i, head, :n].detach().cpu().tolist()
        v_prev = v[i, head, :n].detach().cpu().tolist()
        print(f"\nToken {i:02d} {t!r}:")
        print(f"  emb[:{n}] = {emb_prev}")
        print(f"  Q[:{n}]   = {q_prev}")
        print(f"  K[:{n}]   = {k_prev}")
        print(f"  V[:{n}]   = {v_prev}")

    if args.show_attn:
        # We can also compute attention probs for the chosen head from Q,K (scaled dot-prod + causal mask)
        with torch.no_grad():
            # (T, hs)
            Q = q[:, head, :]
            K = k[:, head, :]
            scores = (Q @ K.t()) * (hs ** -0.5)  # (T,T)

            causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
            scores = scores.masked_fill(~causal, float("-inf"))
            probs = torch.softmax(scores, dim=-1).detach().cpu()  # (T,T)

        print(f"\n== Attention probs (head={head}) ==")
        for i, tok in enumerate(tokens):
            row = probs[i].tolist()
            # show full row; future positions should be ~0
            print(f"query {i:02d} {tok!r}: {row}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())