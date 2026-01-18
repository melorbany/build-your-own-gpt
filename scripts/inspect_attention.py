from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer  # pip install tokenizers

from src.models.attention import AttentionConfig, CausalSelfAttention


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser(
        description="Phase 5.2: inspect Q/K/V (and attention) for a given text using your CausalSelfAttention."
    )
    p.add_argument("--tokenizer-json", type=str, required=True, help="Tokenizer JSON produced by tokenization/train_tokenizer.py")
    p.add_argument("--text", type=str, default="", help="Text to inspect. If empty, read from stdin.")

    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)

    p.add_argument("--head", type=int, default=0, help="Which head to print")
    p.add_argument("--preview", type=int, default=8, help="How many values to print from each vector")
    p.add_argument("--show-attn", action="store_true", help="Also print attention probabilities for the selected head")

    args = p.parse_args()

    text = args.text if args.text else sys.stdin.read().strip()
    if not text:
        print("No text provided. Use --text or pipe stdin.", file=sys.stderr)
        return 2

    if args.d_model % args.n_heads != 0:
        print("Error: --d-model must be divisible by --n-heads", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Load tokenizer artifact
    tok_path = Path(args.tokenizer_json)
    if not tok_path.exists():
        print(f"Tokenizer JSON not found: {tok_path}", file=sys.stderr)
        return 2

    tokenizer = Tokenizer.from_file(str(tok_path))
    enc = tokenizer.encode(text)
    ids = enc.ids
    tokens = enc.tokens
    T = len(ids)
    if T == 0:
        print("Tokenizer produced zero tokens.", file=sys.stderr)
        return 2

    vocab_size = tokenizer.get_vocab_size()

    # Phase 5.2 scaffold: random embedding table
    # (Later, replace with your model's learned embeddings.)
    tok_emb = torch.nn.Embedding(vocab_size, args.d_model).to(device)

    # Your attention module (Phase 5.1)
    cfg = AttentionConfig(d_model=args.d_model, n_heads=args.n_heads, dropout=0.0, bias=True)
    attn = CausalSelfAttention(cfg).to(device).eval()

    ids_t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1,T)
    x = tok_emb(ids_t)  # (1,T,C)

    # ---- Compute Q, K, V using the known .qkv layer from your attention.py ----
    qkv = attn.qkv(x)  # (1,T,3C)
    C = args.d_model
    q, k, v = qkv.split(C, dim=-1)  # each (1,T,C)

    nh = args.n_heads
    hs = C // nh

    # reshape to (T, nh, hs) for easy printing
    q = q.view(1, T, nh, hs)[0]
    k = k.view(1, T, nh, hs)[0]
    v = v.view(1, T, nh, hs)[0]

    # optional: get attention probs from the module (uses its internal causal mask cache)
    att_probs = None
    if args.show_attn:
        _, att = attn(x, return_attn=True)  # (1, nh, T, T)
        att_probs = att[0]  # (nh, T, T)

    head = max(0, min(args.head, nh - 1))
    n = max(1, min(args.preview, hs))

    # ---- Display ----
    print("== Input ==")
    print(f"text: {text!r}")
    print(f"T={T} vocab_size={vocab_size} d_model={C} n_heads={nh} head_dim={hs}\n")

    print("Tokens / IDs:")
    for i, (t, id_) in enumerate(zip(tokens, ids)):
        print(f"  {i:02d}: {t!r} -> id={id_}")

    print("\n== Shapes ==")
    print(f"x (embeddings): {tuple(x.shape)}")
    print(f"q: {tuple(q.shape)}  k: {tuple(k.shape)}  v: {tuple(v.shape)}")

    print(f"\n== Q/K/V detail (head={head}, first {n} values) ==")
    for i, t in enumerate(tokens):
        qi = q[i, head, :n].detach().cpu().tolist()
        ki = k[i, head, :n].detach().cpu().tolist()
        vi = v[i, head, :n].detach().cpu().tolist()
        print(f"\nToken {i:02d} {t!r}")
        print(f"  Q[:{n}] = {qi}")
        print(f"  K[:{n}] = {ki}")
        print(f"  V[:{n}] = {vi}")

    if att_probs is not None:
        print(f"\n== Attention probabilities (head={head}) ==")
        probs = att_probs[head].detach().cpu()  # (T,T)
        for i, t in enumerate(tokens):
            # Future positions should be ~0 because of the causal mask.
            print(f"query {i:02d} {t!r}: {probs[i].tolist()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())