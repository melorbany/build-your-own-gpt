from __future__ import annotations

import argparse
import numpy as np
from tokenizers import Tokenizer


def deterministic_embedding_table(vocab_size: int, d_model: int, seed: int, dtype: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dtype == "f16":
        return rng.standard_normal((vocab_size, d_model), dtype=np.float32).astype(np.float16)
    if dtype == "f32":
        return rng.standard_normal((vocab_size, d_model), dtype=np.float32)
    raise ValueError("--dtype must be f16 or f32")


def hex_bytes(b: bytes) -> str:
    return " ".join(f"{x:02x}" for x in b)


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 4 tokenizer CLI: text -> tokens + ids (+ static embedding bytes)")
    p.add_argument("--tokenizer", type=str, default="artifacts/tokenizer/tinystories_bpe.json")
    p.add_argument("--text", type=str, default="", help="If omitted, reads from stdin.")
    p.add_argument("--max-tokens", type=int, default=256)

    # static embedding options
    p.add_argument("--with-emb", action="store_true", help="Also show first 16 bytes of a static embedding per token id.")
    p.add_argument("--vocab-size", type=int, default=16000, help="Must match tokenizer vocab size when --with-emb is used.")
    p.add_argument("--d-model", type=int, default=256, help="Embedding dimension when --with-emb is used.")
    p.add_argument("--dtype", type=str, default="f16", choices=["f16", "f32"], help="Embedding dtype.")
    p.add_argument("--seed", type=int, default=12345, help="Seed so embeddings are stable across runs.")
    p.add_argument("--emb-bytes", type=int, default=16, help="How many raw bytes of embedding to show (default 16).")
    args = p.parse_args()

    text = args.text or input("Enter text: ").strip()

    tok = Tokenizer.from_file(args.tokenizer)
    enc = tok.encode(text)

    ids = enc.ids[: args.max_tokens]
    toks = enc.tokens[: args.max_tokens]

    E = None
    if args.with_emb:
        E = deterministic_embedding_table(args.vocab_size, args.d_model, args.seed, args.dtype)

    print(f"Text: {text!r}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Num tokens: {len(ids)} (showing up to {args.max_tokens})")

    if args.with_emb:
        print(f"Static embedding: E[{args.vocab_size}, {args.d_model}] dtype={args.dtype} seed={args.seed}")
        print(f"Showing emb_bytes[:{args.emb_bytes}] as hex\n")
    else:
        print()

    for i, (t, id_) in enumerate(zip(toks, ids)):
        line = f"{i:03d}  token={t!r:<20}  id={id_:>6}"
        if args.with_emb:
            if 0 <= id_ < args.vocab_size:
                b = E[id_].tobytes(order="C")[: args.emb_bytes]
                line += f"  emb_bytes[:{args.emb_bytes}]={hex_bytes(b)}"
            else:
                line += "  emb_bytes=<id out of range for --vocab-size>"
        print(line)


if __name__ == "__main__":
    main()