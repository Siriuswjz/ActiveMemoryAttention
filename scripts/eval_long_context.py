"""Position-conditional perplexity on long held-out documents.

For each document, the model does ONE forward pass over the full sequence
(packed to max_seq_length). We then bucket per-token cross-entropy by
token position and compare buckets between baseline and Active Memory.

The long-range memory claim predicts: AM's advantage (Δ loss) should be
larger in later position buckets, because those tokens depend on more
distant earlier context.

Usage:
  python scripts/eval_long_context.py \
    --baseline_path checkpoints/pretrain_main_100m/final \
    --active_memory_path checkpoints/pretrain_main_100m_active_memory/final
"""

import os
import sys
import argparse
import math

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, AutoConfig, AutoModelForCausalLM

from configuration_deepseek_v4 import DeepseekV4Config
from modeling_deepseek_v4 import DeepseekV4ForCausalLM
from active_memory_attention import replace_attention_with_active_memory

AutoConfig.register("deepseek_v4", DeepseekV4Config, exist_ok=True)
AutoModelForCausalLM.register(DeepseekV4Config, DeepseekV4ForCausalLM, exist_ok=True)


def load_model(path, active_memory, device, dtype=torch.float32):
    config = DeepseekV4Config.from_pretrained(path)
    model = DeepseekV4ForCausalLM(config)
    if active_memory:
        replace_attention_with_active_memory(model)
    state_dict = load_file(os.path.join(path, "model.safetensors"))
    state_dict = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
                  for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    return model.to(device).to(dtype).eval()


def collect_long_docs(tokenizer, n_target_docs, min_tokens, skip_docs):
    """Stream FineWeb-Edu, keep docs whose tokenization is >= min_tokens."""
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    ds = ds.skip(skip_docs)
    docs = []
    scanned = 0
    for row in ds:
        scanned += 1
        ids = tokenizer.encode(row["text"], add_special_tokens=False)
        if len(ids) >= min_tokens:
            docs.append(ids[:min_tokens])  # truncate to common length
            if len(docs) >= n_target_docs:
                break
        if scanned > n_target_docs * 200:  # safety cap
            break
    return docs, scanned


@torch.no_grad()
def per_position_losses(model, doc_ids, device):
    """For one document (list of ids), return a 1D tensor of per-token CE losses
    of length L-1, where position i is the loss of predicting token i+1 from
    context [0..i]."""
    ids = torch.tensor([doc_ids], dtype=torch.long, device=device)
    out = model(input_ids=ids)
    logits = out.logits[0].float()                # (L, V)
    targets = ids[0, 1:]                          # (L-1,)
    shift_logits = logits[:-1]                    # (L-1, V)
    loss = F.cross_entropy(shift_logits, targets, reduction="none")  # (L-1,)
    return loss.detach().cpu()


def bucket_aggregate(per_token_losses, bucket_edges):
    """per_token_losses: list of 1D tensors, each length L-1.
    Returns dict bucket_idx -> (sum_loss, n_tokens)."""
    n_buckets = len(bucket_edges) - 1
    sums = [0.0] * n_buckets
    counts = [0] * n_buckets
    for losses in per_token_losses:
        L = losses.shape[0]
        for b in range(n_buckets):
            lo, hi = bucket_edges[b], min(bucket_edges[b + 1], L)
            if lo >= L:
                break
            chunk = losses[lo:hi]
            sums[b] += chunk.sum().item()
            counts[b] += chunk.numel()
    return sums, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_path", required=True)
    parser.add_argument("--active_memory_path", required=True)
    parser.add_argument("--tokenizer_path", default="tokenizer")
    parser.add_argument("--n_docs", type=int, default=100,
                        help="number of LONG held-out docs to evaluate")
    parser.add_argument("--min_tokens", type=int, default=2048,
                        help="filter docs to tokenize >= this many tokens; then truncate to this length")
    parser.add_argument("--skip_docs", type=int, default=400_000,
                        help="skip from start of stream; disjoint from prior eval sets")
    parser.add_argument("--buckets", type=int, nargs="+",
                        default=[0, 64, 256, 512, 1024, 2048],
                        help="bucket edges (last must equal min_tokens)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    assert args.buckets[-1] <= args.min_tokens, "last bucket edge must be <= min_tokens"

    print(f"\n{'='*78}\nPosition-conditional PPL: baseline vs Active Memory\n{'='*78}")
    print(f"Baseline:      {args.baseline_path}")
    print(f"Active Memory: {args.active_memory_path}")
    print(f"Docs target:   {args.n_docs} (each >= {args.min_tokens} tokens, truncated to {args.min_tokens})")
    print(f"Buckets:       {args.buckets}")

    # Tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)

    # 1. Collect long docs
    print(f"\n[1/4] Streaming long documents (need {args.n_docs} of >= {args.min_tokens} tokens)...")
    docs, scanned = collect_long_docs(tokenizer, args.n_docs, args.min_tokens, args.skip_docs)
    print(f"  Collected {len(docs)} long docs after scanning {scanned} stream items")
    if len(docs) < args.n_docs:
        print(f"  [warn] only got {len(docs)} docs (asked for {args.n_docs})")
    avg_len = sum(len(d) for d in docs) / max(len(docs), 1)
    print(f"  Each doc truncated to {avg_len:.0f} tokens (target {args.min_tokens})")

    # 2. Eval baseline
    print(f"\n[2/4] Loading baseline + computing per-token losses...")
    model_b = load_model(args.baseline_path, active_memory=False, device=args.device)
    losses_b = []
    for i, doc in enumerate(docs):
        losses_b.append(per_position_losses(model_b, doc, args.device))
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(docs)} docs", end="\r")
    print()
    del model_b
    torch.cuda.empty_cache()

    # 3. Eval Active Memory
    print(f"\n[3/4] Loading Active Memory + computing per-token losses...")
    model_a = load_model(args.active_memory_path, active_memory=True, device=args.device)
    losses_a = []
    for i, doc in enumerate(docs):
        losses_a.append(per_position_losses(model_a, doc, args.device))
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(docs)} docs", end="\r")
    print()
    del model_a
    torch.cuda.empty_cache()

    # 4. Bucket aggregate + report
    sums_b, cnts_b = bucket_aggregate(losses_b, args.buckets)
    sums_a, cnts_a = bucket_aggregate(losses_a, args.buckets)

    print(f"\n[4/4] Per-bucket results\n")
    print(f"{'position range':>18} {'n tokens':>12} | {'loss_b':>10} {'loss_a':>10} {'Δ loss':>10} | {'PPL_b':>10} {'PPL_a':>10} {'rel Δ%':>10}")
    print("-" * 105)
    for b in range(len(args.buckets) - 1):
        lo, hi = args.buckets[b], args.buckets[b + 1]
        if cnts_b[b] == 0:
            continue
        lb = sums_b[b] / cnts_b[b]
        la = sums_a[b] / cnts_a[b]
        delta = la - lb
        ppl_b = math.exp(lb)
        ppl_a = math.exp(la)
        rel = (ppl_a - ppl_b) / ppl_b * 100
        print(f"{f'[{lo:>4}, {hi:>4})':>18} {cnts_b[b]:>12,} | {lb:>10.4f} {la:>10.4f} {delta:>+10.4f} | {ppl_b:>10.2f} {ppl_a:>10.2f} {rel:>+9.2f}%")

    # Overall
    total_lb = sum(sums_b) / sum(cnts_b)
    total_la = sum(sums_a) / sum(cnts_a)
    print("-" * 105)
    print(f"{'overall':>18} {sum(cnts_b):>12,} | {total_lb:>10.4f} {total_la:>10.4f} {total_la-total_lb:>+10.4f} | "
          f"{math.exp(total_lb):>10.2f} {math.exp(total_la):>10.2f} {(math.exp(total_la)-math.exp(total_lb))/math.exp(total_lb)*100:>+9.2f}%")

    # Trend check: does Δloss grow with position?
    deltas = []
    for b in range(len(args.buckets) - 1):
        if cnts_b[b] == 0:
            continue
        deltas.append((args.buckets[b], (sums_a[b]/cnts_a[b]) - (sums_b[b]/cnts_b[b])))

    print(f"\n{'-'*78}")
    print("Trend interpretation:")
    print("- If Δloss becomes MORE negative as position increases, AM helps more at long context.")
    print("- If Δloss is flat across buckets, AM is uniformly better (not specifically long-range).")
    print(f"\nΔloss by bucket start: " + "  ".join([f"{p}→{d:+.4f}" for p, d in deltas]))

    # Crude trend: is the last-bucket delta more negative than the first non-trivial bucket?
    if len(deltas) >= 2:
        first_nontrivial = deltas[1][1] if len(deltas) > 1 else deltas[0][1]
        last = deltas[-1][1]
        if last < first_nontrivial - 0.01:
            verdict = "AM advantage GROWS with position (supports long-range claim)"
        elif last > first_nontrivial + 0.01:
            verdict = "AM advantage SHRINKS with position (does NOT support long-range claim)"
        else:
            verdict = "AM advantage FLAT across positions (uniform improvement, not long-range specific)"
        print(f"\nVerdict: {verdict}")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
