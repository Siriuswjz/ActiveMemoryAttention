"""Compare baseline vs Active Memory checkpoints on held-out FineWeb-Edu.

Usage:
  python scripts/eval_compare.py \
    --baseline_path checkpoints/pretrain_main_100m/final \
    --active_memory_path checkpoints/pretrain_main_100m_active_memory/final

Notes:
  - Both checkpoints were saved through torch.compile, so state_dict keys
    carry the `_orig_mod.` prefix. We strip it before load_state_dict.
  - fp32 inference (bf16 NaNs at this scale per CLAUDE.md).
  - Held-out subset: streams FineWeb-Edu, skips the first N docs the model
    likely saw during training, then takes the next eval_docs documents.
"""

import os
import sys
import argparse
import math

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
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
    state_dict = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    return model.to(device).to(dtype).eval()


@torch.no_grad()
def eval_perplexity(model, tokenizer, docs, max_seq_length, device):
    """Return (total_loss, total_tokens, per_doc_loss list)."""
    total_loss = 0.0
    total_tokens = 0
    per_doc_losses = []
    for i, text in enumerate(docs):
        ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_seq_length).to(device)
        if ids.shape[1] < 2:
            continue
        out = model(input_ids=ids, labels=ids)
        n_pred = ids.shape[1] - 1
        loss_val = out.loss.item()
        if not math.isfinite(loss_val):
            print(f"  [warn] non-finite loss at doc {i}, skipping")
            continue
        total_loss += loss_val * n_pred
        total_tokens += n_pred
        per_doc_losses.append(loss_val)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(docs)} docs, running PPL = {math.exp(total_loss/total_tokens):.3f}")
    return total_loss, total_tokens, per_doc_losses


def paired_test(baseline_losses, am_losses):
    """Paired t-test on per-doc losses. Returns (mean_diff, t, p)."""
    import statistics
    diffs = [a - b for a, b in zip(am_losses, baseline_losses)]  # am - baseline; negative = AM better
    n = len(diffs)
    mean = statistics.mean(diffs)
    if n < 2:
        return mean, float("nan"), float("nan")
    stdev = statistics.stdev(diffs)
    if stdev == 0:
        return mean, float("inf") if mean != 0 else 0.0, 0.0
    t = mean / (stdev / math.sqrt(n))
    # two-sided p from normal approx (n is usually >50; for small n this is rough)
    from math import erfc
    p = erfc(abs(t) / math.sqrt(2))
    return mean, t, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_path", required=True)
    parser.add_argument("--active_memory_path", required=True)
    parser.add_argument("--tokenizer_path", default="tokenizer")
    parser.add_argument("--eval_docs", type=int, default=200, help="number of held-out docs")
    parser.add_argument("--skip_docs", type=int, default=200_000,
                        help="skip this many docs from start of FineWeb-Edu stream (avoid overlap with training)")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"\n{'='*70}\nEval Compare: baseline vs Active Memory\n{'='*70}")
    print(f"Baseline:      {args.baseline_path}")
    print(f"Active Memory: {args.active_memory_path}")
    print(f"Eval docs:     {args.eval_docs}  (skipping first {args.skip_docs:,} of FineWeb-Edu)")
    print(f"Max seq len:   {args.max_seq_length}")
    print(f"Device:        {args.device}")

    # 1. Build held-out doc list ONCE (same docs for both models)
    print(f"\n[1/4] Streaming held-out docs...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    ds = ds.skip(args.skip_docs).take(args.eval_docs)
    docs = [row["text"] for row in ds]
    print(f"  Collected {len(docs)} docs, avg chars = {sum(len(d) for d in docs)/len(docs):.0f}")

    # 2. Tokenizer
    print(f"\n[2/4] Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Eval each model
    print(f"\n[3/4] Evaluating baseline...")
    model_b = load_model(args.baseline_path, active_memory=False, device=args.device)
    loss_b, ntok_b, doc_losses_b = eval_perplexity(model_b, tokenizer, docs, args.max_seq_length, args.device)
    del model_b
    torch.cuda.empty_cache()

    print(f"\n[3/4] Evaluating Active Memory...")
    model_a = load_model(args.active_memory_path, active_memory=True, device=args.device)
    loss_a, ntok_a, doc_losses_a = eval_perplexity(model_a, tokenizer, docs, args.max_seq_length, args.device)
    del model_a
    torch.cuda.empty_cache()

    # 4. Report
    print(f"\n[4/4] Results\n")
    mean_b = loss_b / ntok_b
    mean_a = loss_a / ntok_a
    ppl_b = math.exp(mean_b)
    ppl_a = math.exp(mean_a)

    print(f"{'Model':<22} {'Loss':>10} {'PPL':>10} {'tokens':>12}")
    print("-" * 56)
    print(f"{'Standard MLA':<22} {mean_b:>10.4f} {ppl_b:>10.3f} {ntok_b:>12,}")
    print(f"{'Active Memory (full)':<22} {mean_a:>10.4f} {ppl_a:>10.3f} {ntok_a:>12,}")
    delta_loss = mean_a - mean_b
    rel_ppl = (ppl_a - ppl_b) / ppl_b * 100
    print(f"\nLoss delta:   {delta_loss:+.4f}  ({'AM better' if delta_loss < 0 else 'baseline better' if delta_loss > 0 else 'tie'})")
    print(f"PPL  delta:   {ppl_a - ppl_b:+.3f}  ({rel_ppl:+.2f}% relative)")

    # Paired comparison (only on docs both models scored)
    n_pairs = min(len(doc_losses_b), len(doc_losses_a))
    mean_diff, t, p = paired_test(doc_losses_b[:n_pairs], doc_losses_a[:n_pairs])
    print(f"\nPaired per-doc test (n={n_pairs}):")
    print(f"  mean(am - baseline) = {mean_diff:+.4f}")
    print(f"  t = {t:.3f},  approx two-sided p = {p:.4f}")
    if p < 0.05:
        verdict = "Active Memory significantly better" if mean_diff < 0 else "Baseline significantly better"
    else:
        verdict = "No significant difference"
    print(f"  → {verdict}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
