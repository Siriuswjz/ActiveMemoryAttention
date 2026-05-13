"""Tier 2: long-range recall test.

Constructs prompts of the form:
    "The secret word is APPLE. <N filler tokens>. The secret word is "
and measures the model's log-probability of the target token (APPLE) at the
final position. Compares baseline vs Active Memory across increasing N.

Bigger gap at larger N → Active Memory's "memory persistence" claim holds.

Usage:
  python scripts/eval_recall.py \
    --baseline_path checkpoints/pretrain_main_100m/final \
    --active_memory_path checkpoints/pretrain_main_100m_active_memory/final
"""

import os
import sys
import argparse
import math
import json

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


# Candidate single-token secret words. Filtered at runtime to those that
# tokenize to a single token under the project tokenizer.
CANDIDATE_SECRETS = [
    "apple", "river", "mountain", "doctor", "paper", "music", "candle",
    "garden", "mirror", "ocean", "tiger", "violin", "engine", "letter",
    "window", "kitchen", "diamond", "forest", "harbor", "rocket",
]


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


def filter_single_token_secrets(tokenizer, candidates):
    """Keep only words that tokenize to exactly one token (with leading space)."""
    keep = []
    for w in candidates:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            keep.append((w, ids[0]))
    return keep


def build_filler_pool(tokenizer, n_min_tokens, n_pool_docs=50):
    """Pull held-out FineWeb-Edu docs and return their tokenized ids (list of 1D tensors)."""
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    ds = ds.skip(300_000).take(n_pool_docs)
    pool = []
    for row in ds:
        ids = tokenizer.encode(row["text"], add_special_tokens=False)
        if len(ids) >= n_min_tokens:
            pool.append(ids)
        if len(pool) >= 20:
            break
    if not pool:
        raise RuntimeError(f"Could not find enough filler docs of length >= {n_min_tokens}")
    return pool


def build_probe(setup_ids, filler_ids, n_filler, probe_ids, target_id):
    """
    Build: setup + (first n_filler tokens of filler) + probe
    Returns input_ids tensor (1, L) and the position of the target prediction (last position).
    """
    filler = filler_ids[:n_filler]
    ids = setup_ids + filler + probe_ids
    return torch.tensor([ids], dtype=torch.long)


@torch.no_grad()
def score_target(model, input_ids, target_id, device):
    """Return (logprob_of_target, rank_of_target, top1_id)."""
    input_ids = input_ids.to(device)
    out = model(input_ids=input_ids)
    logits = out.logits[0, -1].float()  # logits for the token AFTER the last position
    logp = F.log_softmax(logits, dim=-1)
    target_logp = logp[target_id].item()
    # rank: how many tokens have higher logprob than target?
    rank = int((logp > logp[target_id]).sum().item()) + 1
    top1 = int(logits.argmax().item())
    return target_logp, rank, top1


def run_recall_eval(model, tokenizer, secrets, filler_pool, n_filler_list,
                    n_trials, device, label):
    """Run all probes; returns dict[n_filler] -> list of dicts per trial."""
    setup_template = "The secret word is{word}. "
    probe_template = "The secret word is"

    results = {n: [] for n in n_filler_list}

    n_total = len(secrets) * len(n_filler_list) * n_trials
    done = 0
    for word, target_id in secrets:
        setup_text = setup_template.format(word=" " + word)
        setup_ids = tokenizer.encode(setup_text, add_special_tokens=False)
        probe_ids = tokenizer.encode(probe_template, add_special_tokens=False)
        for n_filler in n_filler_list:
            for trial in range(n_trials):
                filler_ids = filler_pool[trial % len(filler_pool)]
                if len(filler_ids) < n_filler:
                    continue
                ids = build_probe(setup_ids, filler_ids, n_filler, probe_ids, target_id)
                logp, rank, top1 = score_target(model, ids, target_id, device)
                results[n_filler].append({
                    "word": word,
                    "target_id": target_id,
                    "trial": trial,
                    "logp": logp,
                    "rank": rank,
                    "top1": top1,
                    "correct": top1 == target_id,
                    "seq_len": ids.shape[1],
                })
                done += 1
        print(f"  [{label}] {done}/{n_total} probes", end="\r")
    print()
    return results


def summarize(results, n_filler_list):
    rows = []
    for n in n_filler_list:
        probes = results[n]
        if not probes:
            rows.append((n, 0, float("nan"), float("nan"), float("nan")))
            continue
        mean_logp = sum(p["logp"] for p in probes) / len(probes)
        mean_rank = sum(p["rank"] for p in probes) / len(probes)
        acc1 = sum(1 for p in probes if p["correct"]) / len(probes)
        rows.append((n, len(probes), mean_logp, mean_rank, acc1))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_path", required=True)
    parser.add_argument("--active_memory_path", required=True)
    parser.add_argument("--tokenizer_path", default="tokenizer")
    parser.add_argument("--n_filler_list", type=int, nargs="+",
                        default=[0, 64, 256, 512, 1024, 1800])
    parser.add_argument("--n_trials", type=int, default=5,
                        help="number of filler-text variants per (word, N) pair")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"\n{'='*72}\nLong-Range Recall Test: baseline vs Active Memory\n{'='*72}")
    print(f"Baseline:      {args.baseline_path}")
    print(f"Active Memory: {args.active_memory_path}")
    print(f"Filler lengths: {args.n_filler_list}  (trials per length: {args.n_trials})")

    # 1. Tokenizer + secret filter
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    secrets = filter_single_token_secrets(tokenizer, CANDIDATE_SECRETS)
    print(f"\n[1/4] Secret words ({len(secrets)} single-token candidates):")
    print(f"      {[w for w, _ in secrets]}")

    # 2. Filler pool
    max_n = max(args.n_filler_list)
    print(f"\n[2/4] Building filler pool (need >= {max_n} tokens each)...")
    filler_pool = build_filler_pool(tokenizer, n_min_tokens=max_n)
    print(f"      Got {len(filler_pool)} filler documents")

    # 3. Evaluate
    print(f"\n[3/4] Loading baseline + evaluating...")
    model_b = load_model(args.baseline_path, active_memory=False, device=args.device)
    res_b = run_recall_eval(model_b, tokenizer, secrets, filler_pool,
                            args.n_filler_list, args.n_trials, args.device, "baseline")
    del model_b
    torch.cuda.empty_cache()

    print(f"\n[3/4] Loading Active Memory + evaluating...")
    model_a = load_model(args.active_memory_path, active_memory=True, device=args.device)
    res_a = run_recall_eval(model_a, tokenizer, secrets, filler_pool,
                            args.n_filler_list, args.n_trials, args.device, "active_mem")
    del model_a
    torch.cuda.empty_cache()

    # 4. Report
    sum_b = summarize(res_b, args.n_filler_list)
    sum_a = summarize(res_a, args.n_filler_list)

    print(f"\n[4/4] Results\n")
    print(f"{'N filler':>10} {'n':>5} | {'logp(target)':>30} {'rank':>20} {'acc@1':>20}")
    print(f"{'':>10} {'':>5} | {'baseline':>14} {'AM':>14}  {'baseline':>9} {'AM':>9}  {'baseline':>9} {'AM':>9}")
    print("-" * 95)
    for (n, nb, lb, rb, ab), (_, na, la, ra, aa) in zip(sum_b, sum_a):
        print(f"{n:>10} {nb:>5} | {lb:>14.4f} {la:>14.4f}  {rb:>9.1f} {ra:>9.1f}  {ab:>9.2%} {aa:>9.2%}")

    print(f"\n{'-'*72}\nlogp delta (AM - baseline); positive = AM gives higher target logprob:")
    print(f"{'N filler':>10} {'Δ logp':>14} {'Δ acc@1':>14}")
    for (n, _, lb, _, ab), (_, _, la, _, aa) in zip(sum_b, sum_a):
        print(f"{n:>10} {la-lb:>+14.4f} {(aa-ab)*100:>+13.2f}pp")

    print(f"\n{'='*72}")
    print("Interpretation hint:")
    print("- At N=0, the model just copies — both should score similarly. If not, model is too weak.")
    print("- The CLAIM: AM's Δlogp should grow (or stay positive) as N increases, while baseline decays.")
    print("- If Δlogp stays flat or shrinks with N, the 'memory persistence' claim is unsupported.")
    print(f"{'='*72}\n")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({
                "n_filler_list": args.n_filler_list,
                "baseline": res_b,
                "active_memory": res_a,
                "secrets": [w for w, _ in secrets],
            }, f, indent=2)
        print(f"Detailed per-probe results saved to: {args.out_json}")


if __name__ == "__main__":
    main()
