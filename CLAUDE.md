# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**nanowhale** is a ~110M parameter language model trained from scratch using the DeepSeek-V4 architecture. The primary research contribution in this branch is **Active Memory Attention** (`active_memory_attention.py`), a drop-in replacement for `DeepseekV4Attention` that gives each KV cache entry a persistent "activation level" that evolves over time (decay + spread + cue-boost + noise), biologically inspired by associative memory networks.

## Commands

### Install
```bash
pip install -r requirements.txt
```

### Training
```bash
# Pretraining (FineWeb-Edu)
python scripts/train_pretrain.py --config configs/main_100m.yaml

# Quick debug run (50 steps)
python scripts/train_pretrain.py --config configs/debug.yaml

# SFT (SmolTalk)
python scripts/train_sft.py

# Multi-GPU
accelerate launch --num_processes 8 scripts/train_pretrain.py --config configs/main_100m.yaml
```

### Evaluation & Inference
```bash
python scripts/eval_smoke.py    # perplexity + generation
python scripts/chat.py          # interactive chat
python scripts/count_params.py  # parameter breakdown
```

### Active Memory unit tests
```bash
python active_memory_attention.py   # runs 4 built-in tests
```

## Architecture

### Core files
- `modeling_deepseek_v4.py` — full DeepSeek-V4 model (`DeepseekV4ForCausalLM`). All ops are pure PyTorch; no custom kernels required.
- `configuration_deepseek_v4.py` — `DeepseekV4Config` (HF `PretrainedConfig` subclass). Active Memory params are read from config fields prefixed `am_` (e.g., `am_spread_width`, `am_init_decay`).
- `active_memory_attention.py` — `ActiveMemoryAttention` and `ActivationDynamics`. Designed as a drop-in replacement for `DeepseekV4Attention`.

### Active Memory Attention
The key idea: each cached token carries an activation scalar `a_i` that persists in the KV cache. At each forward step:
1. **Decay** — `a_i *= γ` (per-head learnable, sigmoid-bounded)
2. **Spread** — 1D grouped convolution over neighbors
3. **Cue boost** — `a_i += β · relevance(q, k_i)` (query-driven reinforcement)
4. **Noise** — training-only Gaussian perturbation

Attention scores become `q·k^T/√d + λ·a_i` before softmax. The `past_key_value` cache is extended from `(K, V)` to `(K, V, activation)`.

To swap Active Memory into an existing model:
```python
from active_memory_attention import replace_attention_with_active_memory
replace_attention_with_active_memory(model)  # copies all existing weights
```

### KV heads & MLA
The model uses Multi-Head Latent Attention (MQA style): 8 query heads, 1 KV head. KV is broadcast to all heads via `.expand()`. The output projection uses grouped low-rank factoring (`o_groups=2`, `o_lora_rank=80`).

### Known issues
- **bf16 NaN** at this scale due to Hyper-Connections overflow — use fp32 for inference and training.
- **`from_pretrained` reinitializes weights** for custom architectures — use manual `load_state_dict` instead.
- Training configs set `bf16: true`; if you hit NaN, switch to fp32.

## Configs
YAML configs under `configs/` have two top-level keys: `model` (passed directly to `DeepseekV4Config`) and `training` (passed to `SFTConfig`). Use `configs/debug.yaml` for smoke tests (50 steps) before committing to a full run.
