# nanowhale 🐳

A ~110M parameter language model trained from scratch using the **DeepSeek-V4 architecture**. This repo contains all the code, configs, and tokenizer used to pretrain and fine-tune the model.

## Models

| Model | Description | Link |
|---|---|---|
| **nanowhale-100m-base** | Pretrained base model (5K steps on FineWeb-Edu) | [🤗 Hub](https://huggingface.co/HuggingFaceTB/nanowhale-100m-base) |
| **nanowhale-100m** | SFT chat model (3K steps on SmolTalk) | [🤗 Hub](https://huggingface.co/HuggingFaceTB/nanowhale-100m) |

## Architecture

The model implements the full DeepSeek-V4 feature set at miniature scale:

- **Multi-Head Latent Attention (MLA)** — 8 heads, 1 KV head (MQA), head_dim=96 (32 RoPE + 64 NoPE), q_lora_rank=160
- **Mixture-of-Experts (MoE)** — 4 routed + 1 shared expert, top-2 routing, SwiGLU FFN (dim 640)
- **Hyper-Connections** — hc_mult=4, Sinkhorn routing (2 iterations)
- **Multi-Token Prediction (MTP)** — 1 next-token prediction layer

| Parameter | Value |
|---|---|
| Total params | ~110M (41M embeddings + 69M non-embedding) |
| Hidden size | 320 |
| Layers | 8 |
| Vocab size | 129,280 (DeepSeek-V4 tokenizer) |
| Context length | 2,048 tokens |

## Repo Structure

```
├── modeling_deepseek_v4.py         # DeepSeek-V4 model implementation
├── configuration_deepseek_v4.py    # Model config class
├── requirements.txt
├── configs/
│   ├── main_100m.yaml              # Training hyperparameters (100M model)
│   ├── debug.yaml                  # Quick debug config (50 steps)
│   └── fallback_under_1b.yaml      # Alternative config
├── scripts/
│   ├── train_pretrain.py            # Pretraining (SFTTrainer on FineWeb-Edu)
│   ├── train_sft.py                 # SFT fine-tuning (SFTTrainer on SmolTalk)
│   ├── eval_smoke.py                # Perplexity evaluation & generation
│   ├── chat.py                      # Interactive chat
│   ├── upload_to_hub.py             # Hub upload utility
│   ├── count_params.py              # Parameter counting
│   ├── prepare_data.py              # Data preparation
│   └── inspect_deepseek_v4.py       # Architecture inspection
└── tokenizer/
    ├── tokenizer.json
    └── tokenizer_config.json
```

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Pretraining

```bash
python scripts/train_pretrain.py --config configs/main_100m.yaml
```

### SFT

```bash
python scripts/train_sft.py
```

### Chat

```bash
python scripts/chat.py
```

### Evaluation

```bash
python scripts/eval_smoke.py
```

## Training Results

### Pretraining (5,000 steps on FineWeb-Edu)

| Metric | Value |
|---|---|
| Tokens seen | ~2.6B |
| Final loss | ~5.3 |
| Token accuracy | 33.8% |
| Hardware | 1× H100 80GB, bf16 |
| Throughput | 72ms/step (with `torch.compile`) |

### SFT (3,000 steps on SmolTalk)

| Metric | Start | End |
|---|---|---|
| Train loss | 15.41 | 10.22 |
| Eval loss | 2.873 | 2.607 |
| Token accuracy | 36.2% | 48.5% |

### Perplexity (held-out English text)

| Model | Perplexity |
|---|---|
| Pretrained | 13.62 |
| SFT | 12.90 |

## Known Issues

- **bf16 NaN**: The model produces NaN in bf16 at this small scale. Use fp32 for inference and training. This is due to the Hyper-Connections architecture producing values that overflow bf16 range.
- **`from_pretrained` quirk**: The custom architecture causes `from_pretrained` to re-initialize some weights. Use manual `load_state_dict` instead (see model cards for examples).
- **Large vocab / small model**: The 129K vocab embedding table consumes 37% of all parameters, limiting capacity for language modeling.

## License

MIT
