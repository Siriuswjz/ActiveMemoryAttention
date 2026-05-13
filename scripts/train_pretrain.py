"""Pretrain a DeepSeek-V4 model from scratch on FineWeb-Edu using TRL SFTTrainer.

Usage:
  python scripts/train_pretrain.py --config configs/main_100m.yaml [--debug]
  accelerate launch --num_processes 8 scripts/train_pretrain.py --config configs/main_100m.yaml
"""

import os
import sys
import argparse
import yaml

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, AutoConfig, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

from configuration_deepseek_v4 import DeepseekV4Config
from modeling_deepseek_v4 import DeepseekV4ForCausalLM

# Register for Auto classes
AutoConfig.register("deepseek_v4", DeepseekV4Config, exist_ok=True)
AutoModelForCausalLM.register(DeepseekV4Config, DeepseekV4ForCausalLM, exist_ok=True)


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(model_cfg):
    """Build a DeepSeek-V4 model from scratch (random init)."""
    config = DeepseekV4Config(**model_cfg)
    model = DeepseekV4ForCausalLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {total_params:,} parameters ({total_params/1e6:.1f}M)")
    return model


def build_tokenizer(tokenizer_path="tokenizer"):
    """Load the DeepSeek-V4 tokenizer."""
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/main_100m.yaml")
    parser.add_argument("--debug", action="store_true", help="Use tiny subset for smoke testing")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--active_memory", action="store_true", help="Replace attention with ActiveMemoryAttention")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    # Build model and tokenizer
    torch.set_float32_matmul_precision('high')
    model = build_model(model_cfg)
    if args.active_memory:
        from active_memory_attention import replace_attention_with_active_memory
        replace_attention_with_active_memory(model)
        print("Active Memory Attention enabled")
    if torch.cuda.is_available():
        model = torch.compile(model)
        print("torch.compile enabled")
    tokenizer = build_tokenizer()

    # Output directory
    suffix = "_active_memory" if args.active_memory else ""
    output_dir = args.output_dir or f"checkpoints/pretrain_{os.path.basename(args.config).replace('.yaml', '')}{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    # Dataset
    if args.debug:
        # Tiny subset for smoke testing
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
        dataset = dataset.take(200)
        # Convert to regular dataset
        from datasets import Dataset
        rows = list(dataset)
        dataset = Dataset.from_list(rows)
        max_steps = train_cfg.get("max_steps", 50)
        save_steps = 10
        logging_steps = 1
    else:
        # Stream the full dataset
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
        )
        max_steps = train_cfg.get("max_steps", 20000)
        save_steps = train_cfg.get("save_steps", 2000)
        logging_steps = train_cfg.get("logging_steps", 10)

    max_seq_length = train_cfg.get("max_seq_length", 2048)

    # SFT Config for pretraining (raw text, no chat template)
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_length=max_seq_length,
        packing=True,
        dataset_text_field="text",
        # Training
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 6e-4),
        weight_decay=train_cfg.get("weight_decay", 0.1),
        adam_beta1=train_cfg.get("adam_beta1", 0.9),
        adam_beta2=train_cfg.get("adam_beta2", 0.95),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        max_steps=max_steps,
        bf16=torch.cuda.is_available() and train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Logging
        logging_steps=logging_steps,
        logging_first_step=True,
        disable_tqdm=True,
        report_to=["none"],
        # Saving
        save_steps=save_steps,
        save_total_limit=3,
        # Hub
        push_to_hub=args.hub_model_id is not None,
        hub_model_id=args.hub_model_id,
        # Misc
        optim="adamw_torch_fused",
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        seed=42,
        # DDP: MoE has unused params (inactive experts)
        ddp_find_unused_parameters=True,
    )

    # Initialize trackio
    try:
        import trackio
        trackio.init(
            project="smol-deepseek-v4",
            name=f"pretrain-dsv4-{os.path.basename(args.config).replace('.yaml', '')}{suffix}",
        )
        sft_config.report_to = ["trackio"]
        print("Trackio logging enabled")
    except Exception as e:
        print(f"Trackio not available: {e}")

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Print training info
    print(f"\n{'='*60}")
    print(f"Pretraining DeepSeek-V4")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print(f"Max steps: {max_steps}")
    print(f"Batch size: {sft_config.per_device_train_batch_size} x {sft_config.gradient_accumulation_steps} GA")
    print(f"Seq length: {sft_config.max_length}")
    print(f"LR: {sft_config.learning_rate}")
    print(f"Debug mode: {args.debug}")
    print(f"{'='*60}\n")

    # Train
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final checkpoint
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    print(f"\nFinal model saved to {output_dir}/final")


if __name__ == "__main__":
    main()
