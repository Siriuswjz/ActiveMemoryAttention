"""SFT (Supervised Fine-Tuning) for chat on SmolTalk dataset.

Usage:
  python scripts/train_sft.py --pretrained_path checkpoints/pretrain_main_100m/final
  accelerate launch --num_processes 8 scripts/train_sft.py --pretrained_path ...
"""

import os
import sys
import argparse

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Path to pretrained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    # Load pretrained model
    print(f"Loading pretrained model from {args.pretrained_path}")
    model = DeepseekV4ForCausalLM.from_pretrained(args.pretrained_path)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} parameters ({total_params/1e6:.1f}M)")

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.pretrained_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SmolTalk dataset
    print("Loading SmolTalk dataset...")
    dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    print(f"Dataset size: {len(dataset)} examples")

    # SFT Config
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        # Training - conservative for SFT
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_steps=args.max_steps,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Logging
        logging_steps=10,
        logging_first_step=True,
        disable_tqdm=True,
        report_to=["none"],
        # Saving
        save_steps=500,
        save_total_limit=3,
        # Hub
        push_to_hub=args.hub_model_id is not None,
        hub_model_id=args.hub_model_id,
        # Misc
        optim="adamw_torch_fused",
        dataloader_num_workers=4,
        seed=42,
        # DDP: MoE has unused params (inactive experts)
        ddp_find_unused_parameters=True,
    )

    # Initialize trackio
    try:
        import trackio
        trackio.init(
            project="smol-deepseek-v4",
            name="sft-dsv4-smoltalk",
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

    print(f"\n{'='*60}")
    print(f"SFT Training on SmolTalk")
    print(f"{'='*60}")
    print(f"Pretrained from: {args.pretrained_path}")
    print(f"Output: {args.output_dir}")
    print(f"Max steps: {args.max_steps}")
    print(f"LR: {args.lr}")
    print(f"{'='*60}\n")

    # Train
    trainer.train()

    # Save final
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nSFT model saved to {final_dir}")


if __name__ == "__main__":
    main()
