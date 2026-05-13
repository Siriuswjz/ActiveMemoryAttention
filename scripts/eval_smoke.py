"""Smoke evaluation: validate model loads, generates, and compute basic metrics."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from transformers import PreTrainedTokenizerFast, AutoConfig, AutoModelForCausalLM
from configuration_deepseek_v4 import DeepseekV4Config
from modeling_deepseek_v4 import DeepseekV4ForCausalLM

AutoConfig.register("deepseek_v4", DeepseekV4Config, exist_ok=True)
AutoModelForCausalLM.register(DeepseekV4Config, DeepseekV4ForCausalLM, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Smoke Evaluation: {args.model_path}")
    print(f"{'='*60}")

    # 1. Load tokenizer
    print("\n[1/6] Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  BOS: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    print(f"  EOS: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    assert tokenizer.vocab_size > 0, "Tokenizer vocab is empty"

    # 2. Load config
    print("\n[2/6] Loading config...")
    config = DeepseekV4Config.from_pretrained(args.model_path)
    print(f"  Model type: {config.model_type}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")

    # 3. Load model
    print("\n[3/6] Loading model...")
    model = DeepseekV4ForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    model = model.to(args.device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # 4. Test generation
    print("\n[4/6] Testing generation...")
    prompts = [
        "The meaning of life is",
        "Once upon a time in a land far away,",
        "The quick brown fox",
        "Python is a programming language that",
        "In mathematics, the Pythagorean theorem states that",
    ]
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=50, do_sample=True,
                                     temperature=0.8, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"  Prompt: {prompt!r}")
        print(f"  Output: {text!r}")
        print()

    # 5. Test chat
    print("[5/6] Testing chat template...")
    if tokenizer.chat_template:
        messages = [
            {"role": "user", "content": "Explain why the sky is blue in simple terms."}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=100, do_sample=True,
                                     temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"  Chat prompt: {prompt!r}")
        print(f"  Output: {text!r}")
    else:
        print("  No chat template - SKIPPED")

    # 6. Compute perplexity on small held-out text
    print("\n[6/6] Computing perplexity on sample text...")
    test_texts = [
        "The Earth orbits the Sun at a distance of approximately 93 million miles.",
        "Water is composed of two hydrogen atoms and one oxygen atom.",
        "Shakespeare wrote many famous plays including Hamlet and Romeo and Juliet.",
    ]
    total_loss = 0
    total_tokens = 0
    for text in test_texts:
        input_ids = tokenizer.encode(text, return_tensors="pt").to(args.device)
        with torch.no_grad():
            output = model(input_ids, labels=input_ids)
        total_loss += output.loss.item() * (input_ids.shape[1] - 1)
        total_tokens += input_ids.shape[1] - 1

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")

    print(f"\n{'='*60}")
    print("ALL SMOKE TESTS PASSED")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
