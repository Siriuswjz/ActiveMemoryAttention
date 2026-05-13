"""Interactive chat with a DeepSeek-V4 model."""

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
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    model = DeepseekV4ForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    model = model.to(args.device).eval()
    print(f"Model loaded on {args.device}")
    print("Type 'quit' to exit.\n")

    messages = []
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Assistant: {response}\n")
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
