"""Upload the final model to Hugging Face Hub."""

import sys
import os
import json
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from huggingface_hub import HfApi, create_repo
from transformers import PreTrainedTokenizerFast, AutoConfig, AutoModelForCausalLM
from configuration_deepseek_v4 import DeepseekV4Config
from modeling_deepseek_v4 import DeepseekV4ForCausalLM

AutoConfig.register("deepseek_v4", DeepseekV4Config, exist_ok=True)
AutoModelForCausalLM.register(DeepseekV4Config, DeepseekV4ForCausalLM, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint to upload")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="Hub repo ID (e.g. username/model-name)")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()

    # Create repo
    print(f"Creating/getting repo: {args.repo_id}")
    create_repo(args.repo_id, exist_ok=True, private=args.private)

    # Prepare upload directory
    upload_dir = os.path.join(args.model_path, "upload_staging")
    os.makedirs(upload_dir, exist_ok=True)

    # Copy model files
    for f in os.listdir(args.model_path):
        if f.endswith(('.safetensors', '.json', '.txt')):
            shutil.copy2(os.path.join(args.model_path, f), upload_dir)

    # Copy custom code files (needed for trust_remote_code)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for f in ['configuration_deepseek_v4.py', 'modeling_deepseek_v4.py']:
        src = os.path.join(project_root, f)
        if os.path.exists(src):
            shutil.copy2(src, upload_dir)

    # Add auto_map to config.json
    config_path = os.path.join(upload_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        config["auto_map"] = {
            "AutoConfig": "configuration_deepseek_v4.DeepseekV4Config",
            "AutoModelForCausalLM": "modeling_deepseek_v4.DeepseekV4ForCausalLM"
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    # Upload
    print(f"Uploading to {args.repo_id}...")
    api.upload_folder(
        folder_path=upload_dir,
        repo_id=args.repo_id,
        commit_message="Upload SmolDeepSeek-V4 model",
    )

    # Clean up staging
    shutil.rmtree(upload_dir)
    print(f"\nUploaded to: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
