from pathlib import Path

import torch
from transformers import Blip2ForConditionalGeneration
from safetensors import safe_open


def main():
    weights_dir = Path("./weights")
    if not weights_dir.exists():
        weights_dir.mkdir()

    hf_weights_path = weights_dir / "blip2-vit-hf"
    if not hf_weights_path.exists():
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xxl",
            torch_dtype=torch.float16,
        )

        model.vision_model.save_pretrained(hf_weights_path, safe_serialization=True)
        del model

    with safe_open(hf_weights_path / "model.safetensors", framework="np") as f:
        for k in f.keys():
            print(k)


if __name__ == "__main__":
    main()
