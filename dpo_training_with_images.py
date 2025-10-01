#!/usr/bin/env python3
import os
import argparse
import json
import random
from typing import List, Any

from PIL import Image

import torch
from datasets import Dataset

from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor

# PEFT & TRL DPO
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

# -------------------------
# Arg parsing
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base_model', type=str, required=True, help="Path/name for Qwen2.5-VL model (e.g. Qwen/Qwen2.5-VL-7B-Instruct)")
    p.add_argument('--dataset', type=str, default="hotpotqa_7b_data.json", help="JSON file with paired examples")
    p.add_argument('--output_dir', type=str, default="./results_hotpot_7b_qwenvl", help="output dir")
    p.add_argument('--percentage', type=float, default=1.0, help="fraction of dataset to use")
    p.add_argument('--bs', type=int, default=2, help="per-device batch size")
    p.add_argument('--lora_r', type=int, default=8, help="LoRA rank")
    p.add_argument('--learning_rate', type=float, default=5e-6)
    p.add_argument('--max_prompt_length', type=int, default=512)
    p.add_argument('--max_length', type=int, default=1024)
    p.add_argument('--max_steps', type=int, default=900)
    p.add_argument('--logging_steps', type=int, default=100)
    p.add_argument('--eval_steps', type=int, default=100)
    p.add_argument('--save_steps', type=int, default=300)
    p.add_argument('--randomseed', type=int, default=42)
    p.add_argument('--pct_warmup', type=float, default=0.1, help="fraction for warmup steps")
    p.add_argument('--sanity_check', action='store_true')
    return p.parse_args()

args = parse_args()

# -------------------------
# Utility functions for images / bboxes
# -------------------------
def load_image_rgb(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    return img

def parse_bboxes(bbox_field: Any) -> List[List[float]]:
    """Accepts a Python list or a JSON string representing a list of boxes."""
    if bbox_field is None:
        return []
    if isinstance(bbox_field, str):
        try:
            parsed = json.loads(bbox_field)
            return parsed
        except Exception:
            # fallback to eval (risky but sometimes data uses python repr)
            try:
                parsed = eval(bbox_field)
                return parsed
            except Exception:
                return []
    if isinstance(bbox_field, (list, tuple)):
        return list(bbox_field)
    return []

def norm_to_pixel_coords(bbox: List[float], width: int, height: int):
    """Transform [x0,y0,x1,y1] in normalized coordinates to pixel box (left,upper,right,lower)."""
    x0, y0, x1, y1 = bbox
    left = int(round(x0 * width))
    upper = int(round(y0 * height))
    right = int(round(x1 * width))
    lower = int(round(y1 * height))
    left = max(0, min(left, width - 1))
    upper = max(0, min(upper, height - 1))
    right = max(left + 1, min(right, width))
    lower = max(upper + 1, min(lower, height))
    return (left, upper, right, lower)

def crop_image_by_bbox(pil_img: Image.Image, bbox: List[float]) -> Image.Image:
    w, h = pil_img.size
    left, upper, right, lower = norm_to_pixel_coords(bbox, w, h)
    return pil_img.crop((left, upper, right, lower))

# -------------------------
# Load dataset JSON into HuggingFace Dataset
# -------------------------
def load_json_dataset(json_path: str, percentage: float=1.0, sanity_check=False) -> Dataset:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Dataset JSON must be a non-empty list of records.")
    n = len(data)
    take = int(round(n * percentage))
    if percentage < 1.0:
        random.seed(args.randomseed)
        data = random.sample(data, take)
    else:
        data = data[:take]
    if sanity_check:
        data = data[:min(1000, len(data))]
    # build dataset columns we need
    # ensure every example has metadata dict
    for ex in data:
        if "metadata" not in ex:
            ex["metadata"] = {}
    ds = Dataset.from_list(data)
    return ds

if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_json_dataset(args.dataset, percentage=args.percentage, sanity_check=args.sanity_check)
    print(f"Loaded {len(dataset)} examples.")

    # -------------------------
    # Load model & processor / tokenizer
    # -------------------------
    if AutoProcessor is None or Qwen2_5_VLForConditionalGeneration is None:
        raise RuntimeError(
            "AutoProcessor / Qwen2_5_VLForConditionalGeneration not importable. "
            "Install the Qwen-VL-compatible transformers package or check environment."
        )

    print("Loading processor & model...")
    processor = AutoProcessor.from_pretrained(args.base_model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    # Turn off caching for training
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Tokenizer (for fallback processing and text-only needs)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else AutoTokenizer.from_pretrained(args.base_model)

    # -------------------------
    # Map function (multimodal): convert each example into processor inputs
    # We will create fields that the DPOTrainer will accept via processing_class=processor
    # The processor returns tensors; we'll store lists (so Dataset columns remain json-serializable).
    # -------------------------
    def multimodal_map(example):
        """
        Fixed multimodal_map:
        - Inject bbox info into the prompt (text)
        - Always require a valid image (throw if missing/unreadable)
        - Return processor-compatible dict (prompt, images, chosen, rejected, image_path)
        """
        prompt = example.get("prompt", "").strip()
        chosen = example.get("chosen", "")
        rejected = example.get("rejected", "")

        if not prompt.endswith("Answer:"):
            prompt = prompt.rstrip() + "\n\nAnswer:"

        meta = example.get("metadata", {}) or {}
        image_path = meta.get("image_path", None)
        bbox_field = meta.get("bbox", None)

        # Inject bbox text into prompt
        enhanced_prompt = prompt
        if bbox_field:
            try:
                bboxes = parse_bboxes(bbox_field)
                if len(bboxes) > 0:
                    bbox_strs = [f"[x0={b[0]:.4f}, y0={b[1]:.4f}, x1={b[2]:.4f}, y1={b[3]:.4f}]" for b in bboxes]
                    enhanced_prompt = prompt + "\n\nFocus on region(s): " + ", ".join(bbox_strs)
            except Exception as e:
                print(f"Warning: failed to parse bbox_field: {e}")

        # Must have image path
        if not image_path:
            raise ValueError(f"Missing image_path in metadata for example: {example}")

        # Normalize and resolve relative path
        image_path = image_path.replace("\\", os.sep).replace("/", os.sep)
        if not os.path.isabs(image_path):
            base_dir = os.path.dirname(os.path.abspath(args.dataset))
            candidate = os.path.join(base_dir, image_path)
            if os.path.exists(candidate):
                image_path = candidate

        # Try to load image
        try:
            pil_img = load_image_rgb(image_path)
        except Exception as e:
            raise RuntimeError(f"Could not load image at {image_path}: {e}")

        return {
            "prompt": enhanced_prompt,
            "images": [pil_img],         # always a list for processor
            "chosen": str(chosen),
            "rejected": str(rejected),
            "image_path": image_path,    # keep path string for debugging
        }

    print("Mapping dataset with multimodal processor (this may take time)...")
    # Remove original columns to avoid conflicts — we'll keep chosen/rejected as text fields in new columns
    dataset = dataset.map(multimodal_map, remove_columns=dataset.column_names, num_proc=1)
    print("Mapping done. Dataset columns:", dataset.column_names)

    # -------------------------
    # Split train/test and compute training steps/warmup
    # -------------------------
    dataset = dataset.train_test_split(test_size=0.1, seed=args.randomseed)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # compute warmup and steps (simple heuristic as in your original script)
    warmup_steps = max(10, int(round(args.pct_warmup * len(train_dataset) / (4 * args.bs))))
    total_steps = max(1, int(round(len(train_dataset) / (4 * args.bs)) * 3))
    save_steps = args.save_steps
    eval_steps = args.eval_steps

    # align save_steps to multiple of eval_steps
    if save_steps % eval_steps != 0:
        save_steps = ((save_steps // eval_steps) + 1) * eval_steps
    save_steps = min(save_steps, total_steps)

    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}, Save steps: {save_steps}, Eval steps: {eval_steps}")

    # -------------------------
    # DPO config and LoRA (PEFT)
    # -------------------------
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        max_steps=total_steps,
        logging_steps=args.logging_steps,
        save_steps=save_steps,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        eval_strategy="steps",
        eval_steps=eval_steps,
        report_to="none",
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        optim="adamw_torch",
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        run_name="dpo_qwen2vl_run",
        # DPO params
        beta=0.2,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        logging_first_step=True,
        logging_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "out_proj",
            "fc_in", "fc_out", "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # -------------------------
    # Initialize DPO trainer
    # -------------------------
    print("Initializing DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,  # critical: pass processor for multimodal handling
        peft_config=peft_config,
    )

    # -------------------------
    # Train
    # -------------------------
    print("Starting training ...")
    try:
        dpo_trainer.train()
        print("Training finished.")
    except Exception as e:
        print("Error during training:", e)
        raise

    # -------------------------
    # Save final model and processor
    # -------------------------
    print("Saving model and processor...")
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        dpo_trainer.save_model(args.output_dir)
        # also save underlying model weights and processor/tokenizer
        model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint"))
        processor.save_pretrained(os.path.join(args.output_dir, "processor"))
        print(f"Saved to {args.output_dir}")
    except Exception as e:
        print("Save error:", e)
        raise
