#!/usr/bin/env python3
import os
import argparse
import json
import random
from typing import List, Any

from PIL import Image

import torch
from datasets import Dataset

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

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
    p.add_argument('--max_steps', type=int, default=-1, help="training steps; if -1, computed automatically")
    p.add_argument('--logging_steps', type=int, default=100)
    p.add_argument('--eval_steps', type=int, default=100)
    p.add_argument('--save_steps', type=int, default=300)
    p.add_argument('--randomseed', type=int, default=42)
    p.add_argument('--pct_warmup', type=float, default=0.1, help="fraction for warmup steps")
    p.add_argument('--sanity_check', action='store_true')
    return p.parse_args()


args = parse_args()


# -------------------------
# Utility
# -------------------------
def parse_bboxes(bbox_field: Any) -> List[List[float]]:
    if bbox_field is None:
        return []
    if isinstance(bbox_field, str):
        try:
            return json.loads(bbox_field)
        except Exception:
            try:
                return eval(bbox_field)
            except Exception:
                return []
    if isinstance(bbox_field, (list, tuple)):
        return list(bbox_field)
    return []

# -------------------------
# Collator — loads images on the fly
# -------------------------
class QwenDPOCollator:
    def __init__(self, processor, max_prompt_length=512, max_length=2048):
        self.processor = processor
        self.max_prompt_length = max_prompt_length
        self.max_length = max_length

    def __call__(self, features):
        # Features contain:
        # ['prompt', 'metadata', 'images', 'prompt_input_ids', 'pixel_values', 'chosen_input_ids', 'rejected_input_ids']
        batch_prompts = []
        batch_images = []
        batch_prompt_ids = []
        batch_chosen_ids = []
        batch_rejected_ids = []

        for ex in features:
            prompt = ex["prompt"]
            batch_prompts.append(prompt)

            # save prompt token ids directly if available
            if "prompt_input_ids" in ex:
                batch_prompt_ids.append(ex["prompt_input_ids"])

            # load images from paths
            imgs = []
            for img_path in ex.get("images", []):
                if img_path is None or img_path == "":
                    continue
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception as e:
                    raise RuntimeError(f"Could not open image {img_path}: {e}")
                imgs.append(img)

            batch_images.append(imgs)
            batch_chosen_ids.append(ex["chosen_input_ids"])
            batch_rejected_ids.append(ex["rejected_input_ids"])

        # Process chosen inputs
        chosen_inputs = self.processor(
            images=batch_images,
            input_ids=batch_chosen_ids,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        # Process rejected inputs
        rejected_inputs = self.processor(
            images=batch_images,
            input_ids=batch_rejected_ids,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        # Process prompt-only inputs (no continuation)
        prompt_inputs = self.processor(
            images=batch_images,
            input_ids=batch_prompt_ids,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length
        ) if batch_prompt_ids else None

        return {
            # raw fields
            "prompt": batch_prompts,
            "images": batch_images,

            # prompt-only tokenized
            "prompt_input_ids": prompt_inputs["input_ids"] if prompt_inputs else None,
            "prompt_attention_mask": prompt_inputs["attention_mask"] if prompt_inputs else None,

            # chosen/rejected tokenized
            "chosen_input_ids": chosen_inputs["input_ids"],
            "chosen_attention_mask": chosen_inputs["attention_mask"],
            "rejected_input_ids": rejected_inputs["input_ids"],
            "rejected_attention_mask": rejected_inputs["attention_mask"],

            # vision inputs
            "pixel_values": prompt_inputs["pixel_values"],
            "image_grid_thw": prompt_inputs["image_grid_thw"],
        }
    
# -------------------------
# Load dataset JSON into HuggingFace Dataset
# -------------------------
def load_json_dataset(json_path: str, percentage: float = 1.0, sanity_check=False) -> Dataset:
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

    for ex in data:
        if "metadata" not in ex:
            ex["metadata"] = {}
    return Dataset.from_list(data)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_json_dataset(args.dataset, percentage=args.percentage, sanity_check=args.sanity_check)
    print(f"Loaded {len(dataset)} examples.")

    def prepare_for_trl(example):
        """
        Convert each raw example into the fields DPOTrainer expects:
        - prompt (str)
        - chosen (str)
        - rejected (str)
        - images (list of image_path strings)  # keep strings in dataset
        - metadata (keep original metadata if needed)
        """
        # get base strings
        prompt = example.get("prompt", "") or ""
        chosen = example.get("chosen", "") or ""
        rejected = example.get("rejected", "") or ""

        # ensure prompt ends with Answer:
        prompt = prompt.strip()
        if not prompt.endswith("Answer:"):
            prompt = prompt.rstrip() + "\n\nAnswer:"

        # metadata may include image_path and bbox
        meta = example.get("metadata", {}) or {}
        image_path = meta.get("image_path", None)
        bbox =  meta.get("bbox", None)

        if bbox:
            prompt = prompt.rstrip() + f"\n\nFocus on region: {bbox}"
        # Resolve relative path using dataset file base dir if necessary
        if image_path:
            image_path = image_path.replace("\\", os.sep).replace("/", os.sep)

        # store images as a list of strings (paths). DPOTrainer will call the processor later
        images = [Image.open(image_path).convert("RGB")] if image_path else []

        # return only the expected columns (avoid carrying heavy objects)
        return {
            "prompt": [{"content": [{"text": None, "type": "image"}, {"text": prompt,"type": "text"}],"role": "user"}],
            "chosen": [{"content": [{"text": str(chosen),"type": "text"}],"role": "user"}],
            "rejected": [{"content": [{"text": str(rejected),"type": "text"}],"role": "user"}],
            "images": images,
        }

    print("Normalizing dataset to TRL schema...")
    # map the dataset (do not remove metadata here; we keep it)
    dataset = dataset.map(prepare_for_trl)
    print("After normalization — example keys:", dataset.column_names)

    print("Loading processor & model...")
    MIN_PIXELS = 256 * 28 * 28
    MAX_PIXELS = 1280 * 28 * 28
    processor = Qwen2_5_VLProcessor.from_pretrained(args.base_model, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,   # or float16 if you’re on GPU
        low_cpu_mem_usage=True
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # train/test split
    dataset = dataset.train_test_split(test_size=0.1, seed=args.randomseed)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # steps
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = max(1, int(round(len(train_dataset) / (4 * args.bs)) * 3))

    warmup_steps = max(10, int(round(args.pct_warmup * total_steps)))
    save_steps = args.save_steps
    eval_steps = args.eval_steps
    if save_steps % eval_steps != 0:
        save_steps = ((save_steps // eval_steps) + 1) * eval_steps
    save_steps = min(save_steps, total_steps)

    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}, Save steps: {save_steps}, Eval steps: {eval_steps}")

    # check LoRA target modules dynamically
    all_modules = [n for n, _ in model.named_modules()]
    target_modules = [m for m in ["q_proj", "v_proj", "k_proj", "o_proj", "fc_in", "fc_out", "wte"] if any(m in n for n in all_modules)]
    print("LoRA target modules:", target_modules)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

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
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
        remove_unused_columns=False,
        run_name="dpo_qwen2vl_run",
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

    print("Initializing DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
    )

    print("Starting training...")
    dpo_trainer.train()
    print("Training finished.")

    print("Saving model and processor...")
    os.makedirs(args.output_dir, exist_ok=True)
    dpo_trainer.save_model(args.output_dir)
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint"))
    processor.save_pretrained(os.path.join(args.output_dir, "processor"))
    print(f"Saved to {args.output_dir}")
