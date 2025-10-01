import os
import json
import argparse
import random
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from PIL import Image
from peft import LoraConfig
from transformers import HfArgumentParser, AutoTokenizer
from trl import DPOTrainer, DPOConfig

from transformers import LlavaForConditionalGeneration, LlavaProcessor


# ----------------------------
# 1. Parse CLI args
# ----------------------------
def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--percentage', type=float, default=1)
    args.add_argument('--output_dir', type=str, default="./results_llava_dpo")
    args.add_argument('--base_model', type=str, default="liuhaotian/llava-v1.5-7b")
    args.add_argument('--wandb_name', type=str, default='dpo_llava')
    args.add_argument('--dataset', type=str, default='hotpotqa_7b_data.json')
    args.add_argument('--bs', type=int, default=2)
    args.add_argument('--lora_r', type=int, default=8)
    args.add_argument('--mixed', type=bool, default=False)
    args.add_argument('--randomseed', type=int, default=42)
    return args.parse_args()


args = parse_args()
pct = args.percentage
bs = args.bs
r = args.lora_r
mixed = args.mixed


# ----------------------------
# 2. ScriptArguments
# ----------------------------
@dataclass
class ScriptArguments:
    beta: Optional[float] = field(default=0.2)

    base_model: Optional[str] = field(default=args.base_model)
    learning_rate: Optional[float] = field(default=5e-6)
    lr_scheduler_type: Optional[str] = field(default="cosine")
    warmup_steps: Optional[int] = field(default=100)
    weight_decay: Optional[float] = field(default=0.0)
    optimizer_type: Optional[str] = field(default="adamw_torch")
    mixed: Optional[bool] = field(default=mixed)
    per_device_train_batch_size: Optional[int] = field(default=bs)
    per_device_eval_batch_size: Optional[int] = field(default=bs)
    randomseed: Optional[int] = field(default=0)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    gradient_checkpointing: Optional[bool] = field(default=True)

    lora_alpha: Optional[float] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_r: Optional[int] = field(default=r)

    max_prompt_length: Optional[int] = field(default=512)
    max_length: Optional[int] = field(default=1024)
    max_steps: Optional[int] = field(default=900)
    logging_steps: Optional[int] = field(default=100)
    save_steps: Optional[int] = field(default=300)
    eval_steps: Optional[int] = field(default=100)
    wandb_name: Optional[str] = field(default="dpo_llava")
    dataset: Optional[str] = field(default="hotpotqa_7b_data.json")
    output_dir: Optional[str] = field(default="./results_llava_dpo")

    sanity_check: Optional[bool] = field(default=False)
    report_to: Optional[str] = field(default="wandb")


# ----------------------------
# 3. Dataset Validation
# ----------------------------
def validate_dataset_format(dataset: Dataset) -> bool:
    required_columns = ['prompt', 'chosen', 'rejected', 'metadata']
    if not all(col in dataset.column_names for col in required_columns):
        print(f"Error: Dataset missing required columns. Found: {dataset.column_names}")
        print(f"Required: {required_columns}")
        return False
    return True


# ----------------------------
# 4. Collator with bbox
# ----------------------------
def multimodal_collator(batch, processor):
    prompts = [b["prompt"] for b in batch]
    chosen = [b["chosen"] for b in batch]
    rejected = [b["rejected"] for b in batch]

    image_paths = [b["metadata"]["image_path"] for b in batch]
    bboxes = []
    for b in batch:
        bbox_str = b["metadata"].get("bbox", None)
        try:
            bboxes.append(ast.literal_eval(bbox_str) if bbox_str else None)
        except Exception:
            bboxes.append(None)

    # Load images
    images = [Image.open(path).convert("RGB") for path in image_paths]

    # Append bbox to prompts
    prompts_with_bbox = []
    for p, bb in zip(prompts, bboxes):
        if bb:
            prompts_with_bbox.append(f"{p}\n\nFocus on regions: {bb}")
        else:
            prompts_with_bbox.append(p)

    # Encode chosen/rejected with processor
    chosen_inputs = processor(images, prompts_with_bbox, text=chosen, padding=True, return_tensors="pt")
    rejected_inputs = processor(images, prompts_with_bbox, text=rejected, padding=True, return_tensors="pt")

    return {
        "prompts": prompts_with_bbox,
        "images": images,
        "bboxes": bboxes,
        "chosen_inputs": chosen_inputs,
        "rejected_inputs": rejected_inputs,
    }


# ----------------------------
# 5. Main
# ----------------------------
if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Load LLaVA model & processor
    print("===== Load LLaVA model =====")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    processor = LlavaProcessor.from_pretrained(args.base_model)

    # 2. Load dataset
    print("===== Load dataset =====")
    with open(args.dataset, 'r') as f:
        ori_dataset = json.load(f)

    len_data = round(len(ori_dataset) * pct)
    if pct < 1:
        random.seed(args.randomseed)
        random_numbers = random.sample(range(0, len(ori_dataset)), len_data)
        ori_dataset = [d for i, d in enumerate(ori_dataset) if i in random_numbers]
    else:
        ori_dataset = ori_dataset[:len_data]

    print('number of paired_data:', len(ori_dataset))

    data_dict = {key: [item[key] for item in ori_dataset] for key in ori_dataset[0]}
    dataset = Dataset.from_dict(data_dict)

    if not validate_dataset_format(dataset):
        exit(1)

    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    warmup_steps = max(10, round(0.1 * len(train_dataset) / (4 * bs)))
    total_steps = round(len(train_dataset) / (4 * bs)) * 3
    save_steps = max(script_args.eval_steps, round(len(train_dataset) / (4 * bs) * 0.5))

    # 3. Config
    print("===== DPO config =====")
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=total_steps,
        logging_steps=script_args.logging_steps,
        save_steps=save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        eval_strategy="steps",
        eval_steps=script_args.eval_steps,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=args.wandb_name,
        beta=script_args.beta,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        logging_first_step=True,
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc_in", "fc_out", "wte"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Trainer
    print("===== Init DPO trainer =====")
    dpo_trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
        data_collator=lambda b: multimodal_collator(b, processor),
    )

    # 5. Train
    print("===== Training =====")
    dpo_trainer.train()

    # 6. Save
    print("===== Save =====")
    dpo_trainer.save_model(script_args.output_dir)
    print(f"Model saved to {script_args.output_dir}")
