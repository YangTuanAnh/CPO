# cpo_multimodal_test.py
import json
import argparse
import time
import ast
import os
import warnings

import torch
import torch.nn.functional as F
from PIL import Image
from datasets import load_dataset

from transformers import (
    CLIPModel,
    CLIPTokenizer,
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM,
)

# LLaVA imports (model + processor)
# adjust import names if your local package exposes different names
try:
    from transformers import LlavaForConditionalGeneration, LlavaProcessor
    LLAVA_AVAILABLE = True
except Exception:
    # If these imports fail, user must have transformers with LLaVA support in their env.
    LLAVA_AVAILABLE = False

warnings.filterwarnings("ignore")


# -----------------------------
# Utilities: CLIP init + similarity
# -----------------------------
def init_clip(device=None, clip_name="openai/clip-vit-base-patch32"):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_name)
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_model.eval()
    return clip_model, clip_tokenizer, device


def calculate_similarity_clip(text1, text2, clip_model, clip_tokenizer, device="cpu"):
    """Compute cosine similarity between CLIP text embeddings for text1 and text2."""
    if (not text1) and (not text2):
        return 1.0
    if (not text1) or (not text2):
        return 0.0

    texts = [text1, text2]
    inputs = clip_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clip_model.get_text_features(**inputs)  # (2, D)

    emb1 = F.normalize(outputs[0], p=2, dim=0)
    emb2 = F.normalize(outputs[1], p=2, dim=0)
    sim = torch.dot(emb1, emb2).item()
    return sim


# -----------------------------
# Multimodal preference test
# -----------------------------
def test_cpo_preference_accuracy_multimodal(
    model,
    tokenizer_or_processor,
    test_data,
    clip_model,
    clip_tokenizer,
    clip_device,
    device,
    is_llava: bool = False,
    num_samples: int = 100,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
):
    """
    Test CPO model on preference pairs for multimodal/text cases.
    - model: either text-only LM (AutoModelForCausalLM/LlamaForCausalLM) or LlavaForConditionalGeneration
    - tokenizer_or_processor: tokenizer for text LM, processor for Llava
    - test_data: list of dicts like your example (includes metadata.image_path, metadata.bbox)
    """
    model.eval()
    correct = 0
    total = 0
    results = []

    samples = test_data[:num_samples]
    print(f"Testing {len(samples)} samples (multimodal-aware)...")

    for i, sample in enumerate(samples):
        try:
            prompt = sample.get("prompt", "")
            chosen = sample.get("chosen", "")
            rejected = sample.get("rejected", "")

            # detect image presence in metadata
            metadata = sample.get("metadata", {}) or {}
            image_path = metadata.get("image_path", None)
            bbox_str = metadata.get("bbox", None)

            # append bbox to prompt if present
            if bbox_str:
                try:
                    # keep original format but parse safely
                    parsed_bbox = ast.literal_eval(bbox_str) if isinstance(bbox_str, str) else bbox_str
                except Exception:
                    parsed_bbox = bbox_str
                prompt_with_bbox = f"{prompt}\n\nFocus on regions: {parsed_bbox}"
            else:
                prompt_with_bbox = prompt

            # Build inputs and generate depending on model type
            generated_response = ""
            if is_llava and image_path:
                # Llava path: use the processor with image + prompt
                processor = tokenizer_or_processor  # for Llava, we passed processor here
                # load image
                if not os.path.exists(image_path):
                    print(f"Warning: image not found at {image_path} (sample {i}), skipping.")
                    total += 1
                    results.append({
                        'sample_id': i,
                        'prompt': prompt_with_bbox,
                        'generated': "",
                        'chosen': chosen,
                        'rejected': rejected,
                        'chosen_similarity': None,
                        'rejected_similarity': None,
                        'predicted_chosen': False,
                        'correct': False,
                        'error': 'image_missing'
                    })
                    continue

                image = Image.open(image_path).convert("RGB")

                # Processor typical call: processor(images=..., text=..., return_tensors="pt", padding=True)
                inputs = processor(images=image, text=prompt_with_bbox, return_tensors="pt", padding=True)
                # Move tensors to device
                inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

                with torch.no_grad():
                    # Generate with model. This API may vary by version; adapt as needed.
                    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature)
                # decode using model's tokenizer inside processor or model.config (processor has a tokenizer)
                if hasattr(processor, "tokenizer"):
                    generated_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    # fallback: try AutoTokenizer decode if available
                    try:
                        generated_text = tokenizer_or_processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    except Exception:
                        generated_text = "<could_not_decode>"
                # remove the prompt prefix if present
                if generated_text.startswith(prompt_with_bbox):
                    generated_response = generated_text[len(prompt_with_bbox):].strip()
                else:
                    # Try to remove just the "Question:" prefix if necessary
                    generated_response = generated_text.strip()

            else:
                # Text-only path (use tokenizer + text LLM)
                tokenizer = tokenizer_or_processor
                inputs = tokenizer(prompt_with_bbox, return_tensors="pt", truncation=True, max_length=1024)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # strip prompt prefix
                if generated_text.startswith(prompt_with_bbox):
                    generated_response = generated_text[len(prompt_with_bbox):].strip()
                else:
                    generated_response = generated_text.strip()

            # compute similarities via CLIP text encoder
            chosen_sim = calculate_similarity_clip(generated_response, chosen, clip_model, clip_tokenizer, clip_device)
            rejected_sim = calculate_similarity_clip(generated_response, rejected, clip_model, clip_tokenizer, clip_device)

            predicted_chosen = chosen_sim > rejected_sim
            is_correct = bool(predicted_chosen)

            if is_correct:
                correct += 1
            total += 1

            results.append({
                'sample_id': i,
                'prompt': prompt_with_bbox,
                'generated': generated_response,
                'chosen': chosen,
                'rejected': rejected,
                'chosen_similarity': chosen_sim,
                'rejected_similarity': rejected_sim,
                'predicted_chosen': predicted_chosen,
                'correct': is_correct
            })

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(samples)} samples. Current accuracy: {correct/total:.3f}")

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            # log the error and continue
            results.append({
                'sample_id': i,
                'prompt': sample.get('prompt', ''),
                'generated': '',
                'chosen': sample.get('chosen', ''),
                'rejected': sample.get('rejected', ''),
                'chosen_similarity': None,
                'rejected_similarity': None,
                'predicted_chosen': False,
                'correct': False,
                'error': str(e)
            })
            continue

    accuracy = correct / total if total > 0 else 0.0
    print(f"CPO Preference Accuracy: {accuracy:.3f} ({correct}/{total})")
    return accuracy, results


# -----------------------------
# CLI + main runner
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpo_test_data", type=str, default="cpo_processed_data/cpo_test_data.json")
    parser.add_argument("--cpo_model_path", type=str, default="", help="Path to multimodal model (LlAVA) or text-only LLM")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load test data
    with open(args.cpo_test_data, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples from {args.cpo_test_data}")

    # Initialize CLIP
    clip_model, clip_tokenizer, clip_device = init_clip(device=device)
    print("Loaded CLIP model on", clip_device)

    # Try to decide whether the model is multimodal (LlAVA) or text-only:
    is_llava = False
    model = None
    tokenizer_or_processor = None

    # Heuristic: if cpo_model_path provided and LLaVA available, load Llava
    if args.cpo_model_path and LLAVA_AVAILABLE:
        try:
            print("Attempting to load Llava model + processor from:", args.cpo_model_path)
            model = LlavaForConditionalGeneration.from_pretrained(args.cpo_model_path, torch_dtype=torch.bfloat16, device_map="auto")
            processor = LlavaProcessor.from_pretrained(args.cpo_model_path)
            model.to(device)
            tokenizer_or_processor = processor
            is_llava = True
            print("Loaded Llava model (multimodal) successfully.")
        except Exception as e:
            print("Could not load Llava model or processor:", e)
            print("Falling back to text-only model path if provided below.")

    # If not Llava, try load text-only Llama/AutoModel
    if model is None:
        try:
            print("Loading text-only tokenizer + model (Llama/AutoModel) from:", args.cpo_model_path or "distilgpt2 fallback")
            if args.cpo_model_path:
                try:
                    tokenizer = LlamaTokenizer.from_pretrained(args.cpo_model_path, padding_side="left")
                    model = AutoModelForCausalLM.from_pretrained(args.cpo_model_path, torch_dtype=torch.float16)
                except Exception:
                    # fallback to generic tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(args.cpo_model_path)
                    model = AutoModelForCausalLM.from_pretrained(args.cpo_model_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
                model = AutoModelForCausalLM.from_pretrained("distilgpt2")

            tokenizer.pad_token = tokenizer.eos_token
            model.to(device)
            tokenizer_or_processor = tokenizer
            is_llava = False
            print("Loaded text-only model successfully.")
        except Exception as e:
            print("Failed to load any model:", e)
            return

    # Run test
    accuracy, results = test_cpo_preference_accuracy_multimodal(
        model=model,
        tokenizer_or_processor=tokenizer_or_processor,
        test_data=test_data,
        clip_model=clip_model,
        clip_tokenizer=clip_tokenizer,
        clip_device=clip_device,
        device=device,
        is_llava=is_llava,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    # Save output
    out_path = "cpo_multimodal_test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "num_samples_tested": args.num_samples,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
