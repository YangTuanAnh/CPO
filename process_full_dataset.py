#!/usr/bin/env python3
"""
Process the full RLHF dataset for CPO training
"""

import pandas as pd
import json
import os
from read_rlhf_dataset import RLHFDatasetProcessor

def process_full_dataset():
    """Process the full dataset for CPO training"""
    print("=== Processing Full RLHF Dataset for CPO Training ===\n")
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_parquet("hf://datasets/wnkh/vlm-project-with-images-with-bbox-images-with-tree-of-thoughts-RLHF-v6/data/train-00000-of-00001.parquet")
    
    print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Check available languages
    available_languages = ['en']  # English is always available
    lang_columns = [col for col in df.columns if col in ['vn', 'fr', 'de', 'mandarin', 'korean', 'japanese', 'vi']]
    if lang_columns:
        available_languages.extend(lang_columns)
        print(f"Available languages: {available_languages}")
    
    # Initialize processor
    processor = RLHFDatasetProcessor(df)
    
    # Process dataset
    print("\nProcessing dataset...")
    pairs = processor.process_dataset(sample_size=None, languages=['en'])  # Process all rows
    
    if pairs:
        print(f"\nProcessing complete!")
        print(f"Total pairs generated: {len(pairs)}")
        
        # Create output directory
        output_dir = "cpo_processed_data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full dataset
        full_file = os.path.join(output_dir, "cpo_full_data.json")
        processor.save_processed_data(pairs, full_file)
        
        # Create train/test split
        train_pairs, test_pairs = processor.create_train_test_split(pairs, test_ratio=0.1)
        
        # Save splits
        train_file = os.path.join(output_dir, "cpo_train_data.json")
        test_file = os.path.join(output_dir, "cpo_test_data.json")
        
        processor.save_processed_data(train_pairs, train_file)
        processor.save_processed_data(test_pairs, test_file)
        
        print(f"Training pairs: {len(train_pairs)}")
        print(f"Test pairs: {len(test_pairs)}")
        print(f"Files saved to: {output_dir}")
        
        # Show statistics
        print(f"\nDataset Statistics:")
        print(f"Total pairs: {len(pairs)}")
        print(f"Average pairs per row: {len(pairs) / len(df):.2f}")
        
        # Check rating distribution
        ratings = [pair['metadata']['scores']['composite'] for pair in pairs]
        print(f"Rating distribution:")
        print(f"  Min: {min(ratings):.3f}")
        print(f"  Max: {max(ratings):.3f}")
        print(f"  Mean: {sum(ratings)/len(ratings):.3f}")
        
        # Check data quality
        empty_chosen = sum(1 for p in pairs if not p['chosen'].strip())
        empty_rejected = sum(1 for p in pairs if not p['rejected'].strip())
        print(f"  Empty chosen answers: {empty_chosen}")
        print(f"  Empty rejected answers: {empty_rejected}")
        
        # Show sample pairs
        print(f"\nSample pairs:")
        for i, pair in enumerate(pairs[:3]):
            print(f"\nPair {i+1}:")
            print(f"  Prompt: {pair['prompt'][:100]}...")
            print(f"  Chosen: {pair['chosen'][:100]}...")
            print(f"  Rejected: {pair['rejected'][:100]}...")
            print(f"  Language: {pair['metadata'].get('language', 'unknown')}")
            print(f"  Metric: {pair['metadata'].get('metric', 'unknown')}")
        
        # Create HuggingFace Dataset format
        print(f"\nCreating HuggingFace Dataset format...")
        from datasets import Dataset
        
        # Convert to HuggingFace format
        dataset_dict = {
            'prompt': [item['prompt'] for item in pairs],
            'chosen': [item['chosen'] for item in pairs],
            'rejected': [item['rejected'] for item in pairs]
        }
        
        hf_dataset = Dataset.from_dict(dataset_dict)
        hf_dataset.save_to_disk(os.path.join(output_dir, "cpo_hf_dataset"))
        print(f"HuggingFace dataset saved to: {os.path.join(output_dir, 'cpo_hf_dataset')}")
        
        # Create training configuration
        config = {
            "model_name": "meta-llama/Llama-2-7b-hf",
            "train_data": train_file,
            "eval_data": test_file,
            "output_dir": "./cpo_results",
            "num_train_samples": len(train_pairs),
            "num_eval_samples": len(test_pairs),
            "training_args": {
                "per_device_train_batch_size": 4,
                "per_device_eval_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "learning_rate": 5e-6,
                "num_train_epochs": 3,
                "warmup_steps": 100,
                "logging_steps": 10,
                "save_steps": 500,
                "eval_steps": 100,
                "evaluation_strategy": "steps",
                "save_strategy": "steps",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "bf16": True,
                "remove_unused_columns": False,
            },
            "cpo_args": {
                "beta": 0.1,
                "max_prompt_length": 512,
                "max_length": 1024,
            }
        }
        
        config_file = os.path.join(output_dir, "cpo_training_config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Training configuration saved to: {config_file}")
        
        print(f"\n=== Processing Complete ===")
        print(f"Ready for CPO training!")
        print(f"Use the following command to start training:")
        print(f"python dpo_training.py --dataset {train_file} --base_model meta-llama/Llama-2-7b-hf")
        
    else:
        print("No pairs generated. Check data format and processing logic.")

if __name__ == "__main__":
    process_full_dataset()
