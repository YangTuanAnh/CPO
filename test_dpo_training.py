#!/usr/bin/env python3
"""
Test script for the updated DPO training script
"""

import json
import os
from datasets import Dataset

def create_test_dataset():
    """Create a small test dataset for validation"""
    test_data = [
        {
            "prompt": "Question: What is the capital of France?\n\nAnswer:",
            "chosen": "The capital of France is Paris.",
            "rejected": "The capital of France is London."
        },
        {
            "prompt": "Question: What is 2+2?\n\nAnswer:",
            "chosen": "2+2 equals 4.",
            "rejected": "2+2 equals 5."
        },
        {
            "prompt": "Question: What is the largest planet?\n\nAnswer:",
            "chosen": "Jupiter is the largest planet in our solar system.",
            "rejected": "Earth is the largest planet in our solar system."
        }
    ]
    
    # Save test dataset
    with open('test_dpo_dataset.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print("Test dataset created: test_dpo_dataset.json")
    return test_data

def test_dpo_config():
    """Test DPOConfig initialization"""
    try:
        from trl import DPOConfig
        
        config = DPOConfig(
            output_dir="./test_dpo_output",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=1,
            learning_rate=5e-6,
            beta=0.1,
            max_prompt_length=512,
            max_length=1024,
            logging_steps=1,
            save_steps=10,
            evaluation_strategy="steps",
            eval_steps=10,
            report_to="none",
            bf16=False,  # Use fp32 for testing
            remove_unused_columns=False,
        )
        
        print("DPOConfig created successfully!")
        print(f"Output dir: {config.output_dir}")
        print(f"Beta: {config.beta}")
        print(f"Max prompt length: {config.max_prompt_length}")
        return True
        
    except Exception as e:
        print(f"Error creating DPOConfig: {e}")
        return False

def test_dataset_validation():
    """Test dataset validation function"""
    try:
        # Import the validation function
        import sys
        sys.path.append('.')
        from dpo_training import validate_dataset_format
        
        # Create test dataset
        test_data = create_test_dataset()
        dataset = Dataset.from_dict({
            'prompt': [item['prompt'] for item in test_data],
            'chosen': [item['chosen'] for item in test_data],
            'rejected': [item['rejected'] for item in test_data]
        })
        
        # Test validation
        result = validate_dataset_format(dataset)
        print(f"Dataset validation result: {result}")
        return result
        
    except Exception as e:
        print(f"Error testing dataset validation: {e}")
        return False

def main():
    """Main test function"""
    print("=== Testing Updated DPO Training Script ===\n")
    
    # Test 1: DPOConfig
    print("1. Testing DPOConfig...")
    config_ok = test_dpo_config()
    
    # Test 2: Dataset validation
    print("\n2. Testing dataset validation...")
    validation_ok = test_dataset_validation()
    
    # Test 3: Check if we can import the main script
    print("\n3. Testing script imports...")
    try:
        import dpo_training
        print("DPO training script imports successfully!")
        import_ok = True
    except Exception as e:
        print(f"Error importing DPO training script: {e}")
        import_ok = False
    
    # Summary
    print(f"\n=== Test Results ===")
    print(f"DPOConfig: {'✓' if config_ok else '✗'}")
    print(f"Dataset validation: {'✓' if validation_ok else '✗'}")
    print(f"Script imports: {'✓' if import_ok else '✗'}")
    
    if all([config_ok, validation_ok, import_ok]):
        print("\nAll tests passed! The updated DPO training script is ready to use.")
    else:
        print("\nSome tests failed. Please check the errors above.")
    
    # Cleanup
    if os.path.exists('test_dpo_dataset.json'):
        os.remove('test_dpo_dataset.json')
        print("\nTest files cleaned up.")

if __name__ == "__main__":
    main()
