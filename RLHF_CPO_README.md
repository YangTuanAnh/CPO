# RLHF Dataset Processing for CPO Training

This repository contains tools to process RLHF (Reinforcement Learning from Human Feedback) datasets with hierarchical Q&A structures for CPO (Constrained Policy Optimization) training.

## Dataset Structure

The dataset follows a tree-like structure where:

- **Q1-Q4**: Main questions in sequence
- **A1-A4**: Corresponding answers to Q1-Q4
- **Q2.1, Q2.2, Q2.3**: Variations of Q2 (and similar for other questions)
- **A2.1, A2.2, A2.3**: Corresponding answers to the variations
- **QA2.1_completion_helpful/honest/harmless**: RLHF ratings for each variation

## Files Overview

### Core Processing
- `read_rlhf_dataset.py`: Main processing script with RLHFDatasetProcessor class
- `test_rlhf_processor.py`: Test script to validate the processing pipeline
- `integrate_cpo_training.py`: Integration script for CPO training setup

### Generated Output
- `cpo_training_data.json`: Full processed dataset
- `cpo_train_data.json`: Training split (90%)
- `cpo_test_data.json`: Test split (10%)
- `cpo_training_config.json`: Training configuration

## Usage

### 1. Basic Processing

```python
from read_rlhf_dataset import RLHFDatasetProcessor
import pandas as pd

# Load your dataset
df = pd.read_parquet("your_dataset.parquet")

# Initialize processor
processor = RLHFDatasetProcessor(df)

# Process dataset
pairs = processor.process_dataset(sample_size=1000)  # Process first 1000 rows

# Save results
processor.save_processed_data(pairs, 'cpo_data.json')
```

### 2. Command Line Processing

```bash
# Process dataset with default settings
python integrate_cpo_training.py --dataset_path "your_dataset.parquet" --sample_size 1000

# Process with custom output directory
python integrate_cpo_training.py --dataset_path "your_dataset.parquet" --output_dir "./my_cpo_data" --sample_size 5000
```

### 3. Testing the Processor

```bash
# Run tests with mock data
python test_rlhf_processor.py
```

## Data Processing Pipeline

### 1. Hierarchy Extraction
The processor extracts Q&A hierarchies from each row:
- Identifies main questions (Q1-Q4) and answers (A1-A4)
- Finds question variations (Q2.1, Q2.2, Q2.3, etc.)
- Extracts corresponding answers (A2.1, A2.2, A2.3, etc.)
- Collects RLHF ratings for each variation

### 2. Preference Pair Creation
For each question with variations:
- Creates prompts in format: "Question: [question]\n\nAnswer:"
- Compares main answer with variation answers using RLHF ratings
- Generates chosen/rejected pairs based on composite scores
- Includes metadata with individual and composite ratings

### 3. Output Format
Each preference pair contains:
```json
{
  "prompt": "Question: [question]\n\nAnswer:",
  "chosen": "[preferred answer]",
  "rejected": "[less preferred answer]",
  "metadata": {
    "question_id": "[original question]",
    "variation_id": 1,
    "scores": {
      "helpful": 0.8,
      "honest": 0.7,
      "harmless": 0.9,
      "composite": 0.8
    }
  }
}
```

## Integration with CPO Training

The processed data is compatible with the existing CPO training pipeline:

```python
# Load processed data
with open('cpo_train_data.json', 'r') as f:
    train_data = json.load(f)

# Convert to HuggingFace Dataset
from datasets import Dataset
dataset = Dataset.from_dict({
    'prompt': [item['prompt'] for item in train_data],
    'chosen': [item['chosen'] for item in train_data],
    'rejected': [item['rejected'] for item in train_data]
})

# Use with your CPO trainer
```

## Configuration

### Training Parameters
The integration script generates a configuration file with recommended settings:
- Batch size: 4
- Learning rate: 5e-6
- Epochs: 3
- Warmup steps: 100
- Evaluation every 100 steps

### Customization
You can modify the processing logic by:
- Adjusting the composite score calculation in `create_cpo_pairs()`
- Changing the preference threshold (currently 0.5)
- Modifying the prompt format
- Adding additional metadata fields

## Example Output

After processing, you'll get preference pairs like:

```
Prompt: Question: Explain photosynthesis
Answer:
Chosen: Through photosynthesis using sunlight
Rejected: Process by which plants convert light to energy
Scores: {'helpful': 0.8, 'honest': 0.7, 'harmless': 0.9, 'composite': 0.8}
```

## Troubleshooting

### Common Issues
1. **No pairs generated**: Check if your data has the expected column structure
2. **Missing ratings**: Ensure RLHF rating columns exist and contain valid values
3. **Memory issues**: Use `sample_size` parameter to process data in chunks

### Validation
The test script validates:
- Data extraction accuracy
- Pair generation logic
- Output format compliance
- Train/test split functionality

## Requirements

- pandas
- datasets (HuggingFace)
- json
- typing
- dataclasses

## License

This code is provided as-is for research and educational purposes.
