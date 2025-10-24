import pandas as pd
import json
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import os
from PIL import Image
import io

@dataclass
class QAItem:
    """Represents a single Q&A item with its variations and ratings"""
    question: str
    answer: str
    variations: List[Dict[str, str]]  # List of {question: str, answer: str}
    ratings: Dict[str, float]  # helpful, honest, harmless ratings

class RLHFDatasetProcessor:
    """Processes RLHF dataset with tree structure for CPO training + image support"""
    
    def __init__(self, df: pd.DataFrame, image_dir: str = "images"):
        self.df = df
        self.processed_data = []
        self.image_dir = image_dir
        os.makedirs(self.image_dir, exist_ok=True)

    def _save_image(self, img_data: Any, row_id: int) -> Optional[str]:
        """
        Save image from dataframe row to images folder.
        Supports base64, PIL image, or numpy array.
        """
        try:
            img_path = os.path.join(self.image_dir, f"img_{row_id}.png")
            img = Image.open(io.BytesIO(img_data['bytes']))
            img.save(img_path)
            return img_path
        except Exception as e:
            print(f"Could not save image for row {row_id}: {e}")
            return None
    
    def process_dataset(self, sample_size: Optional[int] = None, languages: List[str] = None) -> List[Dict[str, Any]]:
        """Process dataset including Q&A, ratings, image, and bbox"""
        if languages is None:
            languages = ['en']  # Default English

        all_rows = []
        processed_rows = 0

        main_questions = ['Q1', 'Q2', 'Q3', 'Q4']
        main_answers = ['A1', 'A2', 'A3', 'A4']

        for idx, row in self.df.iterrows():
            try:
                qa_item = {}
                # Save image if present
                image_path = None
                if "image" in row and pd.notna(row["image"]):
                    self._save_image(row["image"], idx)
                    image_path = f"img_{idx}.png"

                qa_item['image'] = image_path
                qa_item['conversations'] = []

                for q_name, a_name in zip(main_questions, main_answers):
                    question = row.get(q_name, "").strip()
                    if (q_name == "Q1"): question = "<image>\n" + question 
                    answer = row.get(a_name, "").strip()
                    qa_item['conversations'].append({"from": "human", "value": question})
                    qa_item['conversations'].append({"from": "gpt", "value": answer})

                all_rows.append(qa_item)
                
                if all_rows:
                    processed_rows += 1
                    if sample_size and processed_rows >= sample_size:
                        break
                        
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

        print(f"Final: Processed {processed_rows} rows, generated {len(all_rows)} conversations")
        return all_rows
    
    def save_processed_data(self, pairs: List[Dict[str, Any]], output_file: str):
        """Save processed data to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(pairs)} pairs to {output_file}")
    
    def create_train_test_split(self, pairs: List[Dict[str, Any]], test_ratio: float = 0.1):
        """Split data into train and test sets"""
        random.shuffle(pairs)
        split_idx = int(len(pairs) * (1 - test_ratio))
        
        train_pairs = pairs[:split_idx]
        test_pairs = pairs[split_idx:]
        
        return train_pairs, test_pairs

def main():
    """Main processing function"""
    # Load dataset
    print("Loading dataset...")
    df = pd.read_parquet("hf://datasets/wnkh/vlm-project-with-images-with-bbox-images-with-tree-of-thoughts-RLHF-v6/data/train-00000-of-00001.parquet")

    print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
    print("Sample columns:", df.columns[:10].tolist())
    
    # Check for available languages
    available_languages = ['en']  # English is always available
    lang_columns = [col for col in df.columns if col in ['vn', 'fr', 'de', 'mandarin', 'korean', 'japanese', 'vi']]
    if lang_columns:
        available_languages.extend(lang_columns)
        print(f"Available languages: {available_languages}")
    
    # Initialize processor
    processor = RLHFDatasetProcessor(df)
    
    # Process dataset (you can specify sample_size for testing)
    # For now, process only English to avoid too many pairs
    conversations = processor.process_dataset(sample_size=3000, languages=['en'])  # Process first 1000 rows for testing
    
    if conversations:
        # Save full dataset
        processor.save_processed_data(conversations, 'sft_training_data.json')
        
        # Create train/test split
        train_convos, test_convos = processor.create_train_test_split(conversations)
        
        # Save splits
        processor.save_processed_data(train_convos, 'sft_train_data.json')
        processor.save_processed_data(test_convos, 'sft_test_data.json')
        
        print(f"Training pairs: {len(train_convos)}")
        print(f"Test pairs: {len(test_convos)}")
        # Show sample
        print("\nSample pair:")
        print(json.dumps(conversations[0], indent=2))
        
        # Show statistics
        print(f"\nDataset Statistics:")
        print(f"Total pairs: {len(conversations)}")
        print(f"Average pairs per row: {len(conversations) / 1000:.2f}")
        
    else:
        print("No conversations generated. Check data format and processing logic.")

if __name__ == "__main__":
    main()