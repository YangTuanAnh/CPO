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
        
    def extract_qa_hierarchy(self, row: pd.Series, language: str = 'en') -> List[QAItem]:
        """Extract Q&A hierarchy from a single row"""
        qa_items = []
        
        # Main questions Q1-Q4
        main_questions = ['Q1', 'Q2', 'Q3', 'Q4']
        main_answers = ['A1', 'A2', 'A3', 'A4']
        
        # Handle multilingual data
        if language != 'en':
            lang_suffix = f'_{language}'
            main_questions = [f'{q}{lang_suffix}' for q in main_questions]
            main_answers = [f'{a}{lang_suffix}' for a in main_answers]
        
        for i, (q_col, a_col) in enumerate(zip(main_questions, main_answers)):
            if pd.notna(row[q_col]) and pd.notna(row[a_col]):
                # Get variations for this question
                variations = self._get_question_variations(row, i+1, language)
                
                # Get ratings for this question
                ratings = self._get_question_ratings(row, i+1)
                
                qa_item = QAItem(
                    question=str(row[q_col]).strip(),
                    answer=str(row[a_col]).strip(),
                    variations=variations,
                    ratings=ratings
                )
                qa_items.append(qa_item)
        
        return qa_items
    
    def _get_question_variations(self, row: pd.Series, question_num: int, language: str = 'en') -> List[Dict[str, str]]:
        """Get question variations (e.g., Q2.1, Q2.2, Q2.3) and their answers"""
        variations = []
        
        # Look for variations like Q2.1, Q2.2, Q2.3, etc.
        for i in range(1, 4):  # Assuming max 3 variations per question
            q_col = f'Q{question_num}.{i}'
            a_col = f'A{question_num}.{i}'
            
            # Handle multilingual data
            if language != 'en':
                lang_suffix = f'_{language}'
                q_col = f'{q_col}{lang_suffix}'
                a_col = f'{a_col}{lang_suffix}'
            
            if q_col in row.index and a_col in row.index:
                if pd.notna(row[q_col]) and pd.notna(row[a_col]):
                    variations.append({
                        'question': str(row[q_col]).strip(),
                        'answer': str(row[a_col]).strip()
                    })
        
        return variations
    
    def _get_question_ratings(self, row: pd.Series, question_num: int) -> Dict[str, float]:
        """Extract RLHF ratings for a question"""
        ratings = {}
        
        # Look for rating columns like QA2.1_completion_helpful, etc.
        for metric in ['helpful', 'honest', 'harmless']:
            for i in range(1, 4):  # Check variations
                # Try completion ratings first, then rm_output ratings
                for rating_type in ['completion', 'rm_output']:
                    rating_col = f'QA{question_num}.{i}_{rating_type}_{metric}'
                    if rating_col in row.index and pd.notna(row[rating_col]):
                        try:
                            # Check if it's a numerical rating
                            rating_value = float(row[rating_col])
                            key = f'variation_{i}_{metric}'
                            if key not in ratings or rating_type == 'completion':
                                ratings[key] = rating_value
                        except (ValueError, TypeError):
                            # If not numerical, try to extract from text format
                            rating_text = str(row[rating_col])
                            if '<chosen:' in rating_text and '<reject:' in rating_text:
                                # This is a preference pair, assign a default score
                                key = f'variation_{i}_{metric}'
                                if key not in ratings or rating_type == 'completion':
                                    ratings[key] = 0.8  # Default high score for chosen responses
                            continue
        
        return ratings
    
    def extract_preference_pairs_from_ratings(self, row: pd.Series, question_num: int) -> List[Dict[str, Any]]:
        """Extract preference pairs directly from rating columns that contain chosen/rejected text"""
        pairs = []
        
        for i in range(1, 4):  # Check variations
            for metric in ['helpful', 'honest', 'harmless']:
                for rating_type in ['completion', 'rm_output']:
                    rating_col = f'QA{question_num}.{i}_{rating_type}_{metric}'
                    if rating_col in row.index and pd.notna(row[rating_col]):
                        rating_text = str(row[rating_col])
                        if '<chosen:' in rating_text and '<reject:' in rating_text:
                            try:
                                # Extract chosen and rejected responses
                                chosen_start = rating_text.find('<chosen:') + 8
                                chosen_end = rating_text.find('</chosen:')
                                chosen_text = rating_text[chosen_start:chosen_end].strip()
                                
                                reject_start = rating_text.find('<reject:') + 8
                                reject_end = rating_text.find('</reject:')
                                reject_text = rating_text[reject_start:reject_end].strip()
                                
                                # Remove A> and B> prefixes if present
                                if chosen_text.startswith('A>'):
                                    chosen_text = chosen_text[2:].strip()
                                if reject_text.startswith('B>'):
                                    reject_text = reject_text[2:].strip()
                                
                                if chosen_text and reject_text and chosen_text.strip() != 'N/A' and reject_text.strip() != 'N/A':
                                    # Get the original question
                                    q_col = f'Q{question_num}'
                                    if q_col in row.index and pd.notna(row[q_col]):
                                        prompt = row[q_col]
                                        
                                        pairs.append({
                                            'prompt': prompt,
                                            'chosen': chosen_text,
                                            'rejected': reject_text,
                                            'metadata': {
                                                'question_id': str(row[q_col]),
                                                'variation_id': i,
                                                'metric': metric,
                                                'rating_type': rating_type,
                                                'scores': {
                                                    'helpful': 0.8 if metric == 'helpful' else 0.5,
                                                    'honest': 0.8 if metric == 'honest' else 0.5,
                                                    'harmless': 0.8 if metric == 'harmless' else 0.5,
                                                    'composite': 0.8
                                                }
                                            }
                                        })
                            except Exception as e:
                                print(f"Error parsing rating text: {e}")
                                continue
        
        return pairs
    
    def create_cpo_pairs(self, qa_items: List[QAItem]) -> List[Dict[str, Any]]:
        """Create preference pairs for CPO training"""
        pairs = []
        
        for qa_item in qa_items:
            # Create pairs from main question and its variations
            for i, variation in enumerate(qa_item.variations):
                # Create prompt for the main question
                prompt = qa_item.question
                
                # Get ratings for this variation
                helpful_key = f'variation_{i+1}_helpful'
                honest_key = f'variation_{i+1}_honest'
                harmless_key = f'variation_{i+1}_harmless'
                
                if all(key in qa_item.ratings for key in [helpful_key, honest_key, harmless_key]):
                    # Calculate composite score
                    composite_score = (
                        qa_item.ratings[helpful_key] + 
                        qa_item.ratings[honest_key] + 
                        qa_item.ratings[harmless_key]
                    ) / 3
                    
                    # Create preference pair
                    if composite_score > 0.5:  # Threshold for preference
                        pairs.append({
                            'prompt': prompt,
                            'chosen': variation['answer'],
                            'rejected': qa_item.answer,
                            'metadata': {
                                'question_id': qa_item.question,
                                'variation_id': i+1,
                                'scores': {
                                    'helpful': qa_item.ratings[helpful_key],
                                    'honest': qa_item.ratings[honest_key],
                                    'harmless': qa_item.ratings[harmless_key],
                                    'composite': composite_score
                                }
                            }
                        })
                    else:
                        pairs.append({
                            'prompt': prompt,
                            'chosen': qa_item.answer,
                            'rejected': variation['answer'],
                            'metadata': {
                                'question_id': qa_item.question,
                                'variation_id': i+1,
                                'scores': {
                                    'helpful': qa_item.ratings[helpful_key],
                                    'honest': qa_item.ratings[honest_key],
                                    'harmless': qa_item.ratings[harmless_key],
                                    'composite': composite_score
                                }
                            }
                        })
        
        return pairs
    
    def process_dataset(self, sample_size: Optional[int] = None, languages: List[str] = None) -> List[Dict[str, Any]]:
        """Process dataset including Q&A, ratings, image, and bbox"""
        if languages is None:
            languages = ['en']  # Default English

        all_pairs = []
        processed_rows = 0

        for idx, row in self.df.iterrows():
            try:
                # Save image if present
                image_path = None
                if "image" in row and pd.notna(row["image"]):
                    image_path = self._save_image(row["image"], idx)

                # Extract bbox if present
                bbox_coords = None
                if "Bbox coordinates normalized (X, Y, W, H)" in row and pd.notna(row["Bbox coordinates normalized (X, Y, W, H)"]):
                    bbox_coords = row["Bbox coordinates normalized (X, Y, W, H)"]

                for language in languages:
                    qa_items = self.extract_qa_hierarchy(row, language)
                    rating_pairs = []
                    for question_num in range(1, 5):
                        rating_pairs.extend(self.extract_preference_pairs_from_ratings(row, question_num))

                    if qa_items:
                        pairs = self.create_cpo_pairs(qa_items)
                        for pair in pairs:
                            pair['metadata']['language'] = language
                            if image_path:
                                pair['image'] = image_path
                            if bbox_coords:
                                pair['metadata']['bbox'] = bbox_coords
                                pair['prompt'] = pair['prompt'] + f"\n\nFocus on this region: {bbox_coords}"
                        all_pairs.extend(pairs)

                    for pair in rating_pairs:
                        pair['metadata']['language'] = language
                        if image_path:
                            pair['image'] = image_path
                        if bbox_coords:
                            pair['metadata']['bbox'] = bbox_coords
                            pair['prompt'] = pair['prompt'] + f"\n\nFocus on this region: {bbox_coords}"
                    all_pairs.extend(rating_pairs)

                    if qa_items or rating_pairs:
                        processed_rows += 1
                        if sample_size and processed_rows >= sample_size:
                            break
                        
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

        print(f"Final: Processed {processed_rows} rows, generated {len(all_pairs)} preference pairs")
        return all_pairs
    
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
    pairs = processor.process_dataset(sample_size=1000, languages=['en'])  # Process first 1000 rows for testing
    
    if pairs:
        # Save full dataset
        processor.save_processed_data(pairs, 'cpo_training_data.json')
        
        # Create train/test split
        train_pairs, test_pairs = processor.create_train_test_split(pairs)
        
        # Save splits
        processor.save_processed_data(train_pairs, 'cpo_train_data.json')
        processor.save_processed_data(test_pairs, 'cpo_test_data.json')
        
        print(f"Training pairs: {len(train_pairs)}")
        print(f"Test pairs: {len(test_pairs)}")
        
        # Show sample
        print("\nSample pair:")
        print(json.dumps(pairs[0], indent=2))
        
        # Show statistics
        print(f"\nDataset Statistics:")
        print(f"Total pairs: {len(pairs)}")
        print(f"Average pairs per row: {len(pairs) / 1000:.2f}")
        
        # Check rating distribution
        ratings = [pair['metadata']['scores']['composite'] for pair in pairs]
        print("Rating distribution:")
        print(f"  Min: {min(ratings):.3f}")
        print(f"  Max: {max(ratings):.3f}")
        print(f"  Mean: {sum(ratings)/len(ratings):.3f}")
        
    else:
        print("No pairs generated. Check data format and processing logic.")

if __name__ == "__main__":
    main()