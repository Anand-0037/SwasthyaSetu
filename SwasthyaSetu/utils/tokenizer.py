import json
import re
from collections import Counter
from typing import List, Dict, Tuple
import torch

class MedicalTokenizer:
    """
    A simple tokenizer for medical text that builds vocabulary from the dataset
    """
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.word_counts = Counter()
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        self.PATIENT_TOKEN = '<PATIENT>'
        self.CLINICIAN_TOKEN = '<CLINICIAN>'
        
        self.special_tokens = [
            self.PAD_TOKEN, self.UNK_TOKEN, self.START_TOKEN, 
            self.END_TOKEN, self.PATIENT_TOKEN, self.CLINICIAN_TOKEN
        ]
        
        # Initialize special token IDs
        for i, token in enumerate(self.special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization by splitting on whitespace and punctuation"""
        if not text:
            return []
        
        # Convert to lowercase and split on whitespace and punctuation
        text = text.lower()
        # Keep some medical punctuation together
        text = re.sub(r'([.!?,:;])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        
        tokens = text.strip().split()
        return [token for token in tokens if token]
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from a list of texts"""
        print("Building vocabulary...")
        
        # Count all words
        for text in texts:
            tokens = self.tokenize(text)
            self.word_counts.update(tokens)
        
        # Get most common words (excluding special tokens)
        most_common = self.word_counts.most_common(self.vocab_size - len(self.special_tokens))
        
        # Add to vocabulary
        for word, count in most_common:
            if word not in self.word_to_id:
                word_id = len(self.word_to_id)
                self.word_to_id[word] = word_id
                self.id_to_word[word_id] = word
        
        print(f"Built vocabulary with {len(self.word_to_id)} tokens")
        print(f"Most common words: {[word for word, _ in most_common[:10]]}")
    
    def encode(self, text: str, max_length: int = None, truncation: bool = True, 
               padding: str = 'max_length', return_tensors: str = None) -> torch.Tensor:
        """Encode text to token IDs"""
        tokens = self.tokenize(text)
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            token_ids.append(self.word_to_id.get(token, self.word_to_id[self.UNK_TOKEN]))
        
        # Handle max_length
        if max_length:
            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            
            if padding == 'max_length':
                while len(token_ids) < max_length:
                    token_ids.append(self.word_to_id[self.PAD_TOKEN])
        
        if return_tensors == 'pt':
            return torch.tensor(token_ids).unsqueeze(0)
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                token = self.id_to_word[token_id]
                if token not in [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN]:
                    tokens.append(token)
        
        return ' '.join(tokens)
    
    @property
    def vocab_size_actual(self):
        return len(self.word_to_id)
    
    def save(self, path: str):
        """Save tokenizer to file"""
        tokenizer_data = {
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
        
        print(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load tokenizer from file"""
        with open(path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        tokenizer = cls(tokenizer_data['vocab_size'])
        tokenizer.word_to_id = tokenizer_data['word_to_id']
        tokenizer.id_to_word = {int(k): v for k, v in tokenizer_data['id_to_word'].items()}
        tokenizer.special_tokens = tokenizer_data['special_tokens']
        
        return tokenizer

def build_tokenizer_from_dataset(data_path: str, vocab_size: int = 8000) -> MedicalTokenizer:
    """Build tokenizer from the processed dataset"""
    print(f"Building tokenizer from {data_path}")
    
    # Collect all texts
    all_texts = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            
            # Add all text fields
            all_texts.append(item.get('context', ''))
            all_texts.append(item.get('patient_summary', ''))
            all_texts.append(item.get('clinician_summary', ''))
            all_texts.append(item.get('question', ''))
            
            # Add source sentences
            source_sentences = item.get('source_sentences', [])
            all_texts.extend(source_sentences)
    
    # Filter out empty texts
    all_texts = [text for text in all_texts if text.strip()]
    
    print(f"Collected {len(all_texts)} texts for vocabulary building")
    
    # Build tokenizer
    tokenizer = MedicalTokenizer(vocab_size=vocab_size)
    tokenizer.build_vocab(all_texts)
    
    return tokenizer

if __name__ == "__main__":
    # Build tokenizer from the processed dataset
    data_path = "../data/processed/processed_data.jsonl"
    tokenizer = build_tokenizer_from_dataset(data_path, vocab_size=8000)
    
    # Save tokenizer
    tokenizer.save("../model/medical_tokenizer.json")
    
    # Test tokenizer
    test_text = "What is the treatment for diabetes?"
    encoded = tokenizer.encode(test_text, max_length=20, padding='max_length', return_tensors='pt')
    decoded = tokenizer.decode(encoded.squeeze().tolist())
    
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

