import torch
from torch.utils.data import Dataset
import json
from utils.tokenizer import MedicalTokenizer

class MedicalSummarizationDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len_src, max_len_tgt):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_len_src = max_len_src
        self.max_len_tgt = max_len_tgt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        patient_summary = item['patient_summary']
        clinician_summary = item['clinician_summary']
        source_sentences = item['source_sentences']

        # Tokenize context
        src_tokens = self.tokenizer.encode(context, max_length=self.max_len_src, truncation=True, padding='max_length', return_tensors='pt')

        # Tokenize patient summary (for patient perspective)
        patient_tgt_tokens = self.tokenizer.encode(patient_summary, max_length=self.max_len_tgt, truncation=True, padding='max_length', return_tensors='pt')

        # Tokenize clinician summary (for clinician perspective)
        clinician_tgt_tokens = self.tokenizer.encode(clinician_summary, max_length=self.max_len_tgt, truncation=True, padding='max_length', return_tensors='pt')

        # Placeholder for provenance and faithfulness labels
        # These would typically be derived from the data or generated during preprocessing
        # For now, dummy tensors
        provenance_labels = torch.zeros(self.max_len_tgt - 1, dtype=torch.float)
        faithfulness_label = torch.tensor(0.0, dtype=torch.float)

        return {
            'src': src_tokens.squeeze(0),
            'patient_tgt': patient_tgt_tokens.squeeze(0),
            'clinician_tgt': clinician_tgt_tokens.squeeze(0),
            'provenance_labels': provenance_labels,
            'faithfulness_label': faithfulness_label
        }


if __name__ == '__main__':
    # Example usage
    tokenizer = MedicalTokenizer.load("../model/medical_tokenizer.json")
    data_path = "../data/processed/processed_data.jsonl"
    max_len_src = 128
    max_len_tgt = 64

    dataset = MedicalSummarizationDataset(data_path, tokenizer, max_len_src, max_len_tgt)
    print(f"Dataset size: {len(dataset)}")
    item = dataset[0]
    print(f"Source tokens shape: {item['src'].shape}")
    print(f"Patient target tokens shape: {item['patient_tgt'].shape}")
    print(f"Clinician target tokens shape: {item['clinician_tgt'].shape}")
    print(f"Provenance labels shape: {item['provenance_labels'].shape}")
    print(f"Faithfulness label shape: {item['faithfulness_label'].shape}")

    # Test decoding
    decoded_src = tokenizer.decode(item["src"].tolist())
    print(f"Decoded source: {decoded_src[:100]}...")

