import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import numpy as np
from model.perspective_aware_transformer import TinyPerspectiveAwareMedicalSummarizer
from utils.dataset import MedicalSummarizationDataset
from utils.tokenizer import MedicalTokenizer

class PerspectiveAwareDataset(MedicalSummarizationDataset):
    """
    Enhanced dataset that provides perspective labels for training
    """
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        
        # Add perspective labels
        # We'll train the model to generate both patient and clinician summaries
        # So we'll return both perspectives for each sample
        return {
            **item,
            'patient_perspective': 1,
            'clinician_perspective': 2
        }

def train_perspective_aware_model(model, dataloader, optimizer, tokenizer, device, num_epochs=5):
    """
    Training loop for perspective-aware model
    """
    model.train()
    
    # Loss functions
    criterion_ce = nn.CrossEntropyLoss(ignore_index=tokenizer.word_to_id[tokenizer.PAD_TOKEN])
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            src = batch['src'].to(device)
            patient_tgt = batch['patient_tgt'].to(device)
            clinician_tgt = batch['clinician_tgt'].to(device)
            provenance_labels = batch['provenance_labels'].to(device)
            faithfulness_label = batch['faithfulness_label'].to(device)
            
            optimizer.zero_grad()
            
            # Train on patient perspective
            patient_outputs = model(src, patient_tgt[:, :-1], perspective=1)
            patient_vocab_dist, patient_copy_gate, patient_prov, patient_faith, _, patient_persp_probs = patient_outputs
            
            # Patient losses
            loss_patient_ce = criterion_ce(
                patient_vocab_dist.reshape(-1, patient_vocab_dist.size(-1)), 
                patient_tgt[:, 1:].reshape(-1)
            )
            loss_patient_prov = criterion_bce(patient_prov.squeeze(-1), provenance_labels)
            loss_patient_faith = criterion_bce(patient_faith.squeeze(-1), faithfulness_label)
            
            # Perspective classification loss (should predict patient=1)
            patient_perspective_target = torch.full((src.size(0),), 1, dtype=torch.long, device=device)
            loss_patient_perspective = nn.CrossEntropyLoss()(patient_persp_probs, patient_perspective_target)
            
            # Train on clinician perspective
            clinician_outputs = model(src, clinician_tgt[:, :-1], perspective=2)
            clinician_vocab_dist, clinician_copy_gate, clinician_prov, clinician_faith, _, clinician_persp_probs = clinician_outputs
            
            # Clinician losses
            loss_clinician_ce = criterion_ce(
                clinician_vocab_dist.reshape(-1, clinician_vocab_dist.size(-1)), 
                clinician_tgt[:, 1:].reshape(-1)
            )
            loss_clinician_prov = criterion_bce(clinician_prov.squeeze(-1), provenance_labels)
            loss_clinician_faith = criterion_bce(clinician_faith.squeeze(-1), faithfulness_label)
            
            # Perspective classification loss (should predict clinician=2)
            clinician_perspective_target = torch.full((src.size(0),), 2, dtype=torch.long, device=device)
            loss_clinician_perspective = nn.CrossEntropyLoss()(clinician_persp_probs, clinician_perspective_target)
            
            # Combined loss
            total_batch_loss = (
                loss_patient_ce + loss_patient_prov + loss_patient_faith + loss_patient_perspective +
                loss_clinician_ce + loss_clinician_prov + loss_clinician_faith + loss_clinician_perspective
            ) / 8  # Average across all losses
            
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            num_batches += 1
            
            if num_batches % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {num_batches}, Loss: {total_batch_loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")
    
    return model

def evaluate_model(model, tokenizer, device, test_samples=5):
    """
    Evaluate the model with some test examples
    """
    model.eval()
    
    # Test data
    test_contexts = [
        "Patient asks about diabetes management and blood sugar control.",
        "What are the side effects of chemotherapy for cancer treatment?",
        "How to manage high blood pressure with lifestyle changes?",
        "Symptoms of heart attack and when to seek emergency care.",
        "Understanding depression and available treatment options."
    ]
    
    print("\n" + "="*80)
    print("MODEL EVALUATION - PERSPECTIVE-AWARE SUMMARIES")
    print("="*80)
    
    with torch.no_grad():
        for i, context in enumerate(test_contexts[:test_samples]):
            print(f"\nTest Case {i+1}:")
            print(f"Context: {context}")
            print("-" * 60)
            
            # Tokenize input
            src_tokens = tokenizer.encode(context, max_length=128, padding='max_length', return_tensors='pt').to(device)
            
            # Generate patient summary
            patient_summary_tokens = model.generate_summary(src_tokens, max_length=64, perspective=1, temperature=0.8)
            patient_summary = tokenizer.decode(patient_summary_tokens[0].tolist())
            
            # Generate clinician summary
            clinician_summary_tokens = model.generate_summary(src_tokens, max_length=64, perspective=2, temperature=0.8)
            clinician_summary = tokenizer.decode(clinician_summary_tokens[0].tolist())
            
            print(f"Patient Summary: {patient_summary}")
            print(f"Clinician Summary: {clinician_summary}")
            print("-" * 60)

def main():
    # Configuration
    MAX_LEN_SRC = 128
    MAX_LEN_TGT = 64
    D_MODEL = 256
    NUM_HEADS = 4
    NUM_ENCODER_LAYERS = 2
    NUM_DECODER_LAYERS = 2
    D_FF = 512
    DROPOUT = 0.1
    BATCH_SIZE = 2  # Small batch size for CPU training
    NUM_EPOCHS = 3  # Reduced for demo
    LEARNING_RATE = 0.0005
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    try:
        tokenizer = MedicalTokenizer.load("./model/medical_tokenizer.json")
        print(f"Loaded tokenizer with vocabulary size: {tokenizer.vocab_size_actual}")
    except FileNotFoundError:
        print("Tokenizer not found. Please run the tokenizer building script first.")
        return
    
    # Dataset and DataLoader
    data_path = "./data/processed/processed_data.jsonl"
    try:
        dataset = PerspectiveAwareDataset(data_path, tokenizer, MAX_LEN_SRC, MAX_LEN_TGT)
        # Use a subset for faster training
        subset_size = min(1000, len(dataset))
        dataset.data = dataset.data[:subset_size]
        
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"Loaded dataset with {len(dataset)} samples")
    except FileNotFoundError:
        print("Dataset not found. Please run the data preparation script first.")
        return
    
    # Model
    vocab_size = tokenizer.vocab_size_actual
    model = TinyPerspectiveAwareMedicalSummarizer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT
    ).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Optimizer with learning rate scheduling
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    
    print("Starting perspective-aware training...")
    
    # Training
    model = train_perspective_aware_model(model, dataloader, optimizer, tokenizer, device, NUM_EPOCHS)
    
    print("Training completed!")
    
    # Save model
    model_save_path = "./model/perspective_aware_medical_summarizer.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'model_config': {
            'd_model': D_MODEL,
            'num_heads': NUM_HEADS,
            'num_encoder_layers': NUM_ENCODER_LAYERS,
            'num_decoder_layers': NUM_DECODER_LAYERS,
            'd_ff': D_FF,
            'dropout': DROPOUT
        }
    }, model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Evaluate model
    evaluate_model(model, tokenizer, device)
    
    # Export to ONNX for deployment
    try:
        print("Exporting to ONNX...")
        dummy_src = torch.randint(0, vocab_size, (1, MAX_LEN_SRC), device=device)
        dummy_tgt = torch.randint(0, vocab_size, (1, MAX_LEN_TGT - 1), device=device)
        
        torch.onnx.export(
            model,
            (dummy_src, dummy_tgt, 1),  # Include perspective as input
            "./model/perspective_aware_medical_summarizer.onnx",
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["src", "tgt", "perspective"],
            output_names=["vocab_dist", "copy_gate", "provenance_scores", "faithfulness_score", "attention_weights", "perspective_probs"],
            dynamic_axes={
                "src": {0: "batch_size", 1: "src_seq_len"},
                "tgt": {0: "batch_size", 1: "tgt_seq_len"},
                "vocab_dist": {0: "batch_size", 1: "tgt_seq_len"},
                "copy_gate": {0: "batch_size", 1: "tgt_seq_len"},
                "provenance_scores": {0: "batch_size", 1: "tgt_seq_len"},
                "faithfulness_score": {0: "batch_size"},
                "attention_weights": {0: "batch_size", 1: "tgt_seq_len", 2: "src_seq_len"},
                "perspective_probs": {0: "batch_size"}
            }
        )
        print("ONNX export completed!")
    except Exception as e:
        print(f"ONNX export failed: {e}")

if __name__ == "__main__":
    main()

