import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import numpy as np
from sklearn.model_selection import train_test_split
import os

class SwasthyaSetuTrainer:
    """SwasthyaSetu Medical Summarization Trainer using T5"""
    
    def __init__(self, model_name="t5-small", max_input_length=512, max_output_length=128):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_tokenizer_and_model(self):
        """Load the T5 tokenizer and model"""
        print(f"🚀 Loading {self.model_name} for SwasthyaSetu...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        print(f"✅ Model loaded successfully on {self.device}")
        return self.tokenizer, self.model
    
    def load_medical_dataset(self, dataset_size=100):
        """Load medical dialogue dataset for SwasthyaSetu"""
        print(f"📊 Loading medical dataset for SwasthyaSetu (size: {dataset_size})...")
        
        try:
            # Load the medical dialogue dataset
            dataset = load_dataset("omi-health/medical-dialogue-to-soap-summary", split="train")
            
            # Select a subset for faster training
            if dataset_size < len(dataset):
                dataset = dataset.select(range(dataset_size))
                
            print(f"✅ Dataset loaded: {len(dataset)} examples")
            return dataset
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print("🔄 Creating synthetic medical data for SwasthyaSetu...")
            return self.create_synthetic_medical_data(dataset_size)
    
    def create_synthetic_medical_data(self, size=100):
        """Create synthetic medical data for SwasthyaSetu training"""
        print("🔧 Creating synthetic medical data...")
        
        synthetic_data = {
            "dialogue": [],
            "summary": []
        }
        
        # Medical scenarios for SwasthyaSetu
        medical_scenarios = [
            {
                "dialogue": "Patient: I have been experiencing chest pain for the past week. Doctor: Can you describe the pain? Patient: It's a sharp pain that comes and goes. Doctor: Any shortness of breath? Patient: Yes, sometimes when I walk upstairs.",
                "summary": "Patient presents with chest pain for one week, described as sharp and intermittent, accompanied by shortness of breath on exertion."
            },
            {
                "dialogue": "Patient: My blood pressure has been high lately. Doctor: What readings are you getting? Patient: Around 150/95. Doctor: Are you taking your medication? Patient: I sometimes forget. Doctor: Stress levels? Patient: Very high at work.",
                "summary": "Patient reports elevated blood pressure (150/95) with medication non-compliance and high stress levels at work."
            },
            {
                "dialogue": "Patient: I have diabetes and my sugar levels are not controlled. Doctor: What are your current readings? Patient: Fasting around 180, post-meal 250. Doctor: Are you following the diet plan? Patient: I try but it's difficult. Doctor: Exercise routine? Patient: Walking 30 minutes daily.",
                "summary": "Diabetic patient with uncontrolled blood glucose (fasting 180, post-meal 250), struggling with diet compliance but maintaining daily exercise."
            }
        ]
        
        # Generate variations
        for i in range(size):
            base_scenario = medical_scenarios[i % len(medical_scenarios)]
            synthetic_data["dialogue"].append(base_scenario["dialogue"])
            synthetic_data["summary"].append(base_scenario["summary"])
        
        print(f"✅ Created {len(synthetic_data['dialogue'])} synthetic medical examples")
        return synthetic_data
    
    def preprocess_function(self, examples):
        """Preprocess medical dialogue data for SwasthyaSetu"""
        # Add medical summarization prefix
        inputs = [f"summarize medical dialogue: {dialogue}" for dialogue in examples["dialogue"]]
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs, 
            max_length=self.max_input_length, 
            truncation=True, 
            padding="max_length"
        )
        
        # Tokenize labels (summaries)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["summary"], 
                max_length=self.max_output_length, 
                truncation=True, 
                padding="max_length"
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def prepare_dataset(self, dataset):
        """Prepare and tokenize the dataset for SwasthyaSetu"""
        print("🔧 Preparing dataset for SwasthyaSetu...")
        
        # Apply preprocessing
        tokenized_dataset = dataset.map(
            self.preprocess_function, 
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Split into train and validation
        train_size = int(0.8 * len(tokenized_dataset))
        val_size = len(tokenized_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            tokenized_dataset, [train_size, val_size]
        )
        
        print(f"✅ Dataset prepared: {len(train_dataset)} train, {len(val_dataset)} validation")
        return train_dataset, val_dataset
    
    def setup_training(self, output_dir="./swasthyasetu_model"):
        """Setup training configuration for SwasthyaSetu"""
        print("⚙️ Setting up training configuration...")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=3,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            warmup_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        return training_args
    
    def train_model(self, train_dataset, val_dataset, training_args):
        """Train the SwasthyaSetu medical summarization model"""
        print("🏋️ Starting training for SwasthyaSetu...")
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        print(f"✅ Training complete! Model saved to {training_args.output_dir}")
        return trainer
    
    def generate_summary(self, text, max_length=128):
        """Generate medical summary using trained SwasthyaSetu model"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_tokenizer_and_model() first.")
        
        # Prepare input
        inputs = self.tokenizer(
            f"summarize medical dialogue: {text}",
            max_length=self.max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        # Decode output
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    def evaluate_model(self, test_texts, test_summaries):
        """Evaluate SwasthyaSetu model performance"""
        print("📊 Evaluating SwasthyaSetu model...")
        
        generated_summaries = []
        for text in test_texts:
            summary = self.generate_summary(text)
            generated_summaries.append(summary)
        
        # Calculate basic metrics
        exact_matches = sum(1 for gen, ref in zip(generated_summaries, test_summaries) if gen == ref)
        accuracy = exact_matches / len(test_texts) if test_texts else 0
        
        print(f"✅ Evaluation complete:")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"   Exact matches: {exact_matches}/{len(test_texts)}")
        
        return {
            "accuracy": accuracy,
            "exact_matches": exact_matches,
            "total_samples": len(test_texts),
            "generated_summaries": generated_summaries
        }

def main():
    """Main training function for SwasthyaSetu"""
    print("🏥 SwasthyaSetu Medical Summarization Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = SwasthyaSetuTrainer(model_name="t5-small")
    
    # Load model and tokenizer
    tokenizer, model = trainer.load_tokenizer_and_model()
    
    # Load dataset
    dataset = trainer.load_medical_dataset(dataset_size=50)  # Small dataset for demo
    
    # Prepare dataset
    train_dataset, val_dataset = trainer.prepare_dataset(dataset)
    
    # Setup training
    training_args = trainer.setup_training()
    
    # Train model
    trainer_instance = trainer.train_model(train_dataset, val_dataset, training_args)
    
    # Test the model
    print("\n🧪 Testing SwasthyaSetu model...")
    test_text = "Patient: I have been experiencing chest pain for the past week. Doctor: Can you describe the pain? Patient: It's a sharp pain that comes and goes."
    
    summary = trainer.generate_summary(test_text)
    print(f"Input: {test_text}")
    print(f"Generated Summary: {summary}")
    
    print("\n✨ SwasthyaSetu training complete!")

if __name__ == "__main__":
    main()

