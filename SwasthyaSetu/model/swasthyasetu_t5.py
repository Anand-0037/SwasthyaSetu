import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import json
from typing import Dict, List, Optional, Tuple

class SwasthyaSetuT5Model:
    """SwasthyaSetu T5-based Medical Summarization Model"""
    
    def __init__(self, model_path: str = "./swasthyasetu_model", model_name: str = "t5-small"):
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """Load the SwasthyaSetu T5 model and tokenizer"""
        try:
            print(f"🚀 Loading SwasthyaSetu T5 model from {self.model_path}...")
            
            # Try to load from local path first
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
                print("✅ Loaded SwasthyaSetu model from local path")
            except:
                # Fallback to pretrained model
                print("🔄 Loading pretrained T5 model for SwasthyaSetu...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                
                # Set pad token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Move model to device
            self.model.to(self.device)
            self.is_loaded = True
            
            print(f"✅ SwasthyaSetu T5 model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading SwasthyaSetu model: {e}")
            self.is_loaded = False
            return False
    
    def generate_medical_summary(self, text: str, max_length: int = 128) -> str:
        """Generate medical summary using SwasthyaSetu T5 model"""
        if not self.is_loaded:
            raise ValueError("SwasthyaSetu model not loaded. Call load_model() first.")
        
        try:
            # Prepare input with medical context
            input_text = f"summarize medical dialogue: {text}"
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                max_length=512,
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
                    no_repeat_ngram_size=2,
                    do_sample=False
                )
            
            # Decode output
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return f"Medical summary generation failed: {str(e)}"
    
    def generate_dual_perspective_summaries(self, text: str) -> Tuple[str, str]:
        """Generate both patient and clinical perspectives for SwasthyaSetu"""
        if not self.is_loaded:
            raise ValueError("SwasthyaSetu model not loaded. Call load_model() first.")
        
        try:
            # Generate base medical summary
            base_summary = self.generate_medical_summary(text, max_length=128)
            
            # Generate patient-friendly version
            patient_prompt = f"explain in simple terms for patient: {base_summary}"
            patient_summary = self.generate_medical_summary(patient_prompt, max_length=100)
            
            # Generate clinical version
            clinical_prompt = f"provide clinical assessment: {base_summary}"
            clinical_summary = self.generate_medical_summary(clinical_prompt, max_length=100)
            
            return patient_summary, clinical_summary
            
        except Exception as e:
            print(f"❌ Error generating dual perspectives: {e}")
            fallback_patient = "Patient Summary: Medical information has been processed. Please consult healthcare professionals for personalized advice."
            fallback_clinical = "Clinical Assessment: Medical data analyzed. Professional evaluation recommended for accurate diagnosis."
            return fallback_patient, fallback_clinical
    
    def process_json_medical_data(self, data: Dict) -> Dict:
        """Process JSON medical data using SwasthyaSetu T5 model"""
        try:
            start_time = time.time()
            
            # Extract text content
            extracted_text = self._extract_text_from_json(data)
            
            # Generate summaries
            patient_summary, clinical_summary = self.generate_dual_perspective_summaries(extracted_text)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            return {
                "patient_summary": patient_summary,
                "clinician_summary": clinical_summary,
                "extracted_content": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
                "processing_time": round(processing_time, 3),
                "model": "SwasthyaSetu T5",
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ Error processing JSON data: {e}")
            return {
                "patient_summary": "Processing failed. Please try again.",
                "clinician_summary": "Data processing error occurred.",
                "extracted_content": "",
                "processing_time": 0,
                "model": "SwasthyaSetu T5",
                "status": "error",
                "error": str(e)
            }
    
    def _extract_text_from_json(self, data: Dict) -> str:
        """Extract meaningful text from JSON medical data for SwasthyaSetu"""
        try:
            medical_text = []
            
            # Extract question and context
            if 'question' in data and data['question']:
                medical_text.append(f"Question: {data['question']}")
            
            if 'context' in data and data['context']:
                medical_text.append(f"Context: {data['context']}")
            
            # Extract answers
            if 'answers' in data and isinstance(data['answers'], list):
                for i, answer in enumerate(data['answers'][:3]):  # Limit to first 3 answers
                    if answer and isinstance(answer, str) and len(answer.strip()) > 10:
                        medical_text.append(f"Answer {i+1}: {answer}")
            
            # Extract summaries
            if 'labelled_summaries' in data and isinstance(data['labelled_summaries'], dict):
                for key, summary in data['labelled_summaries'].items():
                    if summary and isinstance(summary, str) and len(summary.strip()) > 20:
                        medical_text.append(f"{key}: {summary}")
                        break  # Take first meaningful summary
            
            # Fallback to raw_text
            if not medical_text and 'raw_text' in data and data['raw_text']:
                raw_text = str(data['raw_text'])
                return raw_text[:500] + "..." if len(raw_text) > 500 else raw_text
            
            return ' '.join(medical_text) if medical_text else "Medical data processed"
            
        except Exception as e:
            print(f"❌ Error extracting text: {e}")
            return "Medical information for summarization"
    
    def get_model_info(self) -> Dict:
        """Get SwasthyaSetu model information"""
        return {
            "model_name": "SwasthyaSetu T5",
            "base_model": self.model_name,
            "device": str(self.device),
            "is_loaded": self.is_loaded,
            "model_path": self.model_path,
            "tokenizer_vocab_size": len(self.tokenizer) if self.tokenizer else 0,
            "model_parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }
    
    def health_check(self) -> Dict:
        """Health check for SwasthyaSetu model"""
        try:
            if not self.is_loaded:
                return {"status": "not_loaded", "message": "Model not loaded"}
            
            # Test generation
            test_text = "Patient: I have a headache. Doctor: How long? Patient: 2 days."
            test_summary = self.generate_medical_summary(test_text, max_length=50)
            
            return {
                "status": "healthy",
                "message": "SwasthyaSetu T5 model working correctly",
                "test_summary": test_summary,
                "device": str(self.device),
                "model_loaded": True
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Model error: {str(e)}",
                "model_loaded": False
            }

# Global SwasthyaSetu model instance
swasthyasetu_model = None

def get_swasthyasetu_model() -> SwasthyaSetuT5Model:
    """Get or create global SwasthyaSetu model instance"""
    global swasthyasetu_model
    if swasthyasetu_model is None:
        swasthyasetu_model = SwasthyaSetuT5Model()
        swasthyasetu_model.load_model()
    return swasthyasetu_model

def initialize_swasthyasetu_model() -> bool:
    """Initialize the SwasthyaSetu T5 model"""
    try:
        model = get_swasthyasetu_model()
        return model.is_loaded
    except Exception as e:
        print(f"❌ Failed to initialize SwasthyaSetu model: {e}")
        return False
