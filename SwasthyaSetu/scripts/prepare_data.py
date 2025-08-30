import json
import os
import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def extract_dual_summaries(item: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract patient and clinician summaries from the dataset.
    This dataset appears to have different summary types (INFORMATION_SUMMARY, SUGGESTION_SUMMARY, etc.)
    We'll create dual perspectives from these.
    """
    labelled_summaries = item.get('labelled_summaries', {})
    question = item.get('question', '')
    context = item.get('context', '')
    answers = item.get('answers', [])
    
    # Combine all available information
    all_info = []
    if context:
        all_info.append(context)
    if answers:
        # Take first few answers to avoid too much noise
        all_info.extend(answers[:3])
    
    combined_context = ' '.join(all_info)
    
    # Create patient-friendly summary
    patient_summary = ""
    if 'INFORMATION_SUMMARY' in labelled_summaries:
        info_summary = labelled_summaries['INFORMATION_SUMMARY']
        # Simplify for patient perspective
        patient_summary = simplify_for_patient(info_summary, question)
    elif 'SUGGESTION_SUMMARY' in labelled_summaries:
        suggestion_summary = labelled_summaries['SUGGESTION_SUMMARY']
        patient_summary = simplify_for_patient(suggestion_summary, question)
    else:
        # Fallback: create a simple patient summary from the question and first answer
        if answers:
            patient_summary = f"Regarding your question about {question.lower()}: {answers[0][:200]}..."
    
    # Create clinician summary
    clinician_summary = ""
    if 'INFORMATION_SUMMARY' in labelled_summaries:
        clinician_summary = labelled_summaries['INFORMATION_SUMMARY']
    elif 'CAUSE_SUMMARY' in labelled_summaries and 'SUGGESTION_SUMMARY' in labelled_summaries:
        cause = labelled_summaries.get('CAUSE_SUMMARY', '')
        suggestion = labelled_summaries.get('SUGGESTION_SUMMARY', '')
        clinician_summary = f"Etiology: {cause} Management: {suggestion}"
    else:
        # Fallback: use the first comprehensive answer
        if answers:
            clinician_summary = answers[0][:300] + "..." if len(answers[0]) > 300 else answers[0]
    
    return {
        'context': clean_text(combined_context),
        'patient_summary': clean_text(patient_summary),
        'clinician_summary': clean_text(clinician_summary)
    }

def simplify_for_patient(text: str, question: str) -> str:
    """
    Simplify medical text for patient understanding
    """
    # Basic simplification rules
    text = text.replace('etiology', 'cause')
    text = text.replace('pathophysiology', 'how it works in the body')
    text = text.replace('therapeutic', 'treatment')
    text = text.replace('prophylaxis', 'prevention')
    text = text.replace('contraindicated', 'not recommended')
    text = text.replace('adverse effects', 'side effects')
    
    # Make it more conversational
    if question:
        text = f"About your question on {question.lower()}: {text}"
    
    return text

def extract_source_sentences(item: Dict[str, Any]) -> List[str]:
    """
    Extract source sentences for provenance tracking
    """
    sentences = []
    
    # Extract from answers
    answers = item.get('answers', [])
    for answer in answers[:2]:  # Take first 2 answers
        # Split into sentences (simple approach)
        answer_sentences = re.split(r'[.!?]+', answer)
        sentences.extend([s.strip() for s in answer_sentences if len(s.strip()) > 10])
    
    # Extract from context if available
    context = item.get('context', '')
    if context:
        context_sentences = re.split(r'[.!?]+', context)
        sentences.extend([s.strip() for s in context_sentences if len(s.strip()) > 10])
    
    return sentences[:5]  # Limit to 5 sentences

def process_dataset(input_file: str, output_file: str):
    """
    Process the medical Q&A dataset into our dual-perspective format
    """
    print(f"Processing {input_file}...")
    
    processed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        try:
            # Extract dual summaries
            summaries = extract_dual_summaries(item)
            
            # Skip if we couldn't generate meaningful summaries
            if not summaries['patient_summary'] or not summaries['clinician_summary']:
                continue
                
            # Extract source sentences for provenance
            source_sentences = extract_source_sentences(item)
            
            processed_item = {
                'uri': item.get('uri', ''),
                'question': item.get('question', ''),
                'context': summaries['context'],
                'patient_summary': summaries['patient_summary'],
                'clinician_summary': summaries['clinician_summary'],
                'source_sentences': source_sentences
            }
            
            processed_data.append(processed_item)
            
        except Exception as e:
            print(f"Error processing item {item.get('uri', 'unknown')}: {e}")
            continue
    
    # Save processed data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Processed {len(processed_data)} items and saved to {output_file}")

def main():
    """
    Main function to process all dataset splits
    """
    base_dir = "./data"
    raw_dir = os.path.join(base_dir, "raw")
    processed_dir = os.path.join(base_dir, "processed")
    
    # Process each split
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        input_file = os.path.join(raw_dir, f"{split}.json")
        output_file = os.path.join(processed_dir, f"{split}_processed.jsonl")
        
        if os.path.exists(input_file):
            process_dataset(input_file, output_file)
        else:
            print(f"Warning: {input_file} not found, skipping...")
    
    # Create a combined training file for our model
    train_file = os.path.join(processed_dir, "train_processed.jsonl")
    valid_file = os.path.join(processed_dir, "valid_processed.jsonl")
    combined_file = os.path.join(processed_dir, "processed_data.jsonl")
    
    if os.path.exists(train_file):
        # Combine train and validation for more training data
        with open(combined_file, 'w', encoding='utf-8') as outf:
            if os.path.exists(train_file):
                with open(train_file, 'r', encoding='utf-8') as inf:
                    outf.write(inf.read())
            if os.path.exists(valid_file):
                with open(valid_file, 'r', encoding='utf-8') as inf:
                    outf.write(inf.read())
        
        print(f"Combined dataset saved to {combined_file}")

if __name__ == "__main__":
    main()

