import torch
from rouge_score import rouge_scorer
from bert_score import score as bert_score_calc
import textstat
import json

# Dummy tokenizer for now, will be replaced by a proper one
class DummyTokenizer:
    def encode(self, text, max_length=None, truncation=False, padding=False, return_tensors=None):
        return text.split() # Simple split by space

    def decode(self, tokens):
        return " ".join(tokens)

    @property
    def vocab_size(self):
        return 256

def calculate_rouge(candidate, reference):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {k: v.fmeasure for k, v in scores.items()}

def calculate_bertscore(candidates, references, device="cpu"):
    P, R, F1 = bert_score_calc(candidates, references, lang="en", device=device)
    return F1.mean().item()

def calculate_fkgl(text):
    return textstat.flesch_kincaid_grade(text)

def calculate_terminology_density(text, medical_terms_list):
    # This is a very basic implementation. A more robust one would use NLP techniques
    # like POS tagging and entity recognition.
    words = text.lower().split()
    term_count = 0
    for term in medical_terms_list:
        if term.lower() in words:
            term_count += 1
    return term_count / len(words) if len(words) > 0 else 0

def calculate_entity_overlap(candidate_entities, reference_entities):
    # Assumes entities are lists of strings
    if not reference_entities:
        return 0.0
    overlap = len(set(candidate_entities).intersection(set(reference_entities)))
    return overlap / len(set(reference_entities))

def evaluate_summaries(predictions_path, references_path, medical_terms_list=None, device="cpu"):
    predictions = []
    with open(predictions_path, "r") as f:
        for line in f:
            predictions.append(json.loads(line))

    references = []
    with open(references_path, "r") as f:
        for line in f:
            references.append(json.loads(line))

    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must be the same.")

    rouge_scores = []
    bert_scores = []
    fkgl_scores = []
    terminology_densities = []
    entity_overlaps = []

    for i in range(len(predictions)):
        pred_patient = predictions[i].get("patient_summary", "")
        pred_clinician = predictions[i].get("clinician_summary", "")
        ref_patient = references[i].get("patient_summary", "")
        ref_clinician = references[i].get("clinician_summary", "")
        ref_context = references[i].get("context", "")

        # ROUGE and BERTScore for patient summary
        if pred_patient and ref_patient:
            rouge_scores.append(calculate_rouge(pred_patient, ref_patient))
            bert_scores.append(calculate_bertscore([pred_patient], [ref_patient], device=device))

        # FKGL for both summaries
        if pred_patient: fkgl_scores.append(calculate_fkgl(pred_patient))
        if pred_clinician: fkgl_scores.append(calculate_fkgl(pred_clinician))

        # Terminology density (example: using context as source for terms)
        if medical_terms_list and pred_clinician:
            terminology_densities.append(calculate_terminology_density(pred_clinician, medical_terms_list))

        # Entity overlap (placeholder - requires actual NER to extract entities)
        # For now, let's simulate some entities from context for demonstration
        # In a real scenario, you'd run an NER model on both candidate and reference
        simulated_ref_entities = [word for word in ref_context.lower().split() if len(word) > 4 and word.isalpha()][:5] # Dummy entity extraction
        simulated_pred_entities = [word for word in pred_clinician.lower().split() if len(word) > 4 and word.isalpha()][:5]
        if simulated_ref_entities:
            entity_overlaps.append(calculate_entity_overlap(simulated_pred_entities, simulated_ref_entities))

    avg_rouge = {k: sum(s[k] for s in rouge_scores) / len(rouge_scores) for k in rouge_scores[0]} if rouge_scores else {}
    avg_bert_score = sum(bert_scores) / len(bert_scores) if bert_scores else 0
    avg_fkgl = sum(fkgl_scores) / len(fkgl_scores) if fkgl_scores else 0
    avg_terminology_density = sum(terminology_densities) / len(terminology_densities) if terminology_densities else 0
    avg_entity_overlap = sum(entity_overlaps) / len(entity_overlaps) if entity_overlaps else 0

    results = {
        "avg_rouge": avg_rouge,
        "avg_bert_score": avg_bert_score,
        "avg_fkgl": avg_fkgl,
        "avg_terminology_density": avg_terminology_density,
        "avg_entity_overlap": avg_entity_overlap,
    }
    return results

if __name__ == "__main__":
    # Create dummy prediction and reference files for testing
    dummy_predictions = [
        {"patient_summary": "I had a bad head pain and light hurt.", "clinician_summary": "Pt c/o severe HA, photophobia."},
        {"patient_summary": "Old man with high BP and sugar, chest hurt.", "clinician_summary": "65M HTN T2DM, admitted for CP."}
    ]
    dummy_references = [
        {"context": "Patient presented with a severe headache and photophobia. MRI showed no abnormalities.", "patient_summary": "I had a really bad headache and bright lights hurt my eyes.", "clinician_summary": "Severe headache, photophobia. MRI negative."},
        {"context": "The patient is a 65-year-old male with a history of hypertension and type 2 diabetes. He was admitted for chest pain.", "patient_summary": "I'm 65, have high blood pressure and diabetes. I came in because of chest pain.", "clinician_summary": "65 y.o. M with HTN, T2DM admitted for CP."}
    ]

    os.makedirs("./data/eval", exist_ok=True)
    with open("./data/eval/dummy_predictions.jsonl", "w") as f:
        for p in dummy_predictions:
            f.write(json.dumps(p) + "\n")
    with open("./data/eval/dummy_references.jsonl", "w") as f:
        for r in dummy_references:
            f.write(json.dumps(r) + "\n")

    # Example medical terms list
    medical_terms = ["headache", "photophobia", "hypertension", "diabetes", "chest pain", "MRI"]

    results = evaluate_summaries(
        predictions_path="./data/eval/dummy_predictions.jsonl",
        references_path="./data/eval/dummy_references.jsonl",
        medical_terms_list=medical_terms,
        device="cpu"
    )
    print(json.dumps(results, indent=4))


