# SwasthyaSetu
swasthya setu is Medical discussion summarizer (between patient and doctor) that gives patient and clinician perspective with using preexisting llm's apis. we will parse the text and generate summary from scratch

**Patient Perspective**: Simplified, accessible language for patients and caregivers
**Clinician Perspective**: Technical, detailed summaries for healthcare professionals

## AI Model 
- transformer architecture ( 10 to 30 million) parameters
- Dual mode generation --> clinician and patient (caregivers)
<!-- - Info, sugestion, experience, question, cause. -->
- copy and summary report export feature.
- 

## Research based 
- **Plasma architecture: given latest research paper.
- Runs NLP preprocessing (tokenization, entity recognition, perspective tagging).

## Tech stack
- fastapi backend - rest api
- react frontend - dark light toggle, 
- Mysql database
- Docker deployment so that users can easily fork the repo and modify this.
- ONNX export ( optimised model inference for production)

## 