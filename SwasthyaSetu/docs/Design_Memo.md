
# Design Memo: Medical Summarizer - Dual Perspective

## 1. Introduction

This document outlines the design and architecture for the Medical Summarizer - Dual Perspective system. The goal is to build an end-to-end, open-source solution capable of generating medical summaries from both patient and clinician perspectives. The system will include a custom Transformer model, a FastAPI backend, a React frontend, and a Docker-based deployment.

## 2. Technology Stack

### 2.1. Model

- **Framework**: PyTorch
- **Architecture**: From-scratch Transformer encoder-decoder (6 layers, d_model=256, 4 heads) with copy mechanism, provenance head, and faithfulness classifier.
- **Export**: ONNX Runtime for inference (server and browser fallback).
- **Training**: Teacher forcing, AdamW optimizer, Cross-Entropy + provenance + faithfulness loss.

### 2.2. Backend

- **Framework**: FastAPI
- **ASGI Server**: Uvicorn
- **ORM**: SQLAlchemy
- **Database Migrations**: Alembic
- **Authentication**: JWT (JSON Web Tokens) with bcrypt for password hashing.
- **Inference**: ONNX Runtime for model inference.

### 2.3. Frontend

- **Framework**: React
- **Styling**: Tailwind CSS
- **Scaffolding**: Vite
- **PDF Export**: jsPDF
- **Features**: Login/register pages, summarization interface with input text, mode toggle (patient/clinician/both), provenance highlights, feedback widget, light/dark mode toggle, responsive UI.

### 2.4. Database

- **Type**: MySQL
- **Containerization**: Docker
- **Tables**: Users, tokens, summaries, feedback.
- **Management**: Alembic for migrations, with initial seed data.

## 3. Data Sources

We will prioritize the use of the following open-source medical summarization datasets:

- **PubMedQA**: For question-answering in biomedical research.
- **MEDIQA**: A challenge focusing on medical question answering and summarization.
- **FaMeSumm**: Focused on faithfulness in medical summarization.
- **PerAnsSumm**: For perspective-aware healthcare answer summarization.

If these datasets are not fully accessible or sufficient for dual-perspective summarization, synthetic Q&A and dual summaries will be generated as a fallback.

## 4. Core Requirements Implementation Plan

### 4.1. Model Implementation

- Implement the Transformer encoder-decoder from scratch in PyTorch.
- Develop the copy mechanism, provenance head, and faithfulness classifier.
- Create a tiny CPU variant (2-3 layers) for demo purposes.
- Implement the training loop with teacher forcing, AdamW, and the specified loss functions.
- Export the trained model to ONNX format with dynamic axes.

### 4.2. Data Preprocessing

- Develop `scripts/prepare_data.py` to preprocess the selected datasets.
- The script will output data in JSONL format, including `context`, `patient_summary`, `clinician_summary`, and `source_sentences`.

### 4.3. Backend Development

- Set up a FastAPI application with Uvicorn.
- Implement JWT authentication and bcrypt password hashing.
- Define REST endpoints: `/auth/register`, `/auth/login`, `/summarize`, `/feedback`, `/history`, `/export`.
- Integrate ONNX Runtime for model inference within the `/summarize` endpoint.
- Implement database interactions using SQLAlchemy.
- Write PyTest unit tests for backend functionalities.

### 4.4. Frontend Development

- Scaffold a React project using Vite with Tailwind CSS.
- Design and implement login/register, summarize, and history pages.
- Develop UI components for text input, mode toggling, provenance highlighting, feedback, and theme switching.
- Integrate jsPDF for client-side PDF export.
- Ensure responsive UI design.

### 4.5. Database Setup

- Configure a Dockerized MySQL database.
- Define database schemas for users, tokens, summaries, and feedback using SQLAlchemy models.
- Implement Alembic migrations for schema management.
- Prepare seed data for initial database population.

## 5. Deployment

- Create Dockerfiles for the backend and frontend applications.
- Develop a `docker-compose.yml` file to orchestrate the full stack (database, backend, frontend).
- Provide a `run_local.sh` script for local setup and execution.
- Write a Render deployment guide for the backend.

## 6. Testing and Documentation

- Implement comprehensive PyTest tests for the backend.
- Develop acceptance tests (`run_acceptance_tests.sh`) to verify end-to-end functionality.
- Maintain `Design_Memo.md` and `Sources_and_Influences.md` throughout the project.
- Create a full repository with code, configs, results, notebooks, and documentation.
- Deliver a pretrained tiny demo checkpoint and ONNX export.
- Prepare slides and a demo script.

## 7. Deliverables

- Full source code repository (MIT license).
- `docker-compose.yml` and `run_local.sh` for local execution.
- Pretrained tiny CPU variant checkpoint and ONNX export.
- Evaluation results (ROUGE, BERTScore, FKGL, terminology density, entity overlap).
- `Design_Memo.md` and `Sources_and_Influences.md` (with 6+ papers/repos).
- Acceptance tests (`run_acceptance_tests.sh`).
- Slides and demo script.



