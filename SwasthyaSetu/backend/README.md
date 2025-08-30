# SwasthyaSetu Backend

AI-Powered Medical Summarization Backend with dual-perspective analysis.

## Features

- Ultra-fast medical text processing
- SwasthyaSetu T5 AI model integration
- Dual-perspective summaries (Patient + Clinical)
- JSON/JSONL medical data processing
- Quality assessment and provenance tracking

## API Endpoints

- `/summarize/ultra-fast` - Instant processing
- `/summarize/swasthyasetu-t5` - AI-powered summaries
- `/summarize/large-json` - Large dataset processing
- `/model/info` - System status and model information

## Installation

```bash
uv sync
```

## Usage

```bash
uv run python -m uvicorn main:app --host 127.0.0.1 --port 8000
```
