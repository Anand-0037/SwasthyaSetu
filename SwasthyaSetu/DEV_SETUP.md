# Development Setup Guide

This guide explains how to run the Medical Summarizer application in development mode without Docker.

## Prerequisites

- Python 3.11+
- Node.js 18+ (with npm or pnpm)
- Git

## Quick Start

The easiest way to run the application is using the provided script:

```bash
./run_dev.sh
```

This script will:
1. Check dependencies
2. Start the backend server on http://127.0.0.1:8000
3. Start the frontend server on http://localhost:5173
4. Open the API documentation at http://127.0.0.1:8000/docs

## Manual Setup

If you prefer to run services manually or need to troubleshoot:

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the backend server:
   ```bash
   python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

The backend will be available at http://127.0.0.1:8000

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   pnpm install
   ```

3. Start the development server:
   ```bash
   pnpm run dev
   ```

The frontend will be available at http://localhost:5173

## Database

The application uses SQLite for development, which is automatically configured. The database file `backend/medical_summarizer.db` will be created automatically when you first run the application.

## API Endpoints

- **POST** `/auth/register` - User registration
- **POST** `/auth/login` - User authentication
- **POST** `/summarize` - Generate medical summaries
- **POST** `/feedback` - Submit feedback on summaries
- **GET** `/history` - Get user's summary history
- **GET** `/export/{summary_id}` - Export a specific summary
- **GET** `/model/info` - Get model information

## Troubleshooting

### Backend Issues

- **Import errors**: Make sure you're in the virtual environment
- **Database errors**: The SQLite database should be created automatically
- **Port conflicts**: Change the port in the uvicorn command if 8000 is busy

### Frontend Issues

- **Dependencies not found**: Run `pnpm install` in the frontend directory
- **Port conflicts**: Vite will automatically find an available port
- **Build errors**: Check the console for specific error messages

### Common Issues

- **CORS errors**: The backend is configured to allow requests from both localhost:3000 and localhost:5173
- **Authentication errors**: Make sure you're using the correct JWT token format
- **Model loading errors**: The application will fall back to demo mode if the ML model can't be loaded

## Development Features

- **Hot reload**: Both backend and frontend support automatic reloading
- **API documentation**: Available at http://127.0.0.1:8000/docs
- **Database inspection**: You can use SQLite tools to inspect the database
- **Logging**: Backend logs are displayed in the terminal

## Stopping the Application

- **Using the script**: Press `Ctrl+C` to stop all services
- **Manual**: Stop each service individually with `Ctrl+C` in their respective terminals
