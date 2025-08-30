#!/bin/bash

# Development mode script for Medical Summarizer
# This script runs both backend and frontend without Docker

echo "🚀 Starting Medical Summarizer in Development Mode"
echo "=================================================="

# Function to cleanup background processes
cleanup() {
    echo "🛑 Shutting down services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check if we're in the right directory
if [ ! -f "backend/main.py" ] || [ ! -f "frontend/package.json" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "backend/.venv" ]; then
    echo "❌ Error: Backend virtual environment not found. Please run:"
    echo "   cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "❌ Error: Frontend dependencies not installed. Please run:"
    echo "   cd frontend && pnpm install"
    exit 1
fi

echo "✅ Dependencies check passed"

# Start backend server
echo "🔧 Starting backend server..."
cd backend
source .venv/bin/activate
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Check if backend is running
if ! curl -s http://127.0.0.1:8000/ > /dev/null; then
    echo "❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo "✅ Backend running on http://127.0.0.1:8000"

# Start frontend server
echo "🎨 Starting frontend server..."
cd frontend
pnpm run dev &
FRONTEND_PID=$!
cd ..

# Wait a moment for frontend to start
sleep 3

echo "✅ Frontend should be running on http://localhost:5173"
echo ""
echo "🌐 Services:"
echo "   Backend API: http://127.0.0.1:8000"
echo "   Frontend:    http://localhost:5173"
echo "   API Docs:    http://127.0.0.1:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user to stop
wait
