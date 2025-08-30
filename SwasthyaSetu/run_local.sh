#!/bin/bash

# Medical Summarizer - Local Deployment Script
# This script sets up and runs the complete Medical Summarizer system locally

set -e  # Exit on any error

echo "🏥 Medical Summarizer - Dual Perspective AI"
echo "==========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed and running
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not available. Please make sure 'docker compose' is a valid command."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are ready!"
}

# Setup environment variables
setup_environment() {
    print_status "Setting up environment variables..."
    
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            print_success "Created .env file from template"
            print_warning "Please review and update .env file with your settings"
        else
            print_warning "No .env.example found, creating basic .env file"
            cat > .env << EOF
MYSQL_ROOT_PASSWORD=password
MYSQL_DATABASE=medical_summarizer
MYSQL_USER=meduser
MYSQL_PASSWORD=medpass
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
REACT_APP_API_URL=http://localhost:8000
EOF
        fi
    else
        print_success "Environment file (.env) already exists"
    fi
}

# Build and start services
start_services() {
    print_status "Building and starting services..."
    
    # Stop any existing containers
    docker compose down --remove-orphans
    
    # Build and start services
    DOCKER_BUILDKIT=0 docker compose up --build -d
    
    print_success "Services started successfully!"
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for database
    print_status "Waiting for database..."
    timeout=60
    while ! docker compose exec -T mysql mysqladmin ping -h localhost --silent; do
        timeout=$((timeout - 1))
        if [ $timeout -eq 0 ]; then
            print_error "Database failed to start within 60 seconds"
            exit 1
        fi
        sleep 1
    done
    print_success "Database is ready!"
    
    # Wait for backend
    print_status "Waiting for backend API..."
    timeout=60
    while ! curl -f http://localhost:8000/ &> /dev/null; do
        timeout=$((timeout - 1))
        if [ $timeout -eq 0 ]; then
            print_error "Backend API failed to start within 60 seconds"
            exit 1
        fi
        sleep 1
    done
    print_success "Backend API is ready!"
    
    # Wait for frontend
    print_status "Waiting for frontend..."
    timeout=60
    while ! curl -f http://localhost:3000/ &> /dev/null; do
        timeout=$((timeout - 1))
        if [ $timeout -eq 0 ]; then
            print_error "Frontend failed to start within 60 seconds"
            exit 1
        fi
        sleep 1
    done
    print_success "Frontend is ready!"
}

# Run database migrations (if needed)
run_migrations() {
    print_status "Running database migrations..."
    
    # The init.sql script handles initial database setup
    # In a production environment, you might want to use Alembic here
    
    print_success "Database setup completed!"
}

# Display service information
show_services() {
    print_success "🎉 Medical Summarizer is now running!"
    echo ""
    echo "📊 Service URLs:"
    echo "  Frontend:  http://localhost:3000"
    echo "  Backend:   http://localhost:8000"
    echo "  API Docs:  http://localhost:8000/docs"
    echo "  Database:  localhost:3306"
    echo ""
    echo "🔧 Management Commands:"
    echo "  View logs:     docker compose logs -f"
    echo "  Stop services: docker compose down"
    echo "  Restart:       docker compose restart"
    echo ""
    echo "👤 Demo Credentials:"
    echo "  Username: demo_user"
    echo "  Password: secret (default bcrypt hash)"
    echo ""
    echo "🏥 Features Available:"
    echo "  ✓ Dual Perspective Summarization (Patient + Clinician)"
    echo "  ✓ 5 Medical Perspectives (Information, Suggestion, Experience, Question, Cause)"
    echo "  ✓ Real-time AI Model Inference"
    echo "  ✓ User Authentication & History"
    echo "  ✓ Feedback System"
    echo "  ✓ Export Functionality"
    echo "  ✓ Dark/Light Theme"
    echo ""
}

# Cleanup function
cleanup() {
    print_status "Cleaning up..."
    docker compose down --remove-orphans
    print_success "Cleanup completed!"
}

# Main execution
main() {
    echo "Starting Medical Summarizer deployment..."
    echo ""
    
    # Check prerequisites
    check_docker
    
    # Setup environment
    setup_environment
    
    # Start services
    start_services
    
    # Wait for services
    wait_for_services
    
    # Run migrations
    run_migrations
    
    # Show service information
    show_services
    
    # Handle cleanup on exit
    trap cleanup EXIT
    
    # Keep script running and show logs
    print_status "Showing live logs (Ctrl+C to stop)..."
    docker compose logs -f
}

# Handle command line arguments
case "${1:-}" in
    "stop")
        print_status "Stopping Medical Summarizer..."
        docker compose down --remove-orphans
        print_success "Services stopped!"
        ;;
    "restart")
        print_status "Restarting Medical Summarizer..."
        docker compose restart
        print_success "Services restarted!"
        ;;
    "logs")
        docker compose logs -f
        ;;
    "status")
        docker compose ps
        ;;
    "clean")
        print_status "Cleaning up all containers and volumes..."
        docker compose down --remove-orphans --volumes
        docker system prune -f
        print_success "Cleanup completed!"
        ;;
    *)
        main
        ;;
esac

