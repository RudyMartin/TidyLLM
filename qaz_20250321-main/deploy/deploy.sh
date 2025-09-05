#!/bin/bash
# VectorQA Sage - Deployment Script
# Auth: Rudy Martin - Next Shift Consulting LLC
# Date: 2025-08-23

set -e  # Exit on any error

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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if we're in the right directory
    if [[ ! -f "Dockerfile" ]]; then
        print_error "Dockerfile not found. Make sure you're in the environ_settings directory."
        exit 1
    fi
    
    # Check if app directory exists
    if [[ ! -d "../app" ]]; then
        print_error "App directory not found. Make sure you extracted both zip files."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to deploy with Docker
deploy_docker() {
    print_status "Deploying with Docker..."
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command_exists docker-compose; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Build and start containers
    print_status "Building and starting containers..."
    docker-compose up -d --build
    
    # Wait for application to start
    print_status "Waiting for application to start..."
    sleep 10
    
    # Check if application is running
    if curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
        print_success "Application is running successfully!"
        print_status "Access the application at: http://localhost:8501"
    else
        print_error "Application failed to start. Check logs with: docker-compose logs"
        exit 1
    fi
}

# Function to deploy with virtual environment
deploy_venv() {
    print_status "Deploying with virtual environment..."
    
    if ! command_exists python3; then
        print_error "Python 3 is not installed. Please install Python 3.8+ first."
        exit 1
    fi
    
    # Create virtual environment
    print_status "Creating virtual environment..."
    cd ../app
    python3 -m venv venv
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Install dependencies
    print_status "Installing dependencies..."
    pip install -r requirements_demo.txt
    
    # Start application
    print_status "Starting application..."
    nohup streamlit run src/main.py --server.port=8501 --server.address=0.0.0.0 > app.log 2>&1 &
    
    # Wait for application to start
    print_status "Waiting for application to start..."
    sleep 5
    
    # Check if application is running
    if curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
        print_success "Application is running successfully!"
        print_status "Access the application at: http://localhost:8501"
        print_status "Logs are available in: app.log"
    else
        print_error "Application failed to start. Check logs in app.log"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    echo "VectorQA Sage Deployment Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  docker    Deploy using Docker (recommended)"
    echo "  venv      Deploy using virtual environment"
    echo "  stop      Stop the application"
    echo "  logs      Show application logs"
    echo "  status    Check application status"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 docker    # Deploy with Docker"
    echo "  $0 venv      # Deploy with virtual environment"
    echo "  $0 stop      # Stop the application"
}

# Function to stop application
stop_application() {
    print_status "Stopping application..."
    
    if command_exists docker-compose && [[ -f "docker-compose.yml" ]]; then
        docker-compose down
        print_success "Docker containers stopped"
    else
        # Stop virtual environment deployment
        pkill -f "streamlit run" || true
        print_success "Virtual environment application stopped"
    fi
}

# Function to show logs
show_logs() {
    print_status "Showing application logs..."
    
    if command_exists docker-compose && [[ -f "docker-compose.yml" ]]; then
        docker-compose logs -f
    else
        if [[ -f "../app/app.log" ]]; then
            tail -f ../app/app.log
        else
            print_error "No log file found"
        fi
    fi
}

# Function to check status
check_status() {
    print_status "Checking application status..."
    
    if curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
        print_success "Application is running"
        print_status "Health check: http://localhost:8501/_stcore/health"
    else
        print_error "Application is not running"
    fi
}

# Main script
main() {
    case "${1:-help}" in
        docker)
            check_prerequisites
            deploy_docker
            ;;
        venv)
            check_prerequisites
            deploy_venv
            ;;
        stop)
            stop_application
            ;;
        logs)
            show_logs
            ;;
        status)
            check_status
            ;;
        help|*)
            show_usage
            ;;
    esac
}

# Run main function
main "$@"
