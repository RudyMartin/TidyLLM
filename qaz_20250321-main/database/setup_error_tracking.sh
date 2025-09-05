#!/bin/bash

# PostgreSQL Error Tracking Setup Script
# This script sets up the complete error tracking system

set -e  # Exit on any error

# Configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-vectorqa}"
DB_USER="${DB_USER:-vectorqa_user}"
DB_PASSWORD="${DB_PASSWORD:-}"

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

# Function to check if PostgreSQL is running
check_postgres() {
    print_status "Checking PostgreSQL connection..."
    
    if command -v psql &> /dev/null; then
        if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" &> /dev/null; then
            print_success "PostgreSQL connection successful"
            return 0
        else
            print_error "Cannot connect to PostgreSQL. Please check your connection settings."
            print_status "Current settings:"
            echo "  Host: $DB_HOST"
            echo "  Port: $DB_PORT"
            echo "  Database: $DB_NAME"
            echo "  User: $DB_USER"
            return 1
        fi
    else
        print_error "psql command not found. Please install PostgreSQL client."
        return 1
    fi
}

# Function to create schema
create_schema() {
    print_status "Creating error tracking schema..."
    
    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "database/prompt_pipeline_error_tracking.sql"; then
        print_success "Schema created successfully"
    else
        print_error "Failed to create schema"
        return 1
    fi
}

# Function to insert mock data
insert_mock_data() {
    print_status "Inserting mock data..."
    
    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "database/mock_data_error_tracking.sql"; then
        print_success "Mock data inserted successfully"
    else
        print_error "Failed to insert mock data"
        return 1
    fi
}

# Function to run validation queries
run_validation() {
    print_status "Running validation queries..."
    
    # Test basic connectivity and table creation
    local validation_query="
    SELECT 
        'Tables created' as check_type,
        COUNT(*) as count
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
        AND table_name IN ('prompt_history', 'prompt_pipeline_errors', 'error_patterns', 'alert_history', 'real_time_context', 'batch_processing_status', 'mlflow_integration');
    "
    
    local result=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "$validation_query")
    local table_count=$(echo "$result" | grep -o '[0-9]*' | head -1)
    
    if [ "$table_count" -eq 7 ]; then
        print_success "All 7 tables created successfully"
    else
        print_warning "Only $table_count/7 tables found"
    fi
    
    # Test data insertion
    local data_query="
    SELECT 
        'prompt_history' as table_name,
        COUNT(*) as record_count
    FROM prompt_history
    UNION ALL
    SELECT 
        'prompt_pipeline_errors' as table_name,
        COUNT(*) as record_count
    FROM prompt_pipeline_errors
    UNION ALL
    SELECT 
        'error_patterns' as table_name,
        COUNT(*) as record_count
    FROM error_patterns;
    "
    
    print_status "Data validation results:"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "$data_query"
}

# Function to run example queries
run_example_queries() {
    print_status "Running example queries to demonstrate functionality..."
    
    # Query 1: Critical errors
    print_status "1. Critical errors requiring immediate attention:"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
    SELECT 
        error_id,
        timestamp,
        error_type,
        agent_name,
        resolution_status
    FROM prompt_pipeline_errors
    WHERE severity = 'critical' 
        AND resolution_status != 'resolved'
        AND timestamp >= NOW() - INTERVAL '2 hours'
    ORDER BY timestamp DESC
    LIMIT 5;
    "
    
    # Query 2: Error summary
    print_status "2. Error summary by severity (last 24 hours):"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
    SELECT 
        severity,
        COUNT(*) as error_count,
        COUNT(DISTINCT agent_name) as affected_agents
    FROM prompt_pipeline_errors
    WHERE timestamp >= NOW() - INTERVAL '24 hours'
    GROUP BY severity
    ORDER BY error_count DESC;
    "
    
    # Query 3: Agent health
    print_status "3. Agent health analysis:"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
    SELECT 
        agent_name,
        COUNT(*) as total_errors,
        COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_errors,
        ROUND((COUNT(CASE WHEN resolution_status = 'resolved' THEN 1 END) * 100.0 / COUNT(*)), 2) as resolution_rate_percent
    FROM prompt_pipeline_errors
    WHERE timestamp >= NOW() - INTERVAL '24 hours'
    GROUP BY agent_name
    ORDER BY total_errors DESC
    LIMIT 5;
    "
    
    # Query 4: System health dashboard
    print_status "4. System health dashboard (last hour):"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
    SELECT 
        'Total Prompts' as metric,
        COUNT(*) as value
    FROM prompt_history
    WHERE timestamp >= NOW() - INTERVAL '1 hour'
    UNION ALL
    SELECT 
        'Failed Prompts' as metric,
        COUNT(CASE WHEN success = false THEN 1 END) as value
    FROM prompt_history
    WHERE timestamp >= NOW() - INTERVAL '1 hour'
    UNION ALL
    SELECT 
        'Critical Errors' as metric,
        COUNT(*) as value
    FROM prompt_pipeline_errors
    WHERE severity = 'critical' 
        AND timestamp >= NOW() - INTERVAL '1 hour'
    UNION ALL
    SELECT 
        'Open Issues' as metric,
        COUNT(*) as value
    FROM prompt_pipeline_errors
    WHERE resolution_status = 'open'
        AND timestamp >= NOW() - INTERVAL '1 hour';
    "
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --host HOST       PostgreSQL host (default: localhost)"
    echo "  -p, --port PORT       PostgreSQL port (default: 5432)"
    echo "  -d, --database DB     Database name (default: vectorqa)"
    echo "  -u, --user USER       Database user (default: vectorqa_user)"
    echo "  -w, --password PASS   Database password"
    echo "  --skip-mock-data      Skip inserting mock data"
    echo "  --skip-validation     Skip validation queries"
    echo "  --help                Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD"
    echo ""
    echo "Examples:"
    echo "  $0 -h localhost -d mydb -u myuser -w mypass"
    echo "  DB_PASSWORD=mypass $0"
}

# Parse command line arguments
SKIP_MOCK_DATA=false
SKIP_VALIDATION=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            DB_HOST="$2"
            shift 2
            ;;
        -p|--port)
            DB_PORT="$2"
            shift 2
            ;;
        -d|--database)
            DB_NAME="$2"
            shift 2
            ;;
        -u|--user)
            DB_USER="$2"
            shift 2
            ;;
        -w|--password)
            DB_PASSWORD="$2"
            shift 2
            ;;
        --skip-mock-data)
            SKIP_MOCK_DATA=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=false
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_status "Starting PostgreSQL Error Tracking Setup"
    print_status "=========================================="
    
    # Check if required files exist
    if [ ! -f "database/prompt_pipeline_error_tracking.sql" ]; then
        print_error "Schema file not found: database/prompt_pipeline_error_tracking.sql"
        exit 1
    fi
    
    if [ ! -f "database/mock_data_error_tracking.sql" ]; then
        print_error "Mock data file not found: database/mock_data_error_tracking.sql"
        exit 1
    fi
    
    # Check PostgreSQL connection
    if ! check_postgres; then
        exit 1
    fi
    
    # Create schema
    if ! create_schema; then
        exit 1
    fi
    
    # Insert mock data (unless skipped)
    if [ "$SKIP_MOCK_DATA" = false ]; then
        if ! insert_mock_data; then
            exit 1
        fi
    else
        print_warning "Skipping mock data insertion"
    fi
    
    # Run validation (unless skipped)
    if [ "$SKIP_VALIDATION" = false ]; then
        if ! run_validation; then
            print_warning "Validation failed, but continuing..."
        fi
    else
        print_warning "Skipping validation"
    fi
    
    # Run example queries
    if [ "$SKIP_MOCK_DATA" = false ]; then
        run_example_queries
    fi
    
    print_success "PostgreSQL Error Tracking Setup Complete!"
    print_status ""
    print_status "Next steps:"
    print_status "1. Review the example queries in database/error_tracking_queries.sql"
    print_status "2. Integrate with your MLflow setup"
    print_status "3. Implement the Python error tracking classes"
    print_status "4. Set up monitoring dashboards"
    print_status ""
    print_status "For help with queries, run:"
    print_status "  PGPASSWORD='$DB_PASSWORD' psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f database/error_tracking_queries.sql"
}

# Run main function
main "$@"
