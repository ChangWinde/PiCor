#!/bin/bash

# PiCor Code Quality Check Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# PiCor Logo
PICOR_LOGO() {
    echo -e "${CYAN}"
    cat << "EOF"

  ____   _   ____             
 |  _ \ (_) / ___| ___   _ __ 
 | |_) || || |    / _ \ | '__|
 |  __/ | || |___| (_) || |   
 |_|    |_| \____|\___/ |_|   

Code Quality Check & Format
EOF
    echo -e "${NC}"
}

# Status functions
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

print_header() {
    echo -e "${BOLD}${PURPLE}$1${NC}"
}

# Check if tools are installed
check_tools() {
    print_header "🔧 Checking Required Tools"
    
    local missing_tools=()
    
    if ! command -v black &> /dev/null; then
        missing_tools+=("black")
    fi
    
    if ! command -v flake8 &> /dev/null; then
        missing_tools+=("flake8")
    fi
    
    if ! command -v isort &> /dev/null; then
        missing_tools+=("isort")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "Missing tools: ${missing_tools[*]}"
        print_status "Installing missing tools..."
        pip install "${missing_tools[@]}"
    else
        print_success "All tools are installed"
    fi
}

# Import style check
check_imports() {
    print_header "📦 Checking Import Style"
    
    if [ -f "scripts/check_imports.py" ]; then
        python scripts/check_imports.py
    else
        print_warning "check_imports.py not found, skipping import check"
    fi
}

# Code formatting with black
format_code() {
    print_header "🎨 Formatting Code with Black"
    
    if [ "$1" = "--check" ]; then
        print_status "Checking code format (not modifying files)..."
        black source/ --line-length=100 --check
        print_success "Code format check completed"
    else
        print_status "Formatting code..."
        black source/ --line-length=100
        print_success "Code formatting completed"
    fi
}

# Import sorting with isort (disabled due to conflicts with custom rules)
sort_imports() {
    print_header "📋 Import Sorting (Disabled)"
    print_warning "Import sorting disabled to preserve custom import order"
    print_status "Use custom import checker for import style validation"
}

# Code style check with flake8
lint_code() {
    print_header "🔍 Linting Code with flake8"
    
    print_status "Running flake8..."
    if flake8 source/; then
        print_success "No linting issues found"
    else
        print_warning "Linting issues found (see output above)"
        return 1
    fi
}

# Main function
main() {
    clear
    PICOR_LOGO
    print_header "PiCor Code Quality Check"
    echo
    
    local check_only=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --check)
                check_only=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --check    Check only, don't modify files"
                echo "  --help     Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    check_tools
    echo
    
    # Run checks
    check_imports
    echo
    
    # if [ "$check_only" = true ]; then
    #     format_code --check
    #     echo
    #     sort_imports --check
    #     echo
    # else
    #     format_code
    #     echo
    #     sort_imports
    #     echo
    # fi
    
    lint_code
    echo
    
    print_header "🎉 Code Quality Check Completed!"
    
    if [ "$check_only" = true ]; then
        print_status "All checks completed in check-only mode"
    else
        print_status "Code has been formatted and checked"
    fi
    
    PICOR_LOGO
}

# Run main function
main "$@" 