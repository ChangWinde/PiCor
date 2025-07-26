#!/bin/bash

# PiCor Installation Script
# High-quality installation script with colored output and PiCor branding

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
                          
Multi-task Deep Reinforcement Learning with Policy Correction
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

print_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

print_header() {
    echo -e "${BOLD}${CYAN}$1${NC}"
}

# Check if we're in a virtual environment
check_environment() {
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_warning "You're not in a virtual environment."
        print_status "It's recommended to use uv for environment management."
        read -p "Continue with uv installation? (Y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            print_error "Installation cancelled."
            exit 1
        fi
    fi
}

# Install uv if not available
install_uv() {
    if ! command -v uv &> /dev/null; then
        print_step "Installing uv package manager..."
        
        if command -v curl &> /dev/null; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
            source ~/.bashrc 2>/dev/null || source ~/.zshrc 2>/dev/null || true
            print_success "uv installed successfully"
        else
            print_error "curl is not available. Please install uv manually:"
            print_status "Visit: https://docs.astral.sh/uv/getting-started/installation/"
            exit 1
        fi
    else
        print_success "uv is already available"
    fi
}

# Install dependencies
install_dependencies() {
    print_step "Installing project dependencies..."
    
    # Create virtual environment
    print_status "Creating virtual environment..."
    uv venv --clear
    
    # Fix build tools for gym==0.19.0 compatibility
    print_step "Installing compatible build tools..."
    uv pip uninstall cython -y || true
    uv pip install cython==0.29.21
    uv pip install setuptools==59.5.0
    uv pip install wheel==0.38.0
    print_success "Build tools installed"
    
    # Install base dependencies
    print_step "Installing base dependencies..."
    uv pip install torch torchrl tensordict numpy wandb pyyaml python-dotenv
    print_success "Base dependencies installed"
    
    # Install gym with specific version
    print_step "Installing gym==0.19.0..."
    uv pip install gym==0.19.0
    print_success "Gym installed"
    
    # Install other specific packages
    print_step "Installing other packages..."
    uv pip install mujoco_py==2.0.2.8
    uv pip install garage>=2022.1.0
    print_success "Other packages installed"
    
    # Install custom local libraries
    print_step "Installing custom local libraries..."
    
    if [ -d "./libs/custom_dmcontrol" ]; then
        print_status "Installing custom_dmcontrol..."
        uv pip install -e ./libs/custom_dmcontrol
        print_success "custom_dmcontrol installed"
    else
        print_warning "custom_dmcontrol directory not found"
    fi
    
    if [ -d "./libs/custom_dmc2gym" ]; then
        print_status "Installing custom_dmc2gym..."
        uv pip install -e ./libs/custom_dmc2gym
        print_success "custom_dmc2gym installed"
    else
        print_warning "custom_dmc2gym directory not found"
    fi
    
    # Install specific Metaworld version
    print_step "Installing specific Metaworld version..."
    uv pip install git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb
    print_success "Metaworld installed"
}

# Verify installation
verify_installation() {
    print_step "Verifying installation..."
    
    # Test basic imports
    if uv run python -c "import torch; import gym; import metaworld; print('Core dependencies OK')" 2>/dev/null; then
        print_success "Core dependencies verified"
    else
        print_warning "Some dependencies may not be properly installed"
    fi
    
    # Test main script
    if uv run python source/main.py --help &>/dev/null; then
        print_success "PiCor main script is working"
    else
        print_warning "Main script test failed"
    fi
}

# Main installation function
main() {
    clear
    PICOR_LOGO
    print_header "PiCor Installation Script"
    echo
    
    check_environment
    install_uv
    install_dependencies
    verify_installation
    
    echo
    print_header "🎉 Installation Completed Successfully!"
    echo
    print_status "Next steps:"
    echo "  1. Set up environment variables:"
    echo "     export WANDB_API_KEY='your_key'"
    echo "     export WANDB_PROJECT='picor'"
    echo "     export WANDB_ENTITY='your_username'"
    echo
    echo "  2. Run PiCor:"
    echo "     uv run python source/main.py --help"
    echo
    echo "  3. Start training:"
    echo "     uv run python source/main.py --experiment PiCor --env_name metaworld_mt10 --seed 0"
    echo
    print_status "💡 Tips:"
    echo "  • Use 'uv run' to run commands in the virtual environment"
    echo "  • Use 'uv add <package>' to add new dependencies"
    echo "  • Use 'uv sync' to sync dependencies after changes"
    echo
    PICOR_LOGO
}

# Run main function
main "$@" 