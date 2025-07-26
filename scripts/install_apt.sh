#!/bin/bash

# PiCor System Dependencies Installation Script
# This script installs required system packages for PiCor on Ubuntu/Debian systems

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if running as root or with sudo
check_sudo() {
    if [ "$EUID" -ne 0 ]; then
        print_error "This script requires sudo privileges to install system packages"
        print_status "Please run: sudo $0"
        exit 1
    fi
}

# Update package list
update_packages() {
    print_status "Updating package list..."
    apt-get update
    print_success "Package list updated"
}

# Install system dependencies
install_dependencies() {
    print_status "Installing system dependencies..."
    
    # Graphics and OpenGL libraries
    apt-get install -y \
        libglfw3 \
        libglfw3-dev \
        libgl1-mesa-dev \
        libgl1-mesa-glx \
        libglew-dev \
        libosmesa6-dev \
        libegl1 \
        libopengl0 \
        glew-utils
    
    # Development tools
    apt-get install -y \
        gcc \
        build-essential \
        python3-dev \
        libevent-dev
    
    # Utility tools
    apt-get install -y \
        patchelf \
        unzip \
        screen \
        htop \
        tmux
    
    print_success "System dependencies installed"
}

# Main function
main() {
    echo "PiCor System Dependencies Installation"
    echo "====================================="
    echo
    
    check_sudo
    update_packages
    install_dependencies
    
    echo
    print_success "System dependencies installation completed!"
    print_status "You can now run the PiCor installation script:"
    echo "  ./scripts/install.sh"
}

# Run main function
main "$@"
