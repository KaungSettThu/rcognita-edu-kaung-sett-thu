#!/bin/bash

# This script installs a list of Python packages using pip.
# It is highly recommended to use a virtual environment.

# --- Configuration ---
# List of packages and their versions to install
PACKAGES=(
    "contourpy==1.1.0"
    "cycler==0.11.0"
    "fonttools==4.40.0"
    "kiwisolver==1.4.4"
    "matplotlib==3.7.2"
    "numpy==1.24.0"
    "packaging==23.1"
    "Pillow==10.0.0"
    "pyparsing==3.0.9"
    "python-dateutil==2.8.2"
    "scipy==1.10.1"
    "six==1.16.0"
    "svgpath2mpl==1.0.0"
    "tabulate==0.9.0"
)

# --- Virtual Environment Setup (Recommended) ---
# Uncomment the lines below to use a virtual environment.
# Replace 'my_project_env' with your desired environment name.

# ENV_DIR="my_project_env"

# echo "Checking for Python 3..."
# if ! command -v python3 &> /dev/null
# then
#     echo "Python 3 is not installed. Please install Python 3 first."
#     exit 1
# fi

# echo "Creating virtual environment '$ENV_DIR'..."
# python3 -m venv "$ENV_DIR" || { echo "Failed to create virtual environment."; exit 1; }

# echo "Activating virtual environment..."
# source "./$ENV_DIR/bin/activate" || { echo "Failed to activate virtual environment."; exit 1; }
# echo "Virtual environment activated."

# --- Installation Process ---
echo "Installing Python packages..."

for package in "${PACKAGES[@]}"; do
    echo "Attempting to install: $package"
    # Using 'pip3' is generally safer to ensure you're using Python 3's pip.
    # If you're in a virtual environment, 'pip' will also work.
    # If installing globally and you encounter permission errors, uncomment 'sudo'.
    pip3 install "$package" || { echo "Failed to install $package. Aborting."; exit 1; }
    # sudo pip3 install "$package" # Uncomment if you need sudo for global install
done

echo "All specified packages have been installed."

# --- Deactivate Virtual Environment (if used) ---
# Uncomment the line below if you activated a virtual environment above.
# echo "Deactivating virtual environment..."
# deactivate

echo "Script finished."