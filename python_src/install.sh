# Check if virtual environment already exists and is properly set up
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    echo "Virtual environment already exists. Skipping installation..."
else
    echo "Creating a Python virtual environment..."

    # Check if Python 3.12 is available
    if command -v python3.12 &> /dev/null; then
        echo "Python 3.12 found. Using Python 3.12..."
        PYTHON_CMD="python3.12"
    # Check if Python 3.11 is available
    elif command -v python3.11 &> /dev/null; then
        echo "Python 3.11 found. Using Python 3.11..."
        PYTHON_CMD="python3.11"
    # Check if Python 3.10 is available
    elif command -v python3.10 &> /dev/null; then
        echo "Python 3.10 found. Using Python 3.10..."
        PYTHON_CMD="python3.10"
    else
        echo "Error: None of Python 3.12, 3.11, or 3.10 is installed."
        echo ""
        echo "Please install Python 3.12, 3.11, or 3.10 and try again."
        echo ""
        echo "Installation hints:"
        echo "- Ubuntu/Debian: sudo apt-get update && sudo apt-get install python3.12 python3.12-venv python3.12-dev"
        echo "- macOS: brew install python@3.12"
        echo "- Windows: Download from https://www.python.org/downloads/"
        echo "- Using pyenv: pyenv install 3.12"
        echo ""
        exit 1
    fi

    # Create a virtual environment
    $PYTHON_CMD -m venv venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate


# First install the cpu or cuda version of torch
echo "Installing PyTorch 2.5.1..." # TODO: Update to the latest version, need to fix the weights_only issue first for the used packages
# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA is available. Installing PyTorch with CUDA support..."
    pip install torch~=2.5.1 --index-url https://download.pytorch.org/whl/cu118
else
    echo "CUDA is not available. Installing CPU-only version of PyTorch..."
    pip install torch~=2.5.1 --index-url https://download.pytorch.org/whl/cpu
fi

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r python_src/requirements.txt

echo "Installation complete! To activate the virtual environment, run:"
echo "source venv/bin/activate"
