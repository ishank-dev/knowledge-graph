#!/bin/bash

# Knowledge Graph Extraction Setup Script

echo "🚀 Setting up Knowledge Graph Extraction environment..."
echo "=================================================="

# Check if Python 3.12+ is available
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.12"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,12) else 1)" 2>/dev/null; then
    echo "❌ Python 3.12+ required. Current version: $python_version"
    echo "Please install Python 3.12 or higher"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Get a Gemini API key from: https://ai.google.dev/"
echo "3. Edit .env file and add your API key"
echo "4. Run the example: python run.py"
echo ""
echo "📚 Files created:"
echo "   • .env - Configuration file (add your API key here)"
echo "   • requirements.txt - Python dependencies"
echo "   • example_usage.py - Example script"
echo "   • run.py - Quick start script"
echo ""
