# 🚀 Knowledge Graph Extraction - Quick Start Guide

## 📁 Files Created

I've created all the necessary files to run the knowledge graph extraction system locally:

### Core Files
- **`.env`** - Configuration file where you add your Gemini API key
- **`requirements.txt`** - All Python dependencies 
- **`example_usage.py`** - Complete example showing how to use the system
- **`run.py`** - Quick start script that runs everything
- **`test_setup.py`** - Test script to verify your setup

### Setup Files
- **`setup.sh`** - Automated setup script (Unix/Mac)
- **`.gitignore`** - Git ignore file for common artifacts
- **`README.md`** - Updated with installation and usage instructions

## 🛠️ Installation & Setup

### Option 1: Automated Setup (Recommended)
```bash
# Make setup script executable and run it
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🔑 API Key Configuration

1. **Get a Gemini API key** from: https://ai.google.dev/
2. **Edit the `.env` file** and replace `your_gemini_api_key_here` with your actual key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

## 🧪 Testing Your Setup

Run the test script to verify everything is working:
```bash
python test_setup.py
```

## 🚀 Running the System

### Quick Start
```bash
python run.py
```
This will automatically:
- Extract a knowledge graph from sample Wikipedia articles
- Save visualization as `knowledge_graph.png`
- Save data as `extracted_knowledge_graph.json`

### Custom Usage
```python
from example_usage import run_custom_example

# Your own text and entity types
text = "Your custom text here..."
entity_types = ["PERSON", "COMPANY", "LOCATION"]
result = run_custom_example(text, entity_types)
```

## 📊 What the System Does

### Phase 1: Extract
- Splits text into chunks (4096 chars by default)
- Uses LLM to extract relation triplets: `(subject:type, relation, object:type)`
- Example: `(Mark Zuckerberg:PERSON, founded, Facebook:COMPANY)`

### Phase 2: Build  
- Evaluates each extracted relation for consistency
- Merges similar entities (e.g., "Mark Zuckerberg" and "Zuckerberg")
- Builds coherent knowledge graph avoiding duplicates

## 🎯 Key Features

- **Entity Types**: Customizable (PERSON, COMPANY, LOCATION, etc.)
- **Source Linking**: Relations linked back to original text passages
- **Visualization**: Automatic graph visualization with NetworkX
- **Export**: JSON format for further processing
- **Scalable**: Works with multiple documents/sources

## 🔧 Configuration Options

Edit `.env` file to customize:
```bash
# Model settings
DEFAULT_MODEL=gemini-2.0-flash-exp
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Processing settings  
CHUNK_SIZE=4096
CHUNK_OVERLAP=128
MAX_NEW_TOKENS=4096
TEMPERATURE=0.0
```

## 🎨 Output Files

After running, you'll get:
- **`knowledge_graph.png`** - Visual representation of the graph
- **`extracted_knowledge_graph.json`** - Structured data with all relations
- Console output showing extraction progress and summary

## 🔍 Troubleshooting

### Common Issues:

1. **Import Errors**: Run `pip install -r requirements.txt`
2. **API Key Issues**: Check `.env` file and verify key at https://ai.google.dev/
3. **Memory Issues**: Try smaller `CHUNK_SIZE` in `.env`
4. **Network Issues**: Check internet connection for API calls

### Getting Help:
- Run `python test_setup.py` to diagnose issues
- Check the Colab notebook: https://colab.research.google.com/drive/1st_E7SBEz5GpwCnzGSvKaVUiQuKv3QGZ
- Review the blog post for detailed explanations

## 📚 Next Steps

1. **Start with the basic example**: `python run.py`
2. **Try your own text**: Edit `example_usage.py` 
3. **Customize entity types**: Modify the `entity_types` list
4. **Experiment with different models**: Try local HuggingFace models
5. **Build GraphRAG applications**: Use the extracted graphs for retrieval

## ⚡ Performance Notes

- **First run**: May take longer due to model downloads
- **Gemini API**: Recommended for best results (as per blog post)
- **Local models**: Possible but may give lower quality results
- **Processing time**: Depends on text length and API response times

## 🌟 System Architecture

The workflow uses a two-phase approach:
1. **Extract**: LLM extracts raw triplets from text chunks
2. **Build**: LLM validates and merges triplets into consistent graph

This ensures entity disambiguation and prevents duplicate information while maintaining links to source material.

Happy knowledge graph building! 🎉
