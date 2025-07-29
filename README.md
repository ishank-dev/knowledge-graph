# Knowledge Graph Extraction

A multi-source knowledge graph extractor using Large Language Models with a two-phase Extract-Build agentic workflow. Extract meaningful relationships from any text source and build consistent knowled## 🚨 Troubleshooting

### Common Issues

**1. Missing API Key**
```bash
Error: No API key found
```
Solution: Create `.env` file with `GEMINI_API_KEY=your_key_here`

**2. Package Import Errors**
```bash
ModuleNotFoundError: No module named 'faiss'
```
Solution: Install missing dependencies: `pip install -r requirements.txt`

**3. CUDA/GPU Issues**
```bash
CUDA out of memory
```
Solution: Use CPU-only models or reduce batch size in HuggingFace engines

**4. Rate Limiting**
```bash
API rate limit exceeded
```
Solution: Add delays between API calls or upgrade your Gemini API plan

### Performance Tips

- **Large texts**: The system automatically chunks text (default 4096 characters)
- **Memory usage**: Use smaller embedding models for limited RAM
- **Speed**: Gemini models are faster than local HuggingFace models
- **Accuracy**: Larger models (GPT-4, Claude) generally produce better results

## 🤝 Contributing

This is an educational project. Feel free to:
- Fork and experiment with different LLM backends
- Try custom entity types for your domain
- Improve the visualization components
- Add support for new data sources

## 📄 License

This code is meant for educational purposes only. See the accompanying blog post for more details.

## 🔗 Related Resources

- **Original Blog Post**: [How To Build a Multi-Source Knowledge Graph Extractor from Scratch](https://github.com/GabrieleSgroi/knowledge_graph_extraction)
- **Interactive Notebook**: [Google Colab](https://colab.research.google.com/drive/1st_E7SBEz5GpwCnzGSvKaVUiQuKv3QGZ)
- **Google Gemini API**: [Get API Key](https://makersuite.google.com/app/apikey)
- **NetworkX Documentation**: [Graph Analysis](https://networkx.org/)
- **FAISS Documentation**: [Vector Search](https://faiss.ai/)

---

**Ready to extract knowledge graphs?** Start with `python example_usage.py` and explore the documentation above! 🚀s with entity disambiguation and source linking.

*This code is meant for educational purposes only.*

This code accompanies the blog post [How To Build a Multi-Source Knowledge Graph Extractor from Scratch](https://github.com/GabrieleSgroi/knowledge_graph_extraction). See the [Colab notebook](https://colab.research.google.com/drive/1st_E7SBEz5GpwCnzGSvKaVUiQuKv3QGZ) for example usage.

## ✨ Key Features

- **Two-phase agentic workflow**: Extract relations, then build consistent knowledge graph
- **Entity disambiguation**: Maintains consistency across different text sources
- **Flexible LLM backends**: Supports Gemini and local HuggingFace models
- **Custom entity types**: Define domain-specific entity categories
- **Source linking**: Links relations back to original text passages
- **Rich visualizations**: Built-in graph visualization using NetworkX and matplotlib
- **Multiple data sources**: Works with Wikipedia, custom text, image metadata, and more

## 🚀 Quick Installation

1. **Clone the repository** (if not already done)
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file and add your API key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

4. **Run the examples:**
   ```bash
   python example_usage.py
   ```

## 📋 Requirements

- **Python 3.12+**
- **Dependencies:** torch, transformers, google-genai, langchain-huggingface, faiss-cpu, networkx, matplotlib, wikipedia, python-dotenv
- **API Key:** Google Gemini API key (get one at [Google AI Studio](https://makersuite.google.com/app/apikey))
- **Optional:** CUDA-capable GPU for local HuggingFace models

## 🏃‍♂️ Quick Start

### Option 1: Run Example Scripts
```python
from example_usage import run_basic_example, run_image_metadata_example

# Extract from Wikipedia article
run_basic_example()

# Extract from personal image metadata
run_image_metadata_example()
```

### Option 2: Basic Text Extraction
```python
from kg_builder.workflow import EBWorkflow
from kg_builder.engine import GeminiEngine
from kg_builder.prompts.prompting import ExtractorPrompting, BuilderPrompting
from kg_builder.relations import RelationsData
from langchain_huggingface import HuggingFaceEmbeddings

# Setup
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
workflow = EBWorkflow(embeddings=embeddings)

# Initialize engines
builder_engine = GeminiEngine("gemini-2.0-flash-exp", BuilderPrompting())
extractor_engine = GeminiEngine("gemini-2.0-flash-exp", ExtractorPrompting())

# Create empty knowledge graph with custom entity types
relations_data = RelationsData.empty(
    allowed_entity_types=["PERSON", "COMPANY", "LOCATION", "EVENT", "CONCEPT"]
)

# Extract knowledge graph
text = "Your text here..."
result = workflow(
    text=text,
    relations_data=relations_data,
    builder_engine=builder_engine,
    extractor_engine=extractor_engine
)

# Visualize and save results
result.plot_graph()
result.save_to_json("my_knowledge_graph.json")
```

## ⚙️ Configuration

Edit `.env` file to customize:
```env
GEMINI_API_KEY=your_actual_api_key_here
DEFAULT_MODEL=gemini-2.0-flash-exp
CHUNK_SIZE=4096
CHUNK_OVERLAP=128
```

### Available Options:
- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `DEFAULT_MODEL`: Model to use (default: gemini-2.0-flash-exp)
- `CHUNK_SIZE`: Text chunk size for processing (default: 4096)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 128)

## 💡 Usage Examples

The repository includes several ready-to-run examples:

### 1. Wikipedia Knowledge Extraction
Extract entities and relationships from Wikipedia articles:
```bash
python example_usage.py
# Choose option 1 when prompted
```

### 2. Personal Image Metadata
Extract knowledge graphs from personal photo metadata:
```bash
python example_usage.py
# Choose option 2 when prompted
```

### 3. Custom Entity Types
Define your own entity categories for domain-specific extraction:
```python
# Finance domain
allowed_entity_types = ["COMPANY", "PERSON", "FINANCIAL_INSTRUMENT", "MARKET", "REGULATION"]

# Medical domain
allowed_entity_types = ["PERSON", "CONDITION", "TREATMENT", "MEDICATION", "ANATOMY"]

# Academic domain
allowed_entity_types = ["PERSON", "INSTITUTION", "RESEARCH_AREA", "PUBLICATION", "CONCEPT"]
```

## 📚 Complete Documentation

### 🚀 Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** - Step-by-step setup and first run guide

### 🔧 Technical Deep Dives
- **[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)** - Complete system architecture and component analysis
- **[TRIPLET_GENERATION.md](TRIPLET_GENERATION.md)** - Detailed explanation of how triplets are generated and processed
- **[ENTITY_TYPES_FLOW.md](ENTITY_TYPES_FLOW.md)** - How entity types flow through the system and how to customize them

### 📊 Analysis & Examples
- **[ANALYSIS.md](ANALYSIS.md)** - In-depth analysis of the image metadata knowledge graph example
- **[example_usage.py](example_usage.py)** - Complete runnable examples with both Wikipedia and image metadata
- **[sample_image_metadata.json](sample_image_metadata.json)** - Sample personal data for testing

### 🎨 Visual Resources
- **System Architecture Diagram** - `system_architecture_diagram.png`
- **Sample Knowledge Graph** - `image_metadata_knowledge_graph.png`

## 🏗️ System Architecture

The system uses a **two-phase agentic workflow**:

1. **Extract Phase**: LLM identifies and extracts triplets (subject, relation, object) from text chunks
2. **Build Phase**: Another LLM consolidates triplets, performs entity disambiguation, and builds the final knowledge graph

Key components:
- **EBWorkflow**: Orchestrates the two-phase process
- **GeminiEngine**: Handles LLM interactions with Google Gemini
- **RelationsData**: Manages entity types and relationship storage
- **Embeddings**: Uses HuggingFace sentence transformers for semantic similarity
- **FAISS**: Vector database for efficient similarity search

## 🔍 What Gets Extracted

The system extracts structured knowledge in triplet format:
```
(subject:entity_type, relation, object:entity_type)
```

**Example triplets from personal data:**
```
(Alex:PERSON, lives_in, San Francisco:LOCATION)
(Alex:PERSON, works_at, Tech Company:ORGANIZATION)
(Birthday Party:EVENT, attended_by, Alex:PERSON)
(Wedding:EVENT, location, Napa Valley:LOCATION)
```

**Supported outputs:**
- JSON knowledge graph with source linking
- NetworkX graph object for programmatic access
- PNG visualization with matplotlib
- Relationship statistics and entity counts

## 🎯 Use Cases

- **Personal Knowledge Management**: Extract insights from journals, photos, documents
- **Research & Academia**: Build knowledge graphs from papers, notes, research data
- **Business Intelligence**: Extract relationships from reports, emails, documents
- **Content Analysis**: Understand relationships in articles, books, social media
- **Domain-Specific Extraction**: Finance, healthcare, legal, technical documentation

## 🛠️ Customization

### Custom Entity Types
Define domain-specific entities in your code:
```python
# Medical domain
medical_entities = ["PATIENT", "DOCTOR", "CONDITION", "TREATMENT", "MEDICATION", "HOSPITAL"]

# Legal domain  
legal_entities = ["PERSON", "ORGANIZATION", "LAW", "CASE", "COURT", "CONTRACT"]

# Academic domain
academic_entities = ["RESEARCHER", "INSTITUTION", "PUBLICATION", "CONCEPT", "METHOD", "DATASET"]
```

### Custom Prompts
Modify the system prompts in `kg_builder/prompts/`:
- `extractor_system_prompt.txt` - Controls triplet extraction
- `builder_system_prompt.txt` - Controls knowledge graph building

### Alternative LLM Backends
Switch to local HuggingFace models:
```python
from kg_builder.engine import HuggingFaceEngine

# Use local model instead of Gemini
extractor_engine = HuggingFaceEngine("microsoft/DialoGPT-medium", ExtractorPrompting())
```

## Requirements

- Python 3.12+
- CUDA-capable GPU (for local models, optional)
- Gemini API key (for cloud models)ADME
This code is meant for educational purposes only. 

This code accompanies the blog post How To Build a Multi-Source Knowledge Graph Extractor from Scratch. See the [Colab notebook](https://colab.research.google.com/drive/1st_E7SBEz5GpwCnzGSvKaVUiQuKv3QGZ) for example usage.
