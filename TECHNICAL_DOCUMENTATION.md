# 📚 Knowledge Graph Extraction System - Technical Documentation

## 🔍 How It Works: Complete System Overview

### 🏗️ Architecture

The system implements a **two-phase agentic workflow** for building consistent knowledge graphs from unstructured text:

```
Input Text → [Phase 1: Extract] → Raw Triplets → [Phase 2: Build] → Consistent Knowledge Graph
```

### 📋 Phase Overview

1. **Extract Phase**: LLM extracts raw relation triplets from text chunks
2. **Build Phase**: LLM validates, deduplicates, and integrates triplets into consistent graph

---

## 🔧 Phase 1: Extract - Triplet Generation

### 📝 What are Triplets?

Triplets are the fundamental unit of knowledge representation in the format:
```
(subject:entity_type, relation, object:entity_type)
```

**Examples from our image metadata:**
- `(alex:PERSON, graduated from, columbia university:LOCATION)`
- `(sam:PERSON, organized, 28th birthday:EVENT)`
- `(alex:PERSON, volunteers at, local animal shelter:LOCATION)`

### 🔄 Extract Phase Workflow

#### Step 1: Text Preprocessing
```python
# Text is split into manageable chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=4096,        # Characters per chunk
    chunk_overlap=128,      # Overlap between chunks
    separators=["\n\n", "\n", "\t", ".", " ", ""]
)
chunks = splitter.split_text(text)
```

#### Step 2: LLM Extraction
For each chunk, the system:

1. **Sends to LLM** with extraction prompt
2. **Specifies allowed entity types** (PERSON, EVENT, LOCATION, etc.)
3. **Receives raw triplet text** from LLM
4. **Parses triplets** using regex
5. **Validates format** and entity types

### 🤖 LLM Prompt for Extraction

The extractor uses this system prompt:

```
You are an expert assistant automatizing knowledge graph creation from text. 
You are provided a text passage and some allowed entity types: you are tasked 
with extracting relations between entities of the specified type from the provided passage.

Guidelines:
- Provide your output in triplets (entity1:type1, relation, entity2:type2)
- Only extract triplets involving entities of the types specified by the user
- Don't produce duplicated triplets
- Don't extract properties or attributes as entities
- Entity names should be concise and contain all the necessary information
- Keep entity names consistent: the same entity should have the same name
- Write only the extracted triplets and nothing else
```

### 📊 Example Extraction Process

**Input Text:**
```
Alex graduated from Columbia University in New York City. Professor Smith handed 
him his graduation diploma during the ceremony.
```

**Allowed Entity Types:** `['PERSON', 'EVENT', 'LOCATION', 'OBJECT']`

**LLM Output:**
```
(alex:PERSON, graduated from, columbia university:LOCATION)
(professor smith:PERSON, handed, graduation diploma:OBJECT)
(alex:PERSON, attended, graduation ceremony:EVENT)
```

### 🔍 Parsing and Validation

The system uses regex to extract triplets from LLM output:

```python
# Regex pattern to match triplet format
pattern = r'\(([^:]+):([^,]+),\s*([^,]+),\s*([^:]+):([^)]+)\)'

# Validates:
# 1. Correct format: (subject:type, relation, object:type)
# 2. Entity types are in allowed list
# 3. No empty fields
```

---

## 🔧 Phase 2: Build - Graph Construction

### 🎯 Purpose
The Build phase ensures the knowledge graph is:
- **Consistent**: Same entities have same names
- **Non-redundant**: No duplicate information
- **Coherent**: Relations make sense together

### 🔄 Build Phase Workflow

For each extracted triplet:

#### Step 1: Similarity Retrieval
```python
# Find similar existing triplets using embeddings
similar_triplets = searchable_relations.retrieve_similar_triplets(
    proposed_triplet, k=10
)
```

#### Step 2: LLM Validation
The Builder LLM decides:
1. **Accept**: Add triplet as-is
2. **Modify**: Change entity names/types for consistency
3. **Reject**: Information already exists or invalid

#### Step 3: Integration
Valid triplets are added to the knowledge graph with source passage links.

### 🤖 LLM Prompt for Building

```
You are an expert assistant helping to expand a knowledge graph with new information. 
You are given a triplet of (subject:type, relation, object:type) and you should check 
whether the information conveyed by the triplet is already present in the graph.

Guidelines:
- Discard the triplet if the information is already explicitly present
- Discard if entities are not of allowed types
- Keep the graph consistent: if entities are already present, modify the triplet 
  to match existing entity names and types
```

### 📊 Example Build Process

**Proposed Triplet:** `(marie curie:PERSON, discovery, radium:CHEMICAL)`

**Similar Existing Relations:**
```
(marie curie:PERSON, discovered, polonium:CHEMICAL)
(maria salomea skłodowska-curie:PERSON, known as, marie curie:PERSON)
```

**LLM Decision:** Accept, but note that "Marie Curie" is the consistent name form.

**Result:** `(marie curie:PERSON, discovered, radium:CHEMICAL)` ✅

---

## 🎨 Technical Implementation Details

### 🧠 Core Components

#### 1. **EBWorkflow** (Main Orchestrator)
```python
class EBWorkflow:
    def __call__(self, text, relations_data, builder_engine, extractor_engine):
        # Phase 1: Extract
        chunks = self.split_text(text)
        extracted_triplets = []
        for chunk in chunks:
            triplets = extractor.extract_from_passage(chunk, allowed_types)
            extracted_triplets.append(triplets)
        
        # Phase 2: Build
        for i, chunk in enumerate(chunks):
            builder.build_kg(extracted_triplets[i], passage=chunk)
```

#### 2. **RelationsExtractor** (Phase 1)
```python
class RelationsExtractor:
    def extract_from_passage(self, passage, allowed_types):
        messages = self.engine.prompting.get_messages(passage, allowed_types)
        decoded = self.engine.chat_completion(messages)
        return self.engine.prompting.format_triplets(decoded, allowed_types)
```

#### 3. **KGBuilder** (Phase 2)
```python
class KGBuilder:
    def build_kg(self, triplets, passage):
        for triplet in triplets:
            similar_relations = self.relations_data.retrieve_similar_triplets(triplet)
            decision = self.validate_relation(triplet, similar_relations)
            if decision != "DISCARD":
                self.relations_data.add_relation(decision, passage)
```

### 🔍 Vector Similarity Search

The system uses **FAISS** for efficient similarity search:

```python
# Convert triplets to embeddings for similarity search
documents = [Document(get_triplet_string(t)) for t in triplets]
vector_store = FAISS.from_documents(documents, embeddings)

# Retrieve similar triplets during Build phase
similar_docs = vector_store.similarity_search(query_triplet, k=10)
```

### 📊 Data Structures

#### RelationsData
```python
class RelationsData:
    annotated_passages: dict[str, list[tuple[str, str, str]]]  # passage -> triplets
    allowed_entity_types: list[str]                           # valid entity types
    
    @property
    def flattened_triplets(self) -> list[tuple[str, str, str]]:
        # All triplets across all passages
    
    @property 
    def networkx_graph(self) -> DiGraph:
        # NetworkX graph representation
```

---

## 📈 Performance Characteristics

### ⏱️ Timing Breakdown (from our run)
- **Extract Phase**: ~4.5 seconds for 7 images (2874 characters)
- **Build Phase**: ~131 seconds for 24 triplets
- **Total Relations**: 24 triplets from 1 consolidated passage

### 🎯 Quality Metrics
- **Entity Consistency**: ✅ All "alex" entities normalized to lowercase
- **Relationship Diversity**: 24 different relation types extracted
- **Source Traceability**: ✅ All triplets linked to source passages

### 📊 Extracted Entity Types Distribution
```json
{
  "PERSON": ["alex", "sam", "mike", "sister", "parents", "professor smith", "barista maria", "jane", "brother-in-law"],
  "EVENT": ["28th birthday", "graduation", "sister's wedding", "family trip", "hiking trip"],
  "LOCATION": ["bluebird café", "central park", "columbia university", "mountain trails", "local animal shelter"]
}
```

---

## 🎛️ Configuration Options

### Entity Types
Customize what to extract:
```python
entity_types = [
    "PERSON",      # People
    "EVENT",       # Events/activities  
    "LOCATION",    # Places
    "DATE",        # Temporal information
    "EMOTION",     # Emotional states
    "OBJECT",      # Physical objects
    "RELATIONSHIP" # Relationship types
]
```

### Processing Parameters
```python
SplitConfig(
    chunk_char_size=4096,    # Chunk size for processing
    chunk_char_overlap=128,  # Overlap between chunks
    separators=["\n\n", "\n", "\t", ".", " ", ""]  # Text splitting rules
)
```

### LLM Parameters
```python
{
    "max_new_tokens": 4096,  # Maximum response length
    "temperature": 0.0,      # Deterministic output
    "model": "gemini-2.0-flash-exp"  # LLM model choice
}
```

---

## 🔬 Advanced Features

### 🔗 Source Passage Linking
Every triplet maintains a link to its source:
```python
# Each triplet can be traced back to original text
triplet = ("alex:PERSON", "graduated from", "columbia university:LOCATION")
source = "Alex graduated from Columbia University in New York City..."
```

### 🎨 Graph Visualization
Automatic NetworkX visualization with:
- **Nodes**: Entities (sized by name length)
- **Edges**: Relations (labeled with relationship type)
- **Colors**: Entity types (customizable)

### 💾 Export Formats
- **JSON**: Structured data with source passages
- **NetworkX**: Python graph object for analysis
- **PNG**: Visual representation

---

## 🛠️ Extensibility

### Custom Entity Types
```python
# Add domain-specific entity types
entity_types = ["GENE", "PROTEIN", "DISEASE", "DRUG"]  # Biomedical
entity_types = ["COMPANY", "PRODUCT", "TECHNOLOGY"]    # Business
entity_types = ["CHARACTER", "PLOT", "THEME"]          # Literature
```

### Custom Prompts
Modify extraction behavior by customizing prompts:
```python
# Focus on specific relationship types
"Extract only temporal relationships..."
"Focus on causal relationships..."
"Identify hierarchical structures..."
```

This system provides a robust foundation for automated knowledge graph construction from any domain-specific text or metadata! 🚀
