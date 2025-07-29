# 🎯 Entity Types in Knowledge Graph Extraction - Complete Flow

## 📋 Where Entity Types are Set and How They're Chosen

### 🔄 **Entity Types Flow Through the System**

```
User Code → RelationsData → Workflow → Extractor → LLM → Validation
    ↓            ↓             ↓          ↓        ↓        ↓
entity_types → allowed_entity_types → allowed_types → prompt → filtering
```

---

## 1️⃣ **Initial Definition (User Choice)**

Entity types are **manually chosen by the user** based on their domain and use case:

### **Location: `example_usage.py`**

#### For Image Metadata:
```python
# Line 156 in example_usage.py
def run_image_metadata_example():
    # Define entity types relevant to personal image metadata
    entity_types = ["PERSON", "EVENT", "LOCATION", "DATE", "EMOTION", "OBJECT", "RELATIONSHIP"]
    
    # Initialize empty knowledge graph
    relations_data = RelationsData.empty(allowed_entity_types=entity_types)
```

#### For Wikipedia Articles:
```python
# Line 255 in example_usage.py  
def run_basic_example():
    # Define entity types to extract
    entity_types = ["PERSON", "COMPANY", "APPLICATION"]
    
    relations_data = RelationsData.empty(allowed_entity_types=entity_types)
```

### **How to Choose Entity Types:**

The choice depends on your **domain and goals**:

```python
# Personal/Social Media
entity_types = ["PERSON", "EVENT", "LOCATION", "EMOTION", "RELATIONSHIP"]

# Business/Corporate
entity_types = ["COMPANY", "PRODUCT", "MARKET", "REVENUE", "EXECUTIVE"]

# Academic/Research
entity_types = ["RESEARCHER", "PAPER", "INSTITUTION", "FIELD", "METHODOLOGY"]

# Medical/Healthcare
entity_types = ["PATIENT", "SYMPTOM", "DISEASE", "TREATMENT", "MEDICATION"]

# News/Journalism
entity_types = ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "DATE"]
```

---

## 2️⃣ **Storage in RelationsData Class**

### **Location: `kg_builder/relations.py`**

```python
# Lines 17-21
class RelationsData:
    def __init__(self,
                 annotated_passages: dict[str, list[tuple[str, str, str]]],
                 allowed_entity_types: list[str]):
        self.annotated_passages = annotated_passages
        # IMPORTANT: Converts to uppercase and strips whitespace
        self.allowed_entity_types = [t.strip().upper() for t in allowed_entity_types]
```

**Key Points:**
- Entity types are **automatically converted to UPPERCASE**
- Whitespace is **stripped** from each type
- Example: `["person", " Event ", "location"]` → `["PERSON", "EVENT", "LOCATION"]`

---

## 3️⃣ **Passed to Workflow**

### **Location: `kg_builder/workflow.py`**

```python
# Lines 79-81
for chunk in tqdm(chunks, desc="Extracting relations..."):
    triplets = extractor.extract_from_passage(chunk,
                                              allowed_types=relations_data.allowed_entity_types,
                                              **kwargs)
```

The workflow **extracts** `allowed_entity_types` from the `RelationsData` object and passes them to the extractor.

---

## 4️⃣ **Used in Extraction**

### **Location: `kg_builder/extractor.py`**

```python
# Lines 10-19
def extract_from_passage(self,
                         passage: str,
                         allowed_types: list[str],
                         max_new_tokens: int = 4096,
                         **kwargs) -> list[tuple[str, str, str]]:
    messages = self.engine.prompting.get_messages(passage, allowed_types=allowed_types)
    decoded = self.engine.chat_completion(messages=messages,
                                          max_new_tokens=max_new_tokens,
                                          **kwargs)
    return self.engine.prompting.format_triplets(decoded, allowed_types=allowed_types)
```

---

## 5️⃣ **Converted to LLM Prompt**

### **Location: `kg_builder/prompts/prompting.py`**

```python
# Lines 30-34
def get_messages(self, passage: str, allowed_types: list[str]) -> list[dict[str, str]]:
    return [{"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{self.user_instruction} {passage}\n"
                                        f"{self.allowed_types_prompt} {process_types(allowed_types)}\n"}]
```

**The `process_types()` function formats the list:**

```python
# Lines 135-137
def process_types(types: list[str]) -> str:
    return ', '.join([f"'{t.upper()}'" for t in types])

# Example:
# ["PERSON", "EVENT"] → "'PERSON', 'EVENT'"
```

**This creates a prompt like:**
```
Only extract entities of the following types: 'PERSON', 'EVENT', 'LOCATION', 'DATE', 'EMOTION', 'OBJECT', 'RELATIONSHIP'
```

---

## 6️⃣ **Validation After LLM Response**

### **Location: `kg_builder/prompts/prompting.py`**

```python
# Lines 36-58
@staticmethod
def format_triplets(extracted_relations: str, allowed_types: list[str]) -> list[tuple[str, str, str]]:
    pattern = r'\(([^:]+):([^,]+),\s*([^,]+),\s*([^:]+):([^)]+)\)'
    triplets = []
    
    for r in re.finditer(pattern, extracted_relations):
        entity1, type1, relation, entity2, type2 = r.groups()
        type1, type2 = type1.strip().upper(), type2.strip().upper()
        
        # VALIDATION: Check if types are allowed
        if type1 not in allowed_types:
            warnings.warn(f"Entity type {type1} not in allowed types {allowed_types}")
            continue
        elif type2 not in allowed_types:
            warnings.warn(f"Entity type {type2} not in allowed types {allowed_types}")
            continue
        
        triplet = (f"{entity1.strip().lower()}:{type1}", 
                   relation.strip(), 
                   f"{entity2.strip().lower()}:{type2}")
        triplets.append(triplet)
    
    return triplets
```

**Validation Steps:**
1. **Extract** entity types from LLM response
2. **Convert** to uppercase
3. **Check** if type is in `allowed_types` list
4. **Reject** triplets with invalid types
5. **Warning** logged for rejected triplets

---

## 7️⃣ **Used in Build Phase Validation**

### **Location: `kg_builder/builder.py`**

```python
# Lines 20-22
def decision_from_messages(self, messages, max_new_tokens=1024, **kwargs):
    decoded = self.engine.chat_completion(messages=messages, max_new_tokens=max_new_tokens, **kwargs)
    allowed_types = self.relations_data.relations_data.allowed_entity_types
    decision = self.engine.prompting.extract_new_triplet(decoded, allowed_entity_types=allowed_types)
```

The Builder phase **re-validates** entity types to ensure consistency.

---

## 🎛️ **How to Customize Entity Types**

### **Option 1: Modify in Example Files**

```python
# In example_usage.py
entity_types = ["YOUR", "CUSTOM", "TYPES"]
```

### **Option 2: Create Custom Script**

```python
from kg_builder.relations import RelationsData
from kg_builder.workflow import EBWorkflow

# Define your domain-specific types
entity_types = ["GENE", "PROTEIN", "DISEASE", "PATHWAY"]

# Create knowledge graph
relations_data = RelationsData.empty(allowed_entity_types=entity_types)

# Use in workflow
result = workflow(text, relations_data, builder_engine, extractor_engine)
```

### **Option 3: Load from Configuration**

```python
import json

# Load from config file
with open('entity_config.json', 'r') as f:
    config = json.load(f)
    entity_types = config['entity_types']

relations_data = RelationsData.empty(allowed_entity_types=entity_types)
```

---

## 📊 **Entity Type Selection Guidelines**

### **1. Domain Relevance**
Choose types that are **meaningful** in your domain:
```python
# Good for social media
["PERSON", "POST", "HASHTAG", "PLATFORM"]

# Bad - too generic
["THING", "STUFF", "ITEM"]
```

### **2. Appropriate Granularity**
```python
# Good granularity
["PERSON", "ORGANIZATION", "LOCATION"]

# Too specific
["MALE_PERSON", "FEMALE_PERSON", "CHILD_PERSON"]

# Too broad  
["ENTITY"]
```

### **3. Consistent Naming**
```python
# Good - consistent pattern
["PERSON", "ORGANIZATION", "LOCATION", "EVENT"]

# Bad - inconsistent
["person", "ORG", "Place", "activity"]
```

### **4. Balanced Coverage**
Include types that cover your main concepts:
```python
# Academic domain - balanced coverage
["RESEARCHER", "INSTITUTION", "PAPER", "FIELD", "CONFERENCE", "JOURNAL"]
```

---

## 🔍 **Real Example from Our Run**

From our image metadata extraction:

**Input Entity Types:**
```python
["PERSON", "EVENT", "LOCATION", "DATE", "EMOTION", "OBJECT", "RELATIONSHIP"]
```

**Final Extracted Distribution:**
```json
{
  "PERSON": 9 entities,     # alex, sam, mike, sister, parents, etc.
  "EVENT": 7 entities,      # graduation, wedding, birthday, etc. 
  "LOCATION": 8 entities    # café, university, park, shelter, etc.
}
```

**Unused Types:** `DATE`, `EMOTION`, `OBJECT`, `RELATIONSHIP` - These were defined but no triplets were extracted with these types, which is normal.

---

## ⚙️ **Technical Implementation Notes**

### **Case Sensitivity**
- Input: Case-insensitive (`"person"`, `"Person"`, `"PERSON"`)
- Storage: Always UPPERCASE (`"PERSON"`)
- Validation: UPPERCASE comparison

### **Type Normalization**
```python
# Input
entity_types = [" person ", "Event", "LOCATION "]

# After processing
self.allowed_entity_types = ["PERSON", "EVENT", "LOCATION"]
```

### **Error Handling**
- Invalid types trigger **warnings**, not errors
- Triplets with invalid types are **silently dropped**
- System continues processing valid triplets

This comprehensive flow shows that **entity types are user-defined** at the beginning and flow through the entire system, controlling what gets extracted and validated at every step! 🎯
