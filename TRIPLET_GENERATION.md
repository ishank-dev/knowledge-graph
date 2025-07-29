# 🔗 Triplet Generation Deep Dive

## 🎯 What are Knowledge Graph Triplets?

Triplets are the atomic units of knowledge representation in our system. They follow the **RDF (Resource Description Framework)** pattern:

```
Subject → Predicate → Object
   ↓         ↓         ↓
Entity → Relation → Entity
   ↓         ↓         ↓
(alex:PERSON, graduated from, columbia university:LOCATION)
```

## 📋 Triplet Anatomy

### Structure
```
(subject:entity_type, relation, object:entity_type)
```

### Components
1. **Subject**: The entity performing the action or having the property
2. **Relation**: The relationship/predicate connecting subject and object  
3. **Object**: The entity being acted upon or related to
4. **Entity Types**: Semantic categories (PERSON, LOCATION, EVENT, etc.)

### Real Examples from Our System
```json
[
  ["alex:PERSON", "favorite place", "bluebird café:LOCATION"],
  ["sam:PERSON", "organized", "28th birthday:EVENT"],
  ["alex:PERSON", "volunteers at", "local animal shelter:LOCATION"],
  ["alex:PERSON", "key relationship", "professor smith:PERSON"]
]
```

---

## 🤖 How Triplets are Generated

### Step-by-Step Process

#### 1. **Input Processing**
```python
# Original image metadata text
text = """
Alex graduated from Columbia University in New York City. 
Professor Smith handed him his graduation diploma during the ceremony.
Alex wore his cap and gown with pride.
"""
```

#### 2. **LLM Instruction**
The system sends this prompt to the LLM:

```
User: Extract relations from this text. Allowed entity types: ['PERSON', 'LOCATION', 'EVENT', 'OBJECT']

Text: Alex graduated from Columbia University in New York City. Professor Smith handed him his graduation diploma during the ceremony. Alex wore his cap and gown with pride.
```

#### 3. **LLM Response**
```
(alex:PERSON, graduated from, columbia university:LOCATION)
(professor smith:PERSON, handed, graduation diploma:OBJECT)
(alex:PERSON, received, graduation diploma:OBJECT)
(alex:PERSON, attended, graduation ceremony:EVENT)
(alex:PERSON, wore, cap and gown:OBJECT)
```

#### 4. **Parsing with Regex**
```python
import re

# Regex pattern to extract triplets
pattern = r'\(([^:]+):([^,]+),\s*([^,]+),\s*([^:]+):([^)]+)\)'

# Extract and validate each triplet
for match in re.finditer(pattern, llm_response):
    subject, subject_type, relation, object_name, object_type = match.groups()
    
    # Validate entity types are allowed
    if subject_type in allowed_types and object_type in allowed_types:
        triplet = (f"{subject}:{subject_type}", relation, f"{object_name}:{object_type}")
        validated_triplets.append(triplet)
```

#### 5. **Validation & Cleaning**
```python
# Remove malformed triplets
valid_triplets = []
for triplet in raw_triplets:
    if len(triplet) == 3 and all(part.strip() for part in triplet):
        valid_triplets.append(triplet)
```

---

## 🔍 Triplet Generation Examples

### Example 1: Personal Relationship
**Input:** "Alex's best friend Sam organized his 28th birthday party"

**Entity Types:** `["PERSON", "EVENT", "RELATIONSHIP"]`

**Generated Triplets:**
```python
[
  ("alex:PERSON", "best friend", "sam:PERSON"),
  ("sam:PERSON", "organized", "28th birthday party:EVENT"),
  ("alex:PERSON", "celebrated", "28th birthday party:EVENT")
]
```

### Example 2: Location & Activity
**Input:** "Alex volunteers at the local animal shelter where Jane is the coordinator"

**Entity Types:** `["PERSON", "LOCATION", "ROLE"]`

**Generated Triplets:**
```python
[
  ("alex:PERSON", "volunteers at", "local animal shelter:LOCATION"),
  ("jane:PERSON", "coordinator of", "local animal shelter:LOCATION"),
  ("alex:PERSON", "works with", "jane:PERSON")
]
```

### Example 3: Event & Emotion
**Input:** "The family trip to Cappadocia filled everyone with excitement and wonder"

**Entity Types:** `["PERSON", "EVENT", "LOCATION", "EMOTION"]`

**Generated Triplets:**
```python
[
  ("family:PERSON", "went on", "trip to cappadocia:EVENT"),
  ("trip to cappadocia:EVENT", "took place in", "cappadocia:LOCATION"),
  ("family:PERSON", "felt", "excitement:EMOTION"),
  ("family:PERSON", "felt", "wonder:EMOTION"),
  ("trip to cappadocia:EVENT", "evoked", "excitement:EMOTION")
]
```

---

## 🎛️ Controlling Triplet Generation

### 1. **Entity Type Constraints**
```python
# Medical domain
allowed_types = ["PATIENT", "SYMPTOM", "DISEASE", "TREATMENT"]

# Business domain  
allowed_types = ["COMPANY", "PRODUCT", "MARKET", "REVENUE"]

# Academic domain
allowed_types = ["RESEARCHER", "PAPER", "INSTITUTION", "FIELD"]
```

### 2. **Relation Type Guidance**
You can guide the LLM to focus on specific relationship types:

```python
# Temporal relationships
prompt_addition = "Focus on temporal relationships like 'before', 'after', 'during'"

# Causal relationships
prompt_addition = "Extract causal relationships like 'caused by', 'led to', 'resulted in'"

# Hierarchical relationships  
prompt_addition = "Identify hierarchical relationships like 'part of', 'contains', 'belongs to'"
```

### 3. **Granularity Control**
```python
# High-level relationships
"Extract only major relationships, ignore minor details"

# Fine-grained extraction
"Extract all possible relationships, including subtle connections"
```

---

## 🔧 Technical Implementation

### Core Extraction Function
```python
def extract_from_passage(self, passage: str, allowed_types: list[str]) -> list[tuple[str, str, str]]:
    # 1. Generate prompt with passage and allowed types
    messages = self.engine.prompting.get_messages(passage, allowed_types=allowed_types)
    
    # 2. Get LLM response
    decoded = self.engine.chat_completion(messages=messages, max_new_tokens=4096)
    
    # 3. Parse and validate triplets
    return self.engine.prompting.format_triplets(decoded, allowed_types=allowed_types)
```

### Triplet Parsing Logic
```python
def format_triplets(self, text: str, allowed_entity_types: list[str]) -> list[tuple[str, str, str]]:
    # Regex to match triplet format
    pattern = r'\(([^:]+):([^,]+),\s*([^,]+),\s*([^:]+):([^)]+)\)'
    
    triplets = []
    for match in re.finditer(pattern, text):
        subject, subj_type, relation, obj_name, obj_type = match.groups()
        
        # Clean and validate
        subj_type = subj_type.strip().upper()
        obj_type = obj_type.strip().upper()
        
        if subj_type in allowed_entity_types and obj_type in allowed_entity_types:
            triplet = (
                f"{subject.strip().lower()}:{subj_type}",
                relation.strip(),
                f"{obj_name.strip().lower()}:{obj_type}"
            )
            triplets.append(triplet)
    
    return triplets
```

---

## 📊 Quality Control in Triplet Generation

### 1. **Format Validation**
```python
def validate_triplet_format(triplet):
    """Ensure triplet has exactly 3 parts with proper entity type format"""
    if len(triplet) != 3:
        return False
    
    subject, relation, obj = triplet
    
    # Check entity format: "entity:TYPE"
    if ':' not in subject or ':' not in obj:
        return False
    
    # Check no empty components
    if not all(part.strip() for part in triplet):
        return False
    
    return True
```

### 2. **Entity Type Validation**
```python
def validate_entity_types(triplet, allowed_types):
    """Ensure entity types are in allowed list"""
    subject, relation, obj = triplet
    
    subj_type = subject.split(':')[1].upper()
    obj_type = obj.split(':')[1].upper()
    
    return subj_type in allowed_types and obj_type in allowed_types
```

### 3. **Relationship Quality**
```python
def validate_relationship_quality(triplet):
    """Check for meaningful relationships"""
    subject, relation, obj = triplet
    
    # Avoid reflexive relationships (entity relating to itself)
    subj_entity = subject.split(':')[0]
    obj_entity = obj.split(':')[0]
    
    if subj_entity == obj_entity:
        return False
    
    # Avoid very generic relations
    generic_relations = ["has", "is", "does", "exists"]
    if relation.lower() in generic_relations:
        return False
    
    return True
```

---

## 📈 Performance & Statistics

### From Our Image Metadata Run:

**Input Statistics:**
- 7 images processed
- 2,874 characters of text
- 8 entity types allowed

**Output Statistics:**
- 24 triplets generated
- 100% triplet format validity
- 9 unique person entities
- 5 event types
- 5 location types

**Processing Time:**
- Extract phase: 4.5 seconds
- Build phase: 131 seconds
- Total: ~2.3 minutes

**Triplet Examples:**
```python
# Personal relationships
("alex:PERSON", "key relationship", "sam:PERSON")
("alex:PERSON", "key relationship", "sister:PERSON")

# Activities
("alex:PERSON", "volunteers at", "local animal shelter:LOCATION")
("alex:PERSON", "went on", "hiking trip:EVENT")

# Preferences  
("alex:PERSON", "favorite place", "bluebird café:LOCATION")
("alex:PERSON", "favorite place", "mountain trails:LOCATION")

# Events
("sam:PERSON", "organized", "28th birthday:EVENT")
("alex:PERSON", "graduated from", "columbia university:LOCATION")
```

---

## 🎯 Best Practices for Triplet Generation

### 1. **Entity Type Design**
- **Specific enough**: "PERSON" vs "STUDENT", "PROFESSOR", "EMPLOYEE"
- **Not too granular**: Avoid "MALE_PERSON", "FEMALE_PERSON"
- **Domain relevant**: Include types specific to your use case

### 2. **Relation Naming**
- **Consistent terminology**: "graduated from" vs "attended" vs "studied at"
- **Directional clarity**: Subject performs action on object
- **Meaningful verbs**: Avoid generic "related to", "associated with"

### 3. **Entity Naming**
- **Normalize case**: "alex" vs "Alex" vs "ALEX"
- **Handle variations**: "Mike" vs "Michael", "NYC" vs "New York City"  
- **Consistent format**: "bluebird café" vs "Bluebird Cafe"

### 4. **Quality Filtering**
- **Remove duplicates**: Same information expressed differently
- **Filter noise**: Overly generic or meaningless relations
- **Validate completeness**: Ensure all parts are filled

This triplet generation system provides the foundation for building rich, structured knowledge graphs from any text source! 🚀
