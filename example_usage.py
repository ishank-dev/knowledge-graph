#!/usr/bin/env python3
"""
Example usage of the Knowledge Graph Extraction system.

This script demonstrates how to use the EBWorkflow to extract knowledge graphs
from text using the two-phase Extract-Build approach.
"""

import os
import json
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import wikipedia

from kg_builder.workflow import EBWorkflow
from kg_builder.engine import GeminiEngine
from kg_builder.prompts.prompting import ExtractorPrompting, BuilderPrompting
from kg_builder.relations import RelationsData
from kg_builder.cfg import SplitConfig


def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    
    # Check if API key is set
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError(
            "Please set your GEMINI_API_KEY in the .env file. "
            "You can get one from https://ai.google.dev/"
        )
    
    return {
        "model": os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-exp"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "4096")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "128")),
        "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "4096")),
        "temperature": float(os.getenv("TEMPERATURE", "0.0"))
    }


def get_wikipedia_content(page_titles: list[str]) -> str:
    """Fetch content from Wikipedia pages."""
    print(f"Fetching Wikipedia content for: {', '.join(page_titles)}")
    
    content = []
    for title in page_titles:
        try:
            page = wikipedia.page(title)
            content.append(f"=== {page.title} ===\n{page.summary}\n")
            print(f"✓ Fetched: {page.title}")
        except wikipedia.exceptions.DisambiguationError as e:
            # Try the first option if disambiguation
            try:
                page = wikipedia.page(e.options[0])
                content.append(f"=== {page.title} ===\n{page.summary}\n")
                print(f"✓ Fetched (disambiguated): {page.title}")
            except:
                print(f"✗ Failed to fetch: {title}")
        except Exception as e:
            print(f"✗ Error fetching {title}: {e}")
    
    return "\n".join(content)


def load_image_metadata(metadata_file: str = "sample_image_metadata.json") -> str:
    """Load and process image metadata to create text for knowledge graph extraction."""
    print(f"📸 Loading image metadata from: {metadata_file}")
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        text_content = []
        
        # Process each image's metadata
        for img_data in data["image_metadata"]:
            img_id = img_data["image_id"]
            description = img_data["description"]
            text_desc = img_data["text_description"]
            
            # Create a structured text entry for this image
            img_text = f"=== Image {img_id}: {description} ===\n{text_desc}\n"
            text_content.append(img_text)
            
            print(f"✓ Processed: {img_id} - {description}")
        
        # Add profile information if available
        if "additional_metadata" in data and "person_profile" in data["additional_metadata"]:
            profile = data["additional_metadata"]["person_profile"]
            profile_text = f"""=== Person Profile ===
{profile['name']} is {profile['age']} years old. Key relationships include {', '.join(profile['key_relationships'])}. 
Favorite places are {', '.join(profile['favorite_places'])}. 
Memorable events include {', '.join(profile['memorable_events'])}.
"""
            text_content.append(profile_text)
            print(f"✓ Added profile information for {profile['name']}")
        
        combined_text = "\n".join(text_content)
        print(f"📄 Generated {len(combined_text)} characters of text from {len(data['image_metadata'])} images")
        
        return combined_text
        
    except FileNotFoundError:
        print(f"❌ Metadata file {metadata_file} not found")
        return ""
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return ""
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return ""


def run_image_metadata_example():
    """Run knowledge graph extraction from image metadata."""
    print("🚀 Starting Knowledge Graph Extraction from Image Metadata")
    print("=" * 60)
    
    # Load configuration
    config = load_environment()
    print(f"Using model: {config['model']}")
    print(f"Using embeddings: {config['embedding_model']}")
    
    # Setup embeddings
    print("\n📊 Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name=config["embedding_model"])
    
    # Setup workflow
    split_config = SplitConfig(
        chunk_char_size=config["chunk_size"],
        chunk_char_overlap=config["chunk_overlap"]
    )
    workflow = EBWorkflow(
        embeddings=embeddings,
        split_cfg=split_config,
        extractor_prompting=ExtractorPrompting(),
        builder_prompting=BuilderPrompting()
    )
    
    # Setup engines
    print("\n🤖 Initializing LLM engines...")
    builder_engine = GeminiEngine(config["model"], BuilderPrompting())
    extractor_engine = GeminiEngine(config["model"], ExtractorPrompting())
    
    # Load image metadata instead of Wikipedia
    text = load_image_metadata("sample_image_metadata.json")
    
    if not text.strip():
        print("❌ No metadata loaded. Check the metadata file.")
        return None
    
    # Define entity types relevant to personal image metadata
    entity_types = ["PERSON", "EVENT", "LOCATION", "DATE", "EMOTION", "OBJECT", "RELATIONSHIP"]
    
    # Initialize empty knowledge graph
    print(f"\n🔧 Setting up knowledge graph with entity types: {entity_types}")
    relations_data = RelationsData.empty(allowed_entity_types=entity_types)
    
    # Run extraction
    print("\n🔍 Starting knowledge graph extraction from image metadata...")
    print("This may take a few minutes depending on content complexity and API response times.")
    
    try:
        result = workflow(
            text=text,
            relations_data=relations_data,
            builder_engine=builder_engine,
            extractor_engine=extractor_engine,
            max_new_tokens=config["max_new_tokens"],
            temperature=config["temperature"]
        )
        
        print("\n✅ Extraction completed successfully!")
        
        # Print results
        print(f"\n📈 Results Summary:")
        print(f"   • Total relations extracted: {len(result.flattened_triplets)}")
        print(f"   • Number of source passages: {len(result.annotated_passages)}")
        
        print(f"\n📝 Extracted Relations from Image Metadata:")
        for i, triplet in enumerate(result.flattened_triplets):
            print(f"   {i+1}. {triplet}")
        
        # Create visualization
        print(f"\n🎨 Creating knowledge graph visualization...")
        try:
            result.plot_graph(figsize=(14, 10), save_path="image_metadata_knowledge_graph.png")
            print("   ✓ Graph saved as 'image_metadata_knowledge_graph.png'")
        except Exception as e:
            print(f"   ⚠️ Visualization failed: {e}")
        
        # Save results
        print(f"\n💾 Saving results...")
        try:
            result.save_json("image_metadata_knowledge_graph.json")
            print("   ✓ Results saved as 'image_metadata_knowledge_graph.json'")
        except Exception as e:
            print(f"   ⚠️ Save failed: {e}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        print("   Check your API key and internet connection.")
        raise


def run_basic_example():
    """Run a basic example with predefined Wikipedia articles."""
    print("🚀 Starting Knowledge Graph Extraction Example")
    print("=" * 50)
    
    # Load configuration
    config = load_environment()
    print(f"Using model: {config['model']}")
    print(f"Using embeddings: {config['embedding_model']}")
    
    # Setup embeddings
    print("\n📊 Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name=config["embedding_model"])
    
    # Setup workflow
    split_config = SplitConfig(
        chunk_char_size=config["chunk_size"],
        chunk_char_overlap=config["chunk_overlap"]
    )
    workflow = EBWorkflow(
        embeddings=embeddings,
        split_cfg=split_config,
        extractor_prompting=ExtractorPrompting(),
        builder_prompting=BuilderPrompting()
    )
    
    # Setup engines
    print("\n🤖 Initializing LLM engines...")
    builder_engine = GeminiEngine(config["model"], BuilderPrompting())
    extractor_engine = GeminiEngine(config["model"], ExtractorPrompting())
    
    # Get sample text from Wikipedia
    wiki_pages = ["Facebook", "Instagram", "WhatsApp"]
    text = get_wikipedia_content(wiki_pages)
    
    if not text.strip():
        print("❌ No content retrieved. Using fallback text.")
        text = """
        Facebook is a social media platform founded by Mark Zuckerberg in 2004. 
        Instagram is a photo-sharing app acquired by Facebook in 2012. 
        WhatsApp is a messaging app also acquired by Facebook in 2014.
        """
    
    # Define entity types to extract
    entity_types = ["PERSON", "COMPANY", "APPLICATION"]
    
    # Initialize empty knowledge graph
    print(f"\n🔧 Setting up knowledge graph with entity types: {entity_types}")
    relations_data = RelationsData.empty(allowed_entity_types=entity_types)
    
    # Run extraction
    print("\n🔍 Starting knowledge graph extraction...")
    print("This may take a few minutes depending on text length and API response times.")
    
    try:
        result = workflow(
            text=text,
            relations_data=relations_data,
            builder_engine=builder_engine,
            extractor_engine=extractor_engine,
            max_new_tokens=config["max_new_tokens"],
            temperature=config["temperature"]
        )
        
        print("\n✅ Extraction completed successfully!")
        
        # Print results
        print(f"\n📈 Results Summary:")
        print(f"   • Total relations extracted: {len(result.flattened_triplets)}")
        print(f"   • Number of source passages: {len(result.annotated_passages)}")
        
        print(f"\n📝 Sample Relations:")
        for i, triplet in enumerate(result.flattened_triplets[:10]):  # Show first 10
            print(f"   {i+1}. {triplet}")
        
        if len(result.flattened_triplets) > 10:
            print(f"   ... and {len(result.flattened_triplets) - 10} more relations")
        
        # Create visualization
        print(f"\n🎨 Creating knowledge graph visualization...")
        try:
            result.plot_graph(figsize=(12, 8), save_path="knowledge_graph.png")
            print("   ✓ Graph saved as 'knowledge_graph.png'")
        except Exception as e:
            print(f"   ⚠️ Visualization failed: {e}")
        
        # Save results
        print(f"\n💾 Saving results...")
        try:
            result.save_json("extracted_knowledge_graph.json")
            print("   ✓ Results saved as 'extracted_knowledge_graph.json'")
        except Exception as e:
            print(f"   ⚠️ Save failed: {e}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        print("   Check your API key and internet connection.")
        raise


def run_custom_example(text: str, entity_types: list[str]):
    """Run extraction with custom text and entity types."""
    print("🚀 Starting Custom Knowledge Graph Extraction")
    print("=" * 50)
    
    config = load_environment()
    
    # Setup
    embeddings = HuggingFaceEmbeddings(model_name=config["embedding_model"])
    workflow = EBWorkflow(embeddings=embeddings)
    
    builder_engine = GeminiEngine(config["model"], BuilderPrompting())
    extractor_engine = GeminiEngine(config["model"], ExtractorPrompting())
    
    relations_data = RelationsData.empty(allowed_entity_types=entity_types)
    
    # Run extraction
    result = workflow(
        text=text,
        relations_data=relations_data,
        builder_engine=builder_engine,
        extractor_engine=extractor_engine
    )
    
    return result


if __name__ == "__main__":
    try:
        # Ask user which example to run
        print("📋 Choose extraction method:")
        print("1. Image Metadata (recommended)")
        print("2. Wikipedia Articles")
        
        choice = input("\nEnter choice (1 or 2, default=1): ").strip()
        
        if choice == "2":
            # Run the Wikipedia example
            result = run_basic_example()
            output_files = ["knowledge_graph.png", "extracted_knowledge_graph.json"]
        else:
            # Run the image metadata example (default)
            result = run_image_metadata_example()
            output_files = ["image_metadata_knowledge_graph.png", "image_metadata_knowledge_graph.json"]
        
        print(f"\n🎉 Knowledge Graph Extraction completed successfully!")
        print(f"Check the generated files:")
        for file in output_files:
            print(f"   • {file}")
        
    except KeyboardInterrupt:
        print("\n\n👋 Extraction cancelled by user")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Make sure you've set GEMINI_API_KEY in .env file")
        print(f"2. Check your internet connection")
        print(f"3. Verify you have all dependencies installed: pip install -r requirements.txt")
