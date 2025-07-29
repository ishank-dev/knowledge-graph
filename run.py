#!/usr/bin/env python3
"""
Quick start script for Knowledge Graph Extraction.
"""

import os
import sys


def check_requirements():
    """Check if required packages are installed."""
    missing_packages = []
    
    try:
        import torch
    except ImportError:
        missing_packages.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing_packages.append("transformers")
    
    try:
        from google import genai
    except ImportError:
        missing_packages.append("google-genai")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        missing_packages.append("langchain-huggingface")
    
    try:
        from dotenv import load_dotenv
    except ImportError:
        missing_packages.append("python-dotenv")
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   • {pkg}")
        print("\n📦 Install them with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True


def check_api_key():
    """Check if API key is configured."""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("❌ GEMINI_API_KEY not configured")
        print("\n🔑 To set up your API key:")
        print("1. Get a free API key from: https://ai.google.dev/")
        print("2. Edit the .env file and replace 'your_gemini_api_key_here' with your actual key")
        return False
    
    print("✅ API key configured")
    return True


def main():
    """Main startup function."""
    print("🚀 Knowledge Graph Extraction - Quick Start")
    print("=" * 50)
    
    # Check requirements
    print("\n📋 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    
    print("✅ All packages installed")
    
    # Check API key
    print("\n🔑 Checking API configuration...")
    if not check_api_key():
        sys.exit(1)
    
    # Run example
    print("\n🎯 Starting knowledge graph extraction from image metadata...")
    try:
        from example_usage import run_image_metadata_example
        result = run_image_metadata_example()
        print("\n🎉 Success! Check the generated files:")
        print("   • image_metadata_knowledge_graph.png (visualization)")
        print("   • image_metadata_knowledge_graph.json (data)")
        
    except Exception as e:
        print(f"\n💥 Error: {e}")
        print("\n🔍 Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify your API key is valid") 
        print("3. Try running 'python example_usage.py' for more options")


if __name__ == "__main__":
    main()
