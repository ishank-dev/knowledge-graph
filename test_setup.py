#!/usr/bin/env python3
"""
Test script to verify the knowledge graph extraction setup.
"""

def test_basic_imports():
    """Test if basic imports work."""
    print("🧪 Testing basic imports...")
    
    try:
        from kg_builder.workflow import EBWorkflow
        from kg_builder.engine import GeminiEngine
        from kg_builder.relations import RelationsData
        from kg_builder.prompts.prompting import ExtractorPrompting, BuilderPrompting
        print("✅ All kg_builder imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_dependencies():
    """Test if external dependencies work."""
    print("\n🧪 Testing external dependencies...")
    
    missing = []
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        missing.append("transformers")
    
    try:
        from google import genai
        print("✅ Google GenAI available")
    except ImportError:
        missing.append("google-genai")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        print("✅ LangChain HuggingFace available")
    except ImportError:
        missing.append("langchain-huggingface")
    
    try:
        import networkx as nx
        print(f"✅ NetworkX: {nx.__version__}")
    except ImportError:
        missing.append("networkx")
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        missing.append("matplotlib")
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True


def test_configuration():
    """Test configuration setup."""
    print("\n🧪 Testing configuration...")
    
    try:
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            print("⚠️  No GEMINI_API_KEY found in .env file")
            return False
        elif api_key == "your_gemini_api_key_here":
            print("⚠️  Please replace placeholder API key in .env file")
            return False
        else:
            print("✅ API key configured")
            return True
            
    except ImportError:
        print("❌ python-dotenv not installed")
        return False


def test_simple_workflow():
    """Test if workflow can be instantiated."""
    print("\n🧪 Testing workflow instantiation...")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from kg_builder.workflow import EBWorkflow
        from kg_builder.relations import RelationsData
        
        # This will download the model if not cached
        print("   📥 Loading embeddings model (may take a moment)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        workflow = EBWorkflow(embeddings=embeddings)
        relations_data = RelationsData.empty(["PERSON", "COMPANY"])
        
        print("✅ Workflow instantiation successful")
        return True
        
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Knowledge Graph Extraction - System Test")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_dependencies,
        test_configuration,
        test_simple_workflow
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("🏁 Test Summary:")
    
    success_count = sum(results)
    total_count = len(results)
    
    if success_count == total_count:
        print(f"✅ All {total_count} tests passed!")
        print("\n🎉 System is ready!")
        print("Run 'python run.py' to start extraction.")
    else:
        print(f"❌ {total_count - success_count} of {total_count} tests failed")
        print("\n🔧 Please fix the issues above before running the system.")
        
    print("\n📚 Quick start guide:")
    print("1. Make sure all tests pass")
    print("2. Add your Gemini API key to .env file")
    print("3. Run: python run.py")


if __name__ == "__main__":
    main()
