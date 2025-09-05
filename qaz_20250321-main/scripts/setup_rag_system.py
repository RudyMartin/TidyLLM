#!/usr/bin/env python3
"""
Setup script for Domain Knowledge RAG System
Install dependencies and initialize the system
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("🔧 Installing RAG system requirements...")
    
    try:
        # Install core requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_rag.txt"
        ])
        print("✅ Core requirements installed successfully")
        
        # Install spaCy language model
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ])
        print("✅ spaCy language model installed")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    
    return True

def check_dependencies():
    """Check if all dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'chromadb',
        'sentence_transformers', 
        'networkx',
        'spacy',
        'numpy',
        'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies available")
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "domain_knowledge_db",
        "rag_outputs",
        "knowledge_graphs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created: {directory}")

def test_imports():
    """Test if the RAG system can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Test basic imports
        import chromadb
        from sentence_transformers import SentenceTransformer
        import networkx as nx
        import spacy
        
        print("✅ Basic imports successful")
        
        # Test embedding model
        model = SentenceTransformer("all-MiniLM-L6-v2")
        test_embedding = model.encode("Test sentence")
        print(f"✅ Embedding model working (dimension: {len(test_embedding)})")
        
        # Test spaCy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Test sentence")
        print(f"✅ spaCy working (entities: {len(doc.ents)})")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Domain Knowledge RAG System")
    print("=" * 50)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        return False
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        return False
    
    # Step 4: Test imports
    if not test_imports():
        print("❌ Import test failed")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 RAG System Setup Complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run: python domain_knowledge_rag_implementation.py")
    print("2. Or import the system in your own scripts")
    print("3. Query your domain knowledge with natural language!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
