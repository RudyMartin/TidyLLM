#!/usr/bin/env python3
"""
Install py311 Packages Script

This script helps install the required packages in the py311 conda environment.
It handles the upgrade of sentence-transformers and other dependencies.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main installation function"""
    print("py311 Package Installation Script")
    print("=" * 50)
    
    # Check if we're in the right environment
    python_path = sys.executable
    print(f"Python path: {python_path}")
    
    if "py311" not in python_path:
        print("⚠️  Warning: Not running in py311 environment")
        print("Please activate the py311 environment first:")
        print("conda activate py311")
        return False
    
    # Step 1: Upgrade pip
    run_command(f"{python_path} -m pip install --upgrade pip", "Upgrading pip")
    
    # Step 2: Upgrade sentence-transformers (critical for embedding system)
    run_command(f"{python_path} -m pip install 'sentence-transformers>=5.1.0,<6.0.0'", 
               "Upgrading sentence-transformers to 5.1.0+")
    
    # Step 3: Install core requirements
    run_command(f"{python_path} -m pip install -r py311_requirements.txt", 
               "Installing py311 requirements")
    
    # Step 4: Verify key packages
    print("\nVerifying key packages...")
    key_packages = [
        "sentence_transformers",
        "torch", 
        "transformers",
        "numpy",
        "pandas",
        "streamlit",
        "mlflow"
    ]
    
    for package in key_packages:
        try:
            __import__(package)
            print(f"✅ {package} imported successfully")
        except ImportError as e:
            print(f"❌ {package} import failed: {e}")
    
    # Step 5: Test embedding system
    print("\nTesting enhanced embedding system...")
    try:
        from backend.core.embedding_helper import EmbeddingHelper
        helper = EmbeddingHelper(target_dimensions=1024)
        model_info = helper.get_model_info()
        print(f"✅ Embedding helper initialized")
        print(f"   Model: {model_info['model_metadata']['model_name']}")
        print(f"   Dimensions: {model_info['model_metadata']['original_dimensions']} -> {model_info['model_metadata']['target_dimensions']}")
    except Exception as e:
        print(f"❌ Embedding system test failed: {e}")
    
    print("\n" + "=" * 50)
    print("Installation script completed!")
    print("\nNext steps:")
    print("1. Test the enhanced embedding system: python test_enhanced_embeddings.py")
    print("2. Run the PDF RAG test: python simple_pdf_rag_test.py")
    print("3. Start the Streamlit app: python start_simple.py")
    
    return True

if __name__ == "__main__":
    main()
