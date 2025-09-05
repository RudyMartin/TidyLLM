#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Base Organization Script
==================================

This script organizes input assets into a structured knowledge base
for Model Risk Management and AI/ML applications.

Usage:
    python scripts/organize_knowledge_base.py [--create-structure] [--move-files] [--analyze-only]
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeBaseOrganizer:
    """Organize input assets into structured knowledge base"""
    
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.input_dir = self.project_root / "input"
        self.knowledge_base_dir = self.project_root / "knowledge_base"
        self.training_materials_dir = self.project_root / "training_materials"
        self.archive_dir = self.project_root / "archive"
        
        # Define file categorization
        self.setup_categorization()
        
    def setup_categorization(self):
        """Define how files should be categorized"""
        
        # Priority 1: Model Risk Management & Financial Services
        self.model_risk_files = {
            # SSRN Papers (Financial/ML Research) - from omnibus/
            'omnibus/ssrn_id4794069_code5285150_CNNLSTM.pdf': 'model_risk_management/financial_modeling/cnn_lstm_financial_models/',
            'omnibus/ssrn_id4794069_code5285150_ref1.pdf': 'model_risk_management/financial_modeling/cnn_lstm_financial_models/',
            'omnibus/ssrn_id4794069_code5285150_nlpfinance.pdf': 'model_risk_management/financial_modeling/nlp_applications/',
            'omnibus/ssrn-4977043.pdf': 'model_risk_management/financial_modeling/research_papers/',
            'omnibus/SSRN-id2741701.pdf': 'model_risk_management/financial_modeling/research_papers/',
            'omnibus/SSRN-id2431022.pdf': 'model_risk_management/financial_modeling/research_papers/',
            'omnibus/SSRN-id2604248.pdf': 'model_risk_management/financial_modeling/research_papers/',
            'omnibus/SSRN-id1985860.pdf': 'model_risk_management/financial_modeling/research_papers/',
            'omnibus/SSRN-id2672090.pdf': 'model_risk_management/financial_modeling/research_papers/',
            
            # Risk & Analytics
            'omnibus/helper_functions.txt': 'model_risk_management/financial_modeling/risk_calculation_methods/',
            'omnibus/PredicveMaintenance-AutoEncoder.pdf': 'model_risk_management/financial_modeling/predictive_analytics/',
            'omnibus/_nlp_name_embeddings.pdf': 'model_risk_management/financial_modeling/nlp_applications/',
            
            # Business Intelligence
            'omnibus/Query GA4 BigQuery Data via ChatGPT.pdf': 'model_risk_management/business_intelligence/data_analytics/',
            'omnibus/smarter-attribution-with-google-analytics-150306130949-conversion-gate01.pdf': 'model_risk_management/business_intelligence/attribution_modeling/',
            
            # Root level files
            'ssrn_id4794069_code5285150_CNNLSTM.pdf': 'model_risk_management/financial_modeling/cnn_lstm_financial_models/',
            'sparse_dropout_1905.13678.pdf': 'model_risk_management/financial_modeling/research_papers/',
            'Query GA4 BigQuery Data via ChatGPT.pdf': 'model_risk_management/business_intelligence/data_analytics/',
        }
        
        # Priority 1: AI/ML Research & Implementation
        self.ai_ml_files = {
            # AI/ML Research Papers - from omnibus/
            'omnibus/sparse_dropout_1905.13678.pdf': 'ai_ml_research/deep_learning_techniques/',
            'omnibus/2310.03714v1.pdf': 'ai_ml_research/research_papers/',
            'omnibus/2205.14135v2_FLASH_ATTENTION.pdf': 'ai_ml_research/attention_mechanisms/',
            'omnibus/s43681-023-00289-2.pdf': 'ai_ml_research/research_papers/',
            'omnibus/Generative-AI-and-LLMs-for-Dummies.pdf': 'ai_ml_research/llm_guides/',
            'omnibus/sam2_2408.00714v1.pdf': 'ai_ml_research/research_papers/',
            'omnibus/linear_learners_2309.06979v2.pdf': 'ai_ml_research/deep_learning_techniques/',
            'omnibus/s41586-021-03583-3.pdf': 'ai_ml_research/research_papers/',
            'omnibus/transformative_ai_2306.02519.pdf': 'ai_ml_research/research_papers/',
            'omnibus/UnderstandingDeepLearning_08_05_23_C.pdf': 'ai_ml_research/deep_learning_guides/',
            
            # Network & Embedding Research
            'omnibus/WSDM18-Qiu-et-al-NetMF-network-embedding.pdf': 'ai_ml_research/network_embeddings/',
            'omnibus/v36i08.pdf': 'ai_ml_research/research_papers/',
            'omnibus/v45i03.pdf': 'ai_ml_research/research_papers/',
            'omnibus/v46c02.pdf': 'ai_ml_research/research_papers/',
            'omnibus/v57i12.pdf': 'ai_ml_research/research_papers/',
            
            # Customer Analytics
            'omnibus/_user08_jimp_custseg_revnov08.pdf': 'model_risk_management/business_intelligence/customer_analytics/',
            'omnibus/Pointillist-Customer-Journey-Roadmap-CX-Success-Financial-Services-eBook.pdf': 'model_risk_management/business_intelligence/customer_analytics/',
        }
        
        # Priority 2: Educational & Reference Materials
        self.training_files = {
            # Course Materials - from omnibus/
            'omnibus/XCS236 Syllabus.pdf': 'training_materials/course_materials/',
            'omnibus/XCS224W_Syllabus.pdf': 'training_materials/course_materials/',
            'omnibus/XDGT110_All_Slides.pdf': 'training_materials/course_materials/',
            'omnibus/DataAssignment_1.pdf': 'training_materials/course_materials/',
            
            # Technical References
            'omnibus/_Blockchain4Dummies.PDF': 'training_materials/reference_guides/',
            'omnibus/_Bancor Paper Wallet.pdf': 'training_materials/reference_guides/',
            'omnibus/s40854-016-0029-6.pdf': 'training_materials/reference_guides/',
            
            # Demo & Example Files
            'omnibus/Robot Presentation.pdf': 'training_materials/demo_examples/',
            'omnibus/Smart Fruit Ripeness System.pdf': 'training_materials/demo_examples/',
            'omnibus/WSCLIENTONLINETEMPLATE.pdf': 'training_materials/templates/',
            'omnibus/WSBPONLINETEMPLATE.pdf': 'training_materials/templates/',
            'omnibus/Readme Rag Demo.pdf': 'training_materials/demo_examples/',
        }
        
        # Priority 3: Low Relevance (Archive)
        self.archive_files = {
            # Entertainment & Media - from omnibus/
            'omnibus/PWC - Entertainment and Media Outlook 2023-2027 ES.pdf': 'archive/entertainment_media/',
            'omnibus/Winner.pdf': 'archive/entertainment_media/',
            'omnibus/Townsend-Lookbook-view-in-Chrome.pdf': 'archive/entertainment_media/',
            
            # Academic Papers (Non-Financial)
            'omnibus/Personality&Music_preprint.pdf': 'archive/academic_research/',
            'omnibus/When Equity Seems Unfair The Role of Justice and Enforceability in Temporary Team Coordination.pdf': 'archive/academic_research/',
            'omnibus/v28i05.pdf': 'archive/academic_research/',
            'omnibus/v23i10.pdf': 'archive/academic_research/',
            
            # Miscellaneous
            'omnibus/Rplots.pdf': 'archive/miscellaneous/',
            'omnibus/r_matchit.pdf': 'archive/miscellaneous/',
            'omnibus/r_code.pdf': 'archive/miscellaneous/',
            'omnibus/US Startup Outlook Report 2017.pdf': 'archive/miscellaneous/',
        }
        
    def create_directory_structure(self):
        """Create the knowledge base directory structure"""
        logger.info("🏗️ Creating knowledge base directory structure...")
        
        # Knowledge Base Structure
        kb_structure = [
            'model_risk_management/financial_modeling/cnn_lstm_financial_models',
            'model_risk_management/financial_modeling/risk_calculation_methods',
            'model_risk_management/financial_modeling/predictive_analytics',
            'model_risk_management/financial_modeling/nlp_applications',
            'model_risk_management/financial_modeling/research_papers',
            'model_risk_management/regulatory_compliance/model_validation',
            'model_risk_management/regulatory_compliance/risk_assessment',
            'model_risk_management/regulatory_compliance/governance_frameworks',
            'model_risk_management/business_intelligence/data_analytics',
            'model_risk_management/business_intelligence/customer_analytics',
            'model_risk_management/business_intelligence/attribution_modeling',
            'ai_ml_research/attention_mechanisms',
            'ai_ml_research/deep_learning_techniques',
            'ai_ml_research/nlp_applications',
            'ai_ml_research/uncertainty_quantification',
            'ai_ml_research/research_papers',
            'ai_ml_research/llm_guides',
            'ai_ml_research/deep_learning_guides',
            'ai_ml_research/network_embeddings',
        ]
        
        # Training Materials Structure
        training_structure = [
            'course_materials',
            'reference_guides',
            'demo_examples',
            'templates',
            'code_examples',
        ]
        
        # Archive Structure
        archive_structure = [
            'entertainment_media',
            'academic_research',
            'code_repositories',
            'miscellaneous',
        ]
        
        # Create directories
        for structure, base_dir in [
            (kb_structure, self.knowledge_base_dir),
            (training_structure, self.training_materials_dir),
            (archive_structure, self.archive_dir)
        ]:
            for path in structure:
                full_path = base_dir / path
                full_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {full_path}")
                
        logger.info("✅ Directory structure created")
        
    def analyze_files(self):
        """Analyze files in input directory and categorize them"""
        logger.info("📊 Analyzing input files...")
        
        analysis = {
            'model_risk': [],
            'ai_ml': [],
            'training': [],
            'archive': [],
            'uncategorized': []
        }
        
        total_size = 0
        
        # Analyze files in input directory
        for file_path in self.input_dir.rglob('*'):
            if file_path.is_file():
                file_size = file_path.stat().st_size
                total_size += file_size
                relative_path = file_path.relative_to(self.input_dir)
                
                # Check categorization
                if str(relative_path) in self.model_risk_files:
                    analysis['model_risk'].append({
                        'path': relative_path,
                        'size': file_size,
                        'destination': self.model_risk_files[str(relative_path)]
                    })
                elif str(relative_path) in self.ai_ml_files:
                    analysis['ai_ml'].append({
                        'path': relative_path,
                        'size': file_size,
                        'destination': self.ai_ml_files[str(relative_path)]
                    })
                elif str(relative_path) in self.training_files:
                    analysis['training'].append({
                        'path': relative_path,
                        'size': file_size,
                        'destination': self.training_files[str(relative_path)]
                    })
                elif str(relative_path) in self.archive_files:
                    analysis['archive'].append({
                        'path': relative_path,
                        'size': file_size,
                        'destination': self.archive_files[str(relative_path)]
                    })
                else:
                    analysis['uncategorized'].append({
                        'path': relative_path,
                        'size': file_size
                    })
                    
        # Print analysis
        logger.info(f"📊 Total files analyzed: {sum(len(cat) for cat in analysis.values())}")
        logger.info(f"📊 Total size: {total_size / 1024 / 1024:.1f} MB")
        
        for category, files in analysis.items():
            if files:
                category_size = sum(f['size'] for f in files)
                logger.info(f"📁 {category.upper()}: {len(files)} files, {category_size / 1024 / 1024:.1f} MB")
                
                # Show first few uncategorized files for debugging
                if category == 'uncategorized' and len(files) > 0:
                    logger.info(f"   Sample uncategorized files:")
                    for i, file_info in enumerate(files[:5]):
                        logger.info(f"     - {file_info['path']} ({file_info['size'] / 1024 / 1024:.1f} MB)")
                    if len(files) > 5:
                        logger.info(f"     ... and {len(files) - 5} more files")
                
        return analysis
        
    def move_files(self, analysis):
        """Move files to their categorized locations"""
        logger.info("📦 Moving files to organized structure...")
        
        moved_count = 0
        total_size = 0
        
        # Move categorized files
        for category, files in analysis.items():
            if category == 'uncategorized':
                continue
                
            for file_info in files:
                source_path = self.input_dir / file_info['path']
                dest_path = self.knowledge_base_dir / file_info['destination'] / source_path.name
                
                if category == 'training':
                    dest_path = self.training_materials_dir / file_info['destination'] / source_path.name
                elif category == 'archive':
                    dest_path = self.archive_dir / file_info['destination'] / source_path.name
                    
                try:
                    # Create destination directory if it doesn't exist
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Move file
                    shutil.move(str(source_path), str(dest_path))
                    moved_count += 1
                    total_size += file_info['size']
                    logger.debug(f"Moved: {file_info['path']} -> {dest_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to move {file_info['path']}: {e}")
                    
        logger.info(f"✅ Moved {moved_count} files ({total_size / 1024 / 1024:.1f} MB)")
        
    def create_knowledge_base_summary(self, analysis):
        """Create a summary of the organized knowledge base"""
        logger.info("📋 Creating knowledge base summary...")
        
        summary_content = f"""# Knowledge Base Organization Summary

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 Organization Results

### Model Risk Management & Financial Services
- **Files**: {len(analysis['model_risk'])}
- **Size**: {sum(f['size'] for f in analysis['model_risk']) / 1024 / 1024:.1f} MB
- **Location**: `knowledge_base/model_risk_management/`

### AI/ML Research & Implementation
- **Files**: {len(analysis['ai_ml'])}
- **Size**: {sum(f['size'] for f in analysis['ai_ml']) / 1024 / 1024:.1f} MB
- **Location**: `knowledge_base/ai_ml_research/`

### Training Materials
- **Files**: {len(analysis['training'])}
- **Size**: {sum(f['size'] for f in analysis['training']) / 1024 / 1024:.1f} MB
- **Location**: `training_materials/`

### Archive
- **Files**: {len(analysis['archive'])}
- **Size**: {sum(f['size'] for f in analysis['archive']) / 1024 / 1024:.1f} MB
- **Location**: `archive/`

### Uncategorized
- **Files**: {len(analysis['uncategorized'])}
- **Size**: {sum(f['size'] for f in analysis['uncategorized']) / 1024 / 1024:.1f} MB

## 🎯 Next Steps

1. **Review uncategorized files** and assign them to appropriate categories
2. **Process knowledge base files** for embedding and search
3. **Create demo scenarios** using the organized content
4. **Test knowledge base queries** for practical use cases

## 📁 Directory Structure

```
knowledge_base/
├── model_risk_management/
│   ├── financial_modeling/
│   ├── regulatory_compliance/
│   └── business_intelligence/
├── ai_ml_research/
│   ├── attention_mechanisms/
│   ├── deep_learning_techniques/
│   ├── nlp_applications/
│   └── research_papers/
└── ...

training_materials/
├── course_materials/
├── reference_guides/
├── demo_examples/
└── templates/

archive/
├── entertainment_media/
├── academic_research/
├── code_repositories/
└── miscellaneous/
```
"""
        
        summary_path = self.project_root / "KNOWLEDGE_BASE_SUMMARY.md"
        with open(summary_path, 'w') as f:
            f.write(summary_content)
            
        logger.info(f"📋 Summary created: {summary_path}")
        return summary_path
        
    def organize(self, create_structure=True, move_files=True, analyze_only=False):
        """Main organization process"""
        logger.info("🚀 Starting knowledge base organization...")
        
        try:
            # Analyze files
            analysis = self.analyze_files()
            
            if analyze_only:
                logger.info("📊 Analysis complete (files not moved)")
                return analysis
                
            # Create directory structure
            if create_structure:
                self.create_directory_structure()
                
            # Move files
            if move_files:
                self.move_files(analysis)
                
            # Create summary
            summary_path = self.create_knowledge_base_summary(analysis)
            
            logger.info("🎉 Knowledge base organization complete!")
            logger.info(f"📋 Summary: {summary_path}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Organization failed: {e}")
            raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Organize input assets into knowledge base")
    parser.add_argument("--create-structure", action="store_true", help="Create directory structure")
    parser.add_argument("--move-files", action="store_true", help="Move files to organized structure")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze files, don't move them")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        organizer = KnowledgeBaseOrganizer()
        
        # Default behavior: do everything unless analyze-only is specified
        create_structure = args.create_structure or not args.analyze_only
        move_files = args.move_files or not args.analyze_only
        
        analysis = organizer.organize(
            create_structure=create_structure,
            move_files=move_files,
            analyze_only=args.analyze_only
        )
        
        print(f"\n🎉 Knowledge base organization complete!")
        print(f"📊 Files analyzed: {sum(len(cat) for cat in analysis.values())}")
        
    except Exception as e:
        logger.error(f"❌ Organization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
