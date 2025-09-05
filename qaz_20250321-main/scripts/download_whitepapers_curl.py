#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download Data Science Whitepapers using curl/wget
================================================

This script uses curl/wget to download top Data Science whitepapers
and adds them to our knowledge base.

Usage:
    python scripts/download_whitepapers_curl.py [--limit 20]
"""

import os
import sys
import subprocess
import argparse
import logging
import time
import json
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WhitepaperDownloaderCurl:
    """Download Data Science whitepapers using curl/wget"""
    
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.knowledge_base_dir = self.project_root / "knowledge_base"
        self.whitepapers_dir = self.knowledge_base_dir / "ai_ml_research" / "whitepapers"
        
        # Create directories
        self.whitepapers_dir.mkdir(parents=True, exist_ok=True)
        
        # Top Data Science whitepapers with direct URLs
        self.top_whitepapers = [
            # Foundational Papers
            {
                'title': 'Attention Is All You Need',
                'authors': 'Vaswani et al.',
                'year': 2017,
                'url': 'https://arxiv.org/pdf/1706.03762.pdf',
                'category': 'transformers',
                'description': 'The original transformer architecture paper'
            },
            {
                'title': 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
                'authors': 'Devlin et al.',
                'year': 2018,
                'url': 'https://arxiv.org/pdf/1810.04805.pdf',
                'category': 'nlp',
                'description': 'Bidirectional transformer for language understanding'
            },
            {
                'title': 'Generative Adversarial Networks',
                'authors': 'Goodfellow et al.',
                'year': 2014,
                'url': 'https://arxiv.org/pdf/1406.2661.pdf',
                'category': 'generative_ai',
                'description': 'Original GAN paper'
            },
            {
                'title': 'Deep Learning',
                'authors': 'LeCun et al.',
                'year': 2015,
                'url': 'https://arxiv.org/pdf/1312.6026.pdf',
                'category': 'deep_learning',
                'description': 'Comprehensive review of deep learning'
            },
            {
                'title': 'ImageNet Classification with Deep Convolutional Neural Networks',
                'authors': 'Krizhevsky et al.',
                'year': 2012,
                'url': 'https://arxiv.org/pdf/1207.0580.pdf',
                'category': 'computer_vision',
                'description': 'AlexNet - breakthrough in image classification'
            },
            
            # Modern Papers
            {
                'title': 'Language Models are Few-Shot Learners',
                'authors': 'Brown et al.',
                'year': 2020,
                'url': 'https://arxiv.org/pdf/2005.14165.pdf',
                'category': 'llms',
                'description': 'GPT-3 paper on few-shot learning'
            },
            {
                'title': 'Training language models to follow instructions with human feedback',
                'authors': 'Ouyang et al.',
                'year': 2022,
                'url': 'https://arxiv.org/pdf/2203.02155.pdf',
                'category': 'rlhf',
                'description': 'InstructGPT - RLHF paper'
            },
            {
                'title': 'Scaling Laws for Neural Language Models',
                'authors': 'Kaplan et al.',
                'year': 2020,
                'url': 'https://arxiv.org/pdf/2001.08361.pdf',
                'category': 'scaling',
                'description': 'Scaling laws for language models'
            },
            {
                'title': 'LoRA: Low-Rank Adaptation of Large Language Models',
                'authors': 'Hu et al.',
                'year': 2021,
                'url': 'https://arxiv.org/pdf/2106.09685.pdf',
                'category': 'efficient_training',
                'description': 'Parameter-efficient fine-tuning method'
            },
            {
                'title': 'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness',
                'authors': 'Dao et al.',
                'year': 2022,
                'url': 'https://arxiv.org/pdf/2205.14135.pdf',
                'category': 'efficiency',
                'description': 'Memory-efficient attention mechanism'
            },
            
            # Financial/ML Papers
            {
                'title': 'Deep Learning for Financial Time Series',
                'authors': 'Sezer et al.',
                'year': 2020,
                'url': 'https://arxiv.org/pdf/2003.06778.pdf',
                'category': 'financial_ml',
                'description': 'Comprehensive review of DL in finance'
            },
            {
                'title': 'Machine Learning for Market Microstructure and High Frequency Trading',
                'authors': 'Cartea et al.',
                'year': 2018,
                'url': 'https://arxiv.org/pdf/1805.03231.pdf',
                'category': 'financial_ml',
                'description': 'ML applications in high-frequency trading'
            },
            {
                'title': 'Deep Reinforcement Learning for Trading',
                'authors': 'Deng et al.',
                'year': 2017,
                'url': 'https://arxiv.org/pdf/1706.10059.pdf',
                'category': 'financial_ml',
                'description': 'RL applications in algorithmic trading'
            },
            
            # MLOps & Production
            {
                'title': 'Hidden Technical Debt in Machine Learning Systems',
                'authors': 'Sculley et al.',
                'year': 2015,
                'url': 'https://arxiv.org/pdf/1502.05767.pdf',
                'category': 'mlops',
                'description': 'Technical debt in ML systems'
            },
            {
                'title': 'Machine Learning: The High Interest Credit Card of Technical Debt',
                'authors': 'Sculley et al.',
                'year': 2014,
                'url': 'https://arxiv.org/pdf/1402.1128.pdf',
                'category': 'mlops',
                'description': 'Technical debt in ML'
            },
            {
                'title': 'A Survey of Machine Learning for Computer Architecture and Systems',
                'authors': 'Li et al.',
                'year': 2020,
                'url': 'https://arxiv.org/pdf/2009.00809.pdf',
                'category': 'systems',
                'description': 'ML for computer architecture'
            },
            
            # Recent Breakthroughs
            {
                'title': 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',
                'authors': 'Lewis et al.',
                'year': 2020,
                'url': 'https://arxiv.org/pdf/2005.11401.pdf',
                'category': 'rag',
                'description': 'RAG architecture paper'
            },
            {
                'title': 'Chain-of-Thought Prompting Elicits Reasoning in Large Language Models',
                'authors': 'Wei et al.',
                'year': 2022,
                'url': 'https://arxiv.org/pdf/2201.11903.pdf',
                'category': 'reasoning',
                'description': 'Chain-of-thought reasoning in LLMs'
            },
            {
                'title': 'Large Language Models are Human-Level Prompt Engineers',
                'authors': 'Zhou et al.',
                'year': 2022,
                'url': 'https://arxiv.org/pdf/2211.01910.pdf',
                'category': 'prompting',
                'description': 'Auto-prompting with LLMs'
            },
            {
                'title': 'Constitutional AI: Harmlessness from AI Feedback',
                'authors': 'Bai et al.',
                'year': 2022,
                'url': 'https://arxiv.org/pdf/2212.08073.pdf',
                'category': 'alignment',
                'description': 'Constitutional AI approach'
            }
        ]
    
    def download_with_curl(self, paper_info):
        """Download paper using curl"""
        try:
            # Create filename
            safe_title = "".join(c for c in paper_info['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{paper_info['year']}_{safe_title}.pdf"
            filepath = self.whitepapers_dir / filename
            
            # Download if not already exists
            if not filepath.exists():
                logger.info(f"Downloading: {paper_info['title']}")
                
                # Use curl with proper headers and SSL handling
                cmd = [
                    'curl', '-L', '--max-time', '60', '--retry', '3',
                    '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    '--output', str(filepath),
                    paper_info['url']
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and filepath.exists() and filepath.stat().st_size > 0:
                    logger.info(f"✅ Downloaded: {filename}")
                    return filepath
                else:
                    logger.error(f"Failed to download {paper_info['title']}: {result.stderr}")
                    if filepath.exists():
                        filepath.unlink()  # Remove failed download
                    return None
            else:
                logger.info(f"📄 Already exists: {filename}")
                return filepath
                
        except Exception as e:
            logger.error(f"Failed to download {paper_info['title']}: {e}")
            return None
    
    def download_with_wget(self, paper_info):
        """Download paper using wget (alternative to curl)"""
        try:
            # Create filename
            safe_title = "".join(c for c in paper_info['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{paper_info['year']}_{safe_title}.pdf"
            filepath = self.whitepapers_dir / filename
            
            # Download if not already exists
            if not filepath.exists():
                logger.info(f"Downloading: {paper_info['title']}")
                
                # Use wget with proper options
                cmd = [
                    'wget', '--timeout=60', '--tries=3',
                    '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    '-O', str(filepath),
                    paper_info['url']
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and filepath.exists() and filepath.stat().st_size > 0:
                    logger.info(f"✅ Downloaded: {filename}")
                    return filepath
                else:
                    logger.error(f"Failed to download {paper_info['title']}: {result.stderr}")
                    if filepath.exists():
                        filepath.unlink()  # Remove failed download
                    return None
            else:
                logger.info(f"📄 Already exists: {filename}")
                return filepath
                
        except Exception as e:
            logger.error(f"Failed to download {paper_info['title']}: {e}")
            return None
    
    def organize_by_category(self, downloaded_files):
        """Organize downloaded files by category"""
        logger.info("📁 Organizing files by category...")
        
        category_mappings = {
            'transformers': 'attention_mechanisms',
            'nlp': 'nlp_applications',
            'generative_ai': 'generative_models',
            'deep_learning': 'deep_learning_techniques',
            'computer_vision': 'computer_vision',
            'llms': 'large_language_models',
            'rlhf': 'reinforcement_learning',
            'scaling': 'scaling_techniques',
            'efficient_training': 'efficient_training',
            'efficiency': 'efficiency_optimization',
            'financial_ml': 'financial_modeling',
            'mlops': 'mlops_best_practices',
            'systems': 'systems_integration',
            'rag': 'retrieval_augmented_generation',
            'reasoning': 'reasoning_techniques',
            'prompting': 'prompt_engineering',
            'alignment': 'ai_alignment'
        }
        
        for filepath in downloaded_files:
            if filepath and filepath.exists():
                # Find the paper info to get category
                filename = filepath.name
                paper_info = None
                
                for paper in self.top_whitepapers:
                    safe_title = "".join(c for c in paper['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    expected_filename = f"{paper['year']}_{safe_title}.pdf"
                    if filename == expected_filename:
                        paper_info = paper
                        break
                
                if paper_info:
                    category = category_mappings.get(paper_info['category'], 'general')
                    
                    # Create category directory
                    category_dir = self.knowledge_base_dir / "ai_ml_research" / category
                    category_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Move file to category directory
                    dest_path = category_dir / filename
                    if not dest_path.exists():
                        filepath.rename(dest_path)
                        logger.debug(f"Moved {filename} to {category}")
    
    def create_whitepapers_index(self, downloaded_files):
        """Create an index of downloaded whitepapers"""
        logger.info("📋 Creating whitepapers index...")
        
        index_content = f"""# Data Science Whitepapers Index

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 Download Summary

- **Total Papers**: {len(downloaded_files)}
- **Total Size**: {sum(f.stat().st_size for f in downloaded_files if f and f.exists()) / 1024 / 1024:.1f} MB

## 📚 Papers by Category

"""
        
        # Group papers by category
        papers_by_category = {}
        for filepath in downloaded_files:
            if filepath and filepath.exists():
                filename = filepath.name
                
                # Find paper info to get category
                paper_info = None
                for paper in self.top_whitepapers:
                    safe_title = "".join(c for c in paper['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    expected_filename = f"{paper['year']}_{safe_title}.pdf"
                    if filename == expected_filename:
                        paper_info = paper
                        break
                
                if paper_info:
                    category = paper_info['category'].replace('_', ' ').title()
                    if category not in papers_by_category:
                        papers_by_category[category] = []
                    papers_by_category[category].append({
                        'filename': filename,
                        'title': paper_info['title'],
                        'authors': paper_info['authors'],
                        'year': paper_info['year'],
                        'description': paper_info['description']
                    })
        
        # Add papers to index
        for category, papers in papers_by_category.items():
            index_content += f"### {category}\n"
            for paper in papers:
                index_content += f"- **{paper['title']}** ({paper['year']}) - {paper['authors']}\n"
                index_content += f"  - {paper['description']}\n"
                index_content += f"  - File: `{paper['filename']}`\n\n"
        
        index_content += f"""
## 🎯 Usage

These whitepapers are now part of the knowledge base and can be used for:

1. **Model Risk Management**: Reference papers for validation approaches
2. **AI/ML Research**: Latest developments in the field
3. **Training Materials**: Educational content for teams
4. **Best Practices**: Production and deployment guidance

## 📁 File Locations

All papers are stored in: `knowledge_base/ai_ml_research/whitepapers/`

Papers are also organized by category in their respective subdirectories.

## 🔗 Sources

All papers are downloaded from arXiv.org with proper attribution.
"""
        
        index_path = self.whitepapers_dir / "WHITEPAPERS_INDEX.md"
        with open(index_path, 'w') as f:
            f.write(index_content)
        
        logger.info(f"📋 Index created: {index_path}")
        return index_path
    
    def download_whitepapers(self, limit=20, use_wget=False):
        """Main method to download whitepapers"""
        logger.info(f"🚀 Starting whitepaper download...")
        logger.info(f"Limit: {limit}, Method: {'wget' if use_wget else 'curl'}")
        
        downloaded_files = []
        
        # Download papers
        for i, paper in enumerate(self.top_whitepapers[:limit]):
            logger.info(f"📄 [{i+1}/{min(limit, len(self.top_whitepapers))}] Processing: {paper['title']}")
            
            if use_wget:
                filepath = self.download_with_wget(paper)
            else:
                filepath = self.download_with_curl(paper)
            
            if filepath:
                downloaded_files.append(filepath)
            
            # Be nice to servers
            time.sleep(1)
        
        # Organize files by category
        self.organize_by_category(downloaded_files)
        
        # Create index
        index_path = self.create_whitepapers_index(downloaded_files)
        
        logger.info(f"🎉 Download complete!")
        logger.info(f"📊 Files downloaded: {len(downloaded_files)}")
        logger.info(f"📋 Index: {index_path}")
        
        return downloaded_files, index_path

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Download Data Science whitepapers using curl/wget")
    parser.add_argument("--limit", type=int, default=20, help="Number of papers to download")
    parser.add_argument("--use-wget", action="store_true", help="Use wget instead of curl")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        downloader = WhitepaperDownloaderCurl()
        
        files, index = downloader.download_whitepapers(
            limit=args.limit,
            use_wget=args.use_wget
        )
        
        print(f"\n🎉 Whitepaper download complete!")
        print(f"📊 Files downloaded: {len(files)}")
        print(f"📋 Index: {index}")
        print(f"📁 Location: {downloader.whitepapers_dir}")
        
        if files:
            total_size = sum(f.stat().st_size for f in files if f and f.exists()) / 1024 / 1024
            print(f"📊 Total size: {total_size:.1f} MB")
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
