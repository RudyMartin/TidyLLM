#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download Data Science Whitepapers Script
========================================

This script automatically downloads top Data Science whitepapers from various sources
and adds them to our knowledge base.

Usage:
    python scripts/download_whitepapers.py [--source arxiv] [--category all] [--limit 20]
"""

import os
import sys
import requests
import argparse
import logging
import time
import json
from pathlib import Path
from urllib.parse import urlparse, quote
import arxiv
import feedparser
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WhitepaperDownloader:
    """Download Data Science whitepapers from various sources"""
    
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.knowledge_base_dir = self.project_root / "knowledge_base"
        self.whitepapers_dir = self.knowledge_base_dir / "ai_ml_research" / "whitepapers"
        
        # Create directories
        self.whitepapers_dir.mkdir(parents=True, exist_ok=True)
        
        # User agent for requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Top Data Science whitepapers (curated list)
        self.top_whitepapers = [
            # Foundational Papers
            {
                'title': 'Attention Is All You Need',
                'authors': 'Vaswani et al.',
                'year': 2017,
                'arxiv_id': '1706.03762',
                'category': 'transformers',
                'description': 'The original transformer architecture paper'
            },
            {
                'title': 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
                'authors': 'Devlin et al.',
                'year': 2018,
                'arxiv_id': '1810.04805',
                'category': 'nlp',
                'description': 'Bidirectional transformer for language understanding'
            },
            {
                'title': 'Generative Adversarial Networks',
                'authors': 'Goodfellow et al.',
                'year': 2014,
                'arxiv_id': '1406.2661',
                'category': 'generative_ai',
                'description': 'Original GAN paper'
            },
            {
                'title': 'Deep Learning',
                'authors': 'LeCun et al.',
                'year': 2015,
                'arxiv_id': '1312.6026',
                'category': 'deep_learning',
                'description': 'Comprehensive review of deep learning'
            },
            {
                'title': 'ImageNet Classification with Deep Convolutional Neural Networks',
                'authors': 'Krizhevsky et al.',
                'year': 2012,
                'arxiv_id': '1207.0580',
                'category': 'computer_vision',
                'description': 'AlexNet - breakthrough in image classification'
            },
            
            # Modern Papers
            {
                'title': 'Language Models are Few-Shot Learners',
                'authors': 'Brown et al.',
                'year': 2020,
                'arxiv_id': '2005.14165',
                'category': 'llms',
                'description': 'GPT-3 paper on few-shot learning'
            },
            {
                'title': 'Training language models to follow instructions with human feedback',
                'authors': 'Ouyang et al.',
                'year': 2022,
                'arxiv_id': '2203.02155',
                'category': 'rlhf',
                'description': 'InstructGPT - RLHF paper'
            },
            {
                'title': 'Scaling Laws for Neural Language Models',
                'authors': 'Kaplan et al.',
                'year': 2020,
                'arxiv_id': '2001.08361',
                'category': 'scaling',
                'description': 'Scaling laws for language models'
            },
            {
                'title': 'LoRA: Low-Rank Adaptation of Large Language Models',
                'authors': 'Hu et al.',
                'year': 2021,
                'arxiv_id': '2106.09685',
                'category': 'efficient_training',
                'description': 'Parameter-efficient fine-tuning method'
            },
            {
                'title': 'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness',
                'authors': 'Dao et al.',
                'year': 2022,
                'arxiv_id': '2205.14135',
                'category': 'efficiency',
                'description': 'Memory-efficient attention mechanism'
            },
            
            # Financial/ML Papers
            {
                'title': 'Deep Learning for Financial Time Series',
                'authors': 'Sezer et al.',
                'year': 2020,
                'arxiv_id': '2003.06778',
                'category': 'financial_ml',
                'description': 'Comprehensive review of DL in finance'
            },
            {
                'title': 'Machine Learning for Market Microstructure and High Frequency Trading',
                'authors': 'Cartea et al.',
                'year': 2018,
                'arxiv_id': '1805.03231',
                'category': 'financial_ml',
                'description': 'ML applications in high-frequency trading'
            },
            {
                'title': 'Deep Reinforcement Learning for Trading',
                'authors': 'Deng et al.',
                'year': 2017,
                'arxiv_id': '1706.10059',
                'category': 'financial_ml',
                'description': 'RL applications in algorithmic trading'
            },
            
            # MLOps & Production
            {
                'title': 'Hidden Technical Debt in Machine Learning Systems',
                'authors': 'Sculley et al.',
                'year': 2015,
                'arxiv_id': '1502.05767',
                'category': 'mlops',
                'description': 'Technical debt in ML systems'
            },
            {
                'title': 'Machine Learning: The High Interest Credit Card of Technical Debt',
                'authors': 'Sculley et al.',
                'year': 2014,
                'arxiv_id': '1402.1128',
                'category': 'mlops',
                'description': 'Technical debt in ML'
            },
            {
                'title': 'A Survey of Machine Learning for Computer Architecture and Systems',
                'authors': 'Li et al.',
                'year': 2020,
                'arxiv_id': '2009.00809',
                'category': 'systems',
                'description': 'ML for computer architecture'
            },
            
            # Model Risk Management
            {
                'title': 'Model Risk Management',
                'authors': 'Derman',
                'year': 1996,
                'source': 'goldman_sachs',
                'category': 'model_risk',
                'description': 'Classic paper on model risk management'
            },
            {
                'title': 'Machine Learning Model Risk Management',
                'authors': 'Various',
                'year': 2021,
                'source': 'federal_reserve',
                'category': 'model_risk',
                'description': 'Federal Reserve guidance on ML model risk'
            },
            
            # Recent Breakthroughs
            {
                'title': 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',
                'authors': 'Lewis et al.',
                'year': 2020,
                'arxiv_id': '2005.11401',
                'category': 'rag',
                'description': 'RAG architecture paper'
            },
            {
                'title': 'Chain-of-Thought Prompting Elicits Reasoning in Large Language Models',
                'authors': 'Wei et al.',
                'year': 2022,
                'arxiv_id': '2201.11903',
                'category': 'reasoning',
                'description': 'Chain-of-thought reasoning in LLMs'
            }
        ]
        
        # Category mappings for knowledge base organization
        self.category_mappings = {
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
            'model_risk': 'model_risk_management',
            'rag': 'retrieval_augmented_generation',
            'reasoning': 'reasoning_techniques'
        }
    
    def download_from_arxiv(self, paper_info, limit=20):
        """Download paper from arXiv"""
        try:
            if 'arxiv_id' not in paper_info:
                logger.warning(f"No arXiv ID for paper: {paper_info['title']}")
                return None
                
            # Search for the paper
            search = arxiv.Search(id_list=[paper_info['arxiv_id']])
            results = list(search.results())
            
            if not results:
                logger.warning(f"Paper not found on arXiv: {paper_info['title']}")
                return None
                
            paper = results[0]
            
            # Create filename
            safe_title = "".join(c for c in paper_info['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{paper_info['year']}_{safe_title}_{paper_info['arxiv_id']}.pdf"
            filepath = self.whitepapers_dir / filename
            
            # Download if not already exists
            if not filepath.exists():
                logger.info(f"Downloading: {paper_info['title']}")
                paper.download_pdf(filename=str(filepath))
                time.sleep(1)  # Be nice to arXiv
                logger.info(f"✅ Downloaded: {filename}")
            else:
                logger.info(f"📄 Already exists: {filename}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to download {paper_info['title']}: {e}")
            return None
    
    def download_from_url(self, paper_info, url):
        """Download paper from direct URL"""
        try:
            safe_title = "".join(c for c in paper_info['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{paper_info['year']}_{safe_title}_{paper_info.get('source', 'external')}.pdf"
            filepath = self.whitepapers_dir / filename
            
            if not filepath.exists():
                logger.info(f"Downloading: {paper_info['title']}")
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"✅ Downloaded: {filename}")
            else:
                logger.info(f"📄 Already exists: {filename}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to download {paper_info['title']}: {e}")
            return None
    
    def search_recent_papers(self, category=None, limit=10):
        """Search for recent papers in specific categories"""
        logger.info(f"🔍 Searching for recent papers in category: {category or 'all'}")
        
        # Define search queries based on category
        search_queries = {
            'financial_ml': [
                'machine learning finance',
                'deep learning trading',
                'neural networks financial',
                'AI risk management',
                'model validation financial'
            ],
            'model_risk': [
                'model risk management',
                'AI model governance',
                'machine learning validation',
                'model monitoring',
                'AI risk assessment'
            ],
            'mlops': [
                'MLOps best practices',
                'machine learning production',
                'ML model deployment',
                'model serving',
                'ML pipeline'
            ],
            'all': [
                'machine learning',
                'deep learning',
                'artificial intelligence',
                'data science',
                'neural networks'
            ]
        }
        
        queries = search_queries.get(category, search_queries['all'])
        recent_papers = []
        
        for query in queries[:3]:  # Limit queries to avoid rate limiting
            try:
                search = arxiv.Search(
                    query=query,
                    max_results=limit//len(queries),
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                results = list(search.results())
                for paper in results:
                    recent_papers.append({
                        'title': paper.title,
                        'authors': ', '.join(author.name for author in paper.authors),
                        'year': paper.published.year,
                        'arxiv_id': paper.entry_id.split('/')[-1],
                        'category': category or 'recent',
                        'description': paper.summary[:200] + '...',
                        'url': paper.pdf_url
                    })
                
                time.sleep(2)  # Be nice to arXiv
                
            except Exception as e:
                logger.error(f"Failed to search for '{query}': {e}")
        
        return recent_papers
    
    def organize_by_category(self, downloaded_files):
        """Organize downloaded files by category"""
        logger.info("📁 Organizing files by category...")
        
        for filepath in downloaded_files:
            if filepath and filepath.exists():
                # Extract category from filename or metadata
                filename = filepath.name
                
                # Determine category based on filename patterns
                category = 'general'
                if any(keyword in filename.lower() for keyword in ['financial', 'trading', 'risk']):
                    category = 'financial_modeling'
                elif any(keyword in filename.lower() for keyword in ['attention', 'transformer']):
                    category = 'attention_mechanisms'
                elif any(keyword in filename.lower() for keyword in ['nlp', 'language']):
                    category = 'nlp_applications'
                elif any(keyword in filename.lower() for keyword in ['gan', 'generative']):
                    category = 'generative_models'
                elif any(keyword in filename.lower() for keyword in ['mlops', 'production']):
                    category = 'mlops_best_practices'
                
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
                category = 'general'
                
                # Determine category
                if any(keyword in filename.lower() for keyword in ['financial', 'trading', 'risk']):
                    category = 'Financial Modeling'
                elif any(keyword in filename.lower() for keyword in ['attention', 'transformer']):
                    category = 'Attention Mechanisms'
                elif any(keyword in filename.lower() for keyword in ['nlp', 'language']):
                    category = 'NLP Applications'
                elif any(keyword in filename.lower() for keyword in ['gan', 'generative']):
                    category = 'Generative Models'
                elif any(keyword in filename.lower() for keyword in ['mlops', 'production']):
                    category = 'MLOps & Production'
                else:
                    category = 'General ML'
                
                if category not in papers_by_category:
                    papers_by_category[category] = []
                papers_by_category[category].append(filename)
        
        # Add papers to index
        for category, papers in papers_by_category.items():
            index_content += f"### {category}\n"
            for paper in papers:
                index_content += f"- {paper}\n"
            index_content += "\n"
        
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
"""
        
        index_path = self.whitepapers_dir / "WHITEPAPERS_INDEX.md"
        with open(index_path, 'w') as f:
            f.write(index_content)
        
        logger.info(f"📋 Index created: {index_path}")
        return index_path
    
    def download_whitepapers(self, source='arxiv', category='all', limit=20, include_recent=True):
        """Main method to download whitepapers"""
        logger.info(f"🚀 Starting whitepaper download...")
        logger.info(f"Source: {source}, Category: {category}, Limit: {limit}")
        
        downloaded_files = []
        
        # Download curated top papers
        logger.info("📚 Downloading curated top papers...")
        for paper in self.top_whitepapers[:limit//2]:
            if source == 'arxiv' and 'arxiv_id' in paper:
                filepath = self.download_from_arxiv(paper)
                if filepath:
                    downloaded_files.append(filepath)
        
        # Download recent papers if requested
        if include_recent:
            logger.info("🔍 Downloading recent papers...")
            recent_papers = self.search_recent_papers(category, limit//2)
            
            for paper in recent_papers:
                filepath = self.download_from_arxiv(paper)
                if filepath:
                    downloaded_files.append(filepath)
        
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
    parser = argparse.ArgumentParser(description="Download Data Science whitepapers")
    parser.add_argument("--source", default="arxiv", choices=["arxiv", "url"], help="Source for papers")
    parser.add_argument("--category", default="all", help="Category to focus on")
    parser.add_argument("--limit", type=int, default=20, help="Number of papers to download")
    parser.add_argument("--no-recent", action="store_true", help="Don't include recent papers")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        downloader = WhitepaperDownloader()
        
        files, index = downloader.download_whitepapers(
            source=args.source,
            category=args.category,
            limit=args.limit,
            include_recent=not args.no_recent
        )
        
        print(f"\n🎉 Whitepaper download complete!")
        print(f"📊 Files downloaded: {len(files)}")
        print(f"📋 Index: {index}")
        print(f"📁 Location: {downloader.whitepapers_dir}")
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
