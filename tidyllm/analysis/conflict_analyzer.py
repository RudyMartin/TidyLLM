#!/usr/bin/env python3
"""
Documentation Conflict Analyzer for SOP Domain RAG
===================================================

Analyzes all documentation in date folders to identify:
- Conflicting information between documents
- Different versions of the same topic
- Contradicting instructions or patterns
- Duplicate coverage of topics
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
import hashlib
import json

class DocumentConflictAnalyzer:
    def __init__(self, docs_path: str = "docs"):
        self.docs_path = Path(docs_path)
        self.documents = {}
        self.topics = {}
        self.conflicts = []
        
    def extract_topics(self, content: str) -> Set[str]:
        """Extract key topics from document content"""
        topics = set()
        
        # Architecture patterns
        if 'UnifiedSessionManager' in content:
            topics.add('session_management:unified')
        if 'Gateway' in content and 'pattern' in content.lower():
            topics.add('session_management:gateway')
        if 'boto3.client' in content:
            topics.add('session_management:direct_boto3')
        if 'psycopg2.connect' in content:
            topics.add('session_management:direct_psycopg2')
            
        # Embedding systems
        if 'tidyllm-sentence' in content:
            topics.add('embeddings:tidyllm_sentence')
        if 'tidyllm-vectorqa' in content:
            topics.add('embeddings:tidyllm_vectorqa')
        if 'sentence-transformers' in content or 'sentence_transformers' in content:
            topics.add('embeddings:sentence_transformers')
            
        # Workflow systems
        if 'RAG2DAG' in content:
            topics.add('workflow:rag2dag')
        if 'flow_agreements' in content:
            topics.add('workflow:flow_agreements')
        if 'HeirOS' in content:
            topics.add('workflow:heiros')
        if 'YAML' in content and 'workflow' in content.lower():
            topics.add('workflow:yaml')
            
        # Drop zones
        if 'drop_zones' in content or 'dropzones' in content.lower():
            topics.add('dropzones:implementation')
        if 'unified_drop_zones' in content:
            topics.add('dropzones:unified')
            
        # Testing
        if 'test suite' in content.lower() or 'test_' in content:
            topics.add('testing:framework')
            
        # S3/AWS patterns
        if 'S3' in content or 's3' in content:
            topics.add('aws:s3')
        if 'AWS' in content:
            topics.add('aws:general')
            
        # Extract headings as topics
        headings = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        for heading in headings:
            # Clean heading
            heading = heading.strip().lower()
            heading = re.sub(r'[^a-z0-9\s]', '', heading)
            if len(heading) > 3:
                topics.add(f'heading:{heading[:50]}')
                
        return topics
        
    def find_conflicts(self, doc1_info: Dict, doc2_info: Dict) -> List[Dict]:
        """Find conflicts between two documents"""
        conflicts_found = []
        
        # Check for same topics with different approaches
        shared_topics = doc1_info['topics'] & doc2_info['topics']
        
        for topic in shared_topics:
            if ':' in topic:
                category, value = topic.split(':', 1)
                
                # Check for conflicting patterns in same category
                doc1_category_topics = {t for t in doc1_info['topics'] if t.startswith(category + ':')}
                doc2_category_topics = {t for t in doc2_info['topics'] if t.startswith(category + ':')}
                
                if len(doc1_category_topics) > 1 or len(doc2_category_topics) > 1:
                    # Different approaches to same category
                    conflicts_found.append({
                        'type': 'pattern_conflict',
                        'category': category,
                        'doc1': doc1_info['path'],
                        'doc1_date': doc1_info['date'],
                        'doc1_patterns': list(doc1_category_topics),
                        'doc2': doc2_info['path'],
                        'doc2_date': doc2_info['date'],
                        'doc2_patterns': list(doc2_category_topics),
                        'severity': 'high' if category in ['session_management', 'embeddings'] else 'medium'
                    })
                    
        # Check for contradicting instructions
        if doc1_info['content'] and doc2_info['content']:
            # Look for DO NOT vs DO patterns
            doc1_dont = re.findall(r'DO NOT\s+([^\.]+)', doc1_info['content'], re.IGNORECASE)
            doc2_do = re.findall(r'(?:DO|USE|ALWAYS)\s+([^\.]+)', doc2_info['content'], re.IGNORECASE)
            
            for dont in doc1_dont:
                for do in doc2_do:
                    if any(word in do.lower() for word in dont.lower().split()):
                        conflicts_found.append({
                            'type': 'instruction_conflict',
                            'doc1': doc1_info['path'],
                            'doc1_says': f"DO NOT {dont}",
                            'doc2': doc2_info['path'],
                            'doc2_says': f"DO {do}",
                            'severity': 'high'
                        })
                        
        return conflicts_found
        
    def analyze_all_documents(self) -> Dict:
        """Analyze all documents in date folders"""
        print("[SCAN] Loading all documents from date folders...")
        
        # Load all documents
        for date_folder in sorted(self.docs_path.iterdir()):
            if not date_folder.is_dir() or not re.match(r'\d{4}-\d{2}-\d{2}', date_folder.name):
                continue
                
            date = date_folder.name
            
            for doc_file in date_folder.iterdir():
                if doc_file.suffix in ['.md', '.txt', '.rst']:
                    try:
                        with open(doc_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        doc_info = {
                            'path': str(doc_file),
                            'name': doc_file.name,
                            'date': date,
                            'size': len(content),
                            'content': content,
                            'topics': self.extract_topics(content),
                            'hash': hashlib.md5(content.encode()).hexdigest()
                        }
                        
                        self.documents[str(doc_file)] = doc_info
                        
                        # Index by topic
                        for topic in doc_info['topics']:
                            if topic not in self.topics:
                                self.topics[topic] = []
                            self.topics[topic].append(doc_info)
                            
                    except Exception as e:
                        print(f"[ERROR] Could not read {doc_file}: {e}")
                        
        print(f"[SCAN] Loaded {len(self.documents)} documents covering {len(self.topics)} topics")
        
        # Find conflicts
        print("[ANALYZE] Looking for conflicts...")
        
        # Compare documents with overlapping topics
        doc_list = list(self.documents.values())
        
        for i, doc1 in enumerate(doc_list):
            for doc2 in doc_list[i+1:]:
                # Skip if same document
                if doc1['hash'] == doc2['hash']:
                    continue
                    
                # Check if they share topics
                if doc1['topics'] & doc2['topics']:
                    conflicts = self.find_conflicts(doc1, doc2)
                    self.conflicts.extend(conflicts)
                    
        # Deduplicate conflicts
        seen = set()
        unique_conflicts = []
        for conflict in self.conflicts:
            # Create unique key
            key = f"{conflict.get('type')}:{conflict.get('category', '')}:{conflict.get('doc1', '')}:{conflict.get('doc2', '')}"
            if key not in seen:
                seen.add(key)
                unique_conflicts.append(conflict)
                
        self.conflicts = unique_conflicts
        
        return {
            'total_documents': len(self.documents),
            'total_topics': len(self.topics),
            'total_conflicts': len(self.conflicts),
            'documents': self.documents,
            'topics': {k: len(v) for k, v in self.topics.items()},
            'conflicts': self.conflicts
        }
        
    def generate_report(self) -> str:
        """Generate conflict analysis report"""
        results = self.analyze_all_documents()
        
        report = f"""
=== DOCUMENTATION CONFLICT ANALYSIS FOR SOP ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Documents Path: {self.docs_path}

[SUMMARY]
- Total documents analyzed: {results['total_documents']}
- Total unique topics: {results['total_topics']}
- Total conflicts found: {results['total_conflicts']}

[CRITICAL CONFLICTS - ARCHITECTURAL PATTERNS]
"""
        
        # Group conflicts by type
        high_severity = [c for c in results['conflicts'] if c.get('severity') == 'high']
        medium_severity = [c for c in results['conflicts'] if c.get('severity') == 'medium']
        
        if high_severity:
            report += f"\nFound {len(high_severity)} HIGH severity conflicts:\n"
            
            # Group by category
            by_category = {}
            for conflict in high_severity:
                cat = conflict.get('category', conflict.get('type', 'other'))
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(conflict)
                
            for category, conflicts in by_category.items():
                report += f"\n[{category.upper()}] - {len(conflicts)} conflicts:\n"
                for c in conflicts[:3]:  # Show first 3
                    if c['type'] == 'pattern_conflict':
                        report += f"  - {Path(c['doc1']).name} ({c['doc1_date']}) vs {Path(c['doc2']).name} ({c['doc2_date']})\n"
                        report += f"    Doc1 patterns: {', '.join(c['doc1_patterns'])}\n"
                        report += f"    Doc2 patterns: {', '.join(c['doc2_patterns'])}\n"
                    else:
                        report += f"  - {c}\n"
                        
        report += f"""
[TOPIC COVERAGE ANALYSIS]
Topics covered by multiple documents (potential conflicts):
"""
        
        # Find topics with multiple documents
        multi_doc_topics = {k: v for k, v in self.topics.items() if len(v) > 1}
        
        for topic, docs in sorted(multi_doc_topics.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            report += f"\n{topic}: {len(docs)} documents\n"
            for doc in docs[:3]:
                report += f"  - {doc['name']} ({doc['date']})\n"
                
        report += f"""
[RECOMMENDATIONS FOR SOP DOMAIN RAG]

1. RESOLVE ARCHITECTURAL CONFLICTS:
"""
        
        # Identify main conflict areas
        conflict_areas = set()
        for c in high_severity:
            if 'category' in c:
                conflict_areas.add(c['category'])
                
        for area in conflict_areas:
            report += f"   - {area}: Multiple competing patterns found\n"
            
        report += f"""
2. DOCUMENT VERSIONS TO PRIORITIZE:
   - Use most recent documentation (2025-09-05) as primary source
   - Older docs (2025-09-01 to 2025-09-04) for historical context
   - Flag deprecated patterns explicitly

3. SOP CREATION STRATEGY:
   - Create separate SOPs for each architectural decision area
   - Document approved patterns vs deprecated patterns
   - Include migration paths where conflicts exist
   
4. CONFLICT RESOLUTION NEEDED:
"""
        
        if high_severity:
            for c in high_severity[:5]:
                if c['type'] == 'pattern_conflict':
                    report += f"   - {c['category']}: Choose between {', '.join(set(c['doc1_patterns'] + c['doc2_patterns']))}\n"
                    
        report += f"""
[FOR DOMAIN RAG PROCESSING]
- Feed all documents to establish knowledge base
- Use conflict detection to generate clarification questions
- Prioritize by date (newest = most authoritative)
- Tag conflicting information for human review
"""
        
        return report, results

def main():
    """Main execution"""
    analyzer = DocumentConflictAnalyzer()
    
    print("=" * 60)
    print("DOCUMENTATION CONFLICT ANALYSIS")
    print("Preparing for SOP Domain RAG")
    print("=" * 60)
    
    report, results = analyzer.generate_report()
    
    # Save report
    report_file = f"docs_conflict_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
        
    # Save detailed results as JSON
    json_file = report_file.replace('.txt', '.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        # Convert sets to lists for JSON serialization
        json_results = results.copy()
        json_results['conflicts'] = results['conflicts']
        json_results['documents'] = {k: {**v, 'topics': list(v['topics'])} 
                                     for k, v in results['documents'].items()}
        json.dump(json_results, f, indent=2, default=str)
    
    print(report)
    print(f"\n[SAVED] Conflict analysis: {report_file}")
    print(f"[SAVED] Detailed results: {json_file}")
    print("\n[READY] Documents organized by date and conflicts identified for SOP Domain RAG")

if __name__ == "__main__":
    main()