"""
YRSN Search Tracker
==================

Track Y=R+S+N search results over time to analyze research trends,
quality patterns, and topic evolution.

Uses the existing PostgreSQL database infrastructure.
Created for the TidyLLM Vector QA project.
"""

try:
    import psycopg2
    from psycopg2 import sql
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2 = None

import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import os

@dataclass
class SearchResult:
    """Individual search result with Y=R+S+N analysis"""
    paper_id: str           # ArXiv ID, PubMed ID, etc.
    title: str
    authors: List[str] 
    abstract: str
    source: str             # "ArXiv", "PubMed", etc.
    y_score: float
    r_score: float
    s_score: float  
    n_score: float
    context_risk: float
    semantic_relevance: float

@dataclass
class SearchSession:
    """Complete search session with metadata"""
    session_id: str
    query: str
    search_source: str      # "ArXiv", "Multiple Sources", etc.
    timestamp: datetime
    total_results: int
    avg_y_score: float
    avg_r_score: float
    avg_s_score: float
    avg_n_score: float
    avg_context_risk: float
    top_paper_title: str
    research_domain: str    # Inferred domain category

class YRSNSearchTracker:
    """Track and analyze Y=R+S+N search patterns over time using PostgreSQL"""
    
    def __init__(self, backend_config=None):
        """Initialize search tracker with PostgreSQL database"""
        if not POSTGRES_AVAILABLE:
            raise ImportError("PostgreSQL driver (psycopg2) not available")
            
        self.backend_config = backend_config
        if backend_config:
            self.connection_params = {
                'host': backend_config.settings.postgres.host,
                'port': backend_config.settings.postgres.port,
                'database': backend_config.settings.postgres.database,
                'user': backend_config.settings.postgres.username,
                'password': backend_config.settings.postgres.password,
                'sslmode': backend_config.settings.postgres.ssl_mode
            }
        else:
            # Fallback to environment variables
            self.connection_params = {
                'host': os.getenv('POSTGRES_HOST', 'localhost'),
                'port': int(os.getenv('POSTGRES_PORT', 5432)),
                'database': os.getenv('POSTGRES_DB', 'research_papers_db'),
                'user': os.getenv('POSTGRES_USER', 'postgres'),
                'password': os.getenv('POSTGRES_PASSWORD', ''),
                'sslmode': os.getenv('POSTGRES_SSL', 'prefer')
            }
        
        self._init_database()
    
    def _get_connection(self):
        """Get PostgreSQL connection"""
        return psycopg2.connect(**self.connection_params)
    
    def _init_database(self):
        """Initialize PostgreSQL database with required tables"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    
                    # Search sessions table
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS yrsn_search_sessions (
                            session_id TEXT PRIMARY KEY,
                            query TEXT NOT NULL,
                            search_source TEXT NOT NULL,
                            timestamp TIMESTAMP NOT NULL,
                            total_results INTEGER,
                            avg_y_score FLOAT,
                            avg_r_score FLOAT, 
                            avg_s_score FLOAT,
                            avg_n_score FLOAT,
                            avg_context_risk FLOAT,
                            top_paper_title TEXT,
                            research_domain TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    # Individual results table
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS yrsn_search_results (
                            id SERIAL PRIMARY KEY,
                            session_id TEXT REFERENCES yrsn_search_sessions(session_id),
                            paper_id TEXT,
                            title TEXT,
                            authors JSONB,
                            abstract TEXT,
                            source TEXT,
                            y_score FLOAT,
                            r_score FLOAT,
                            s_score FLOAT,
                            n_score FLOAT,
                            context_risk FLOAT,
                            semantic_relevance FLOAT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    # Research domains trends table
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS yrsn_domain_trends (
                            domain TEXT,
                            query TEXT,
                            date DATE,
                            avg_quality FLOAT,
                            paper_count INTEGER,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (domain, date)
                        )
                    ''')
                    
                    # Create indexes for better performance
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_yrsn_sessions_timestamp 
                        ON yrsn_search_sessions(timestamp DESC)
                    ''')
                    
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_yrsn_sessions_domain 
                        ON yrsn_search_sessions(research_domain)
                    ''')
                    
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_yrsn_trends_domain_date 
                        ON yrsn_domain_trends(domain, date DESC)
                    ''')
                    
        except Exception as e:
            raise Exception(f"Failed to initialize YRSN tracking tables: {e}")
    
    def log_search_session(self, session: SearchSession, results: List[SearchResult]) -> str:
        """Log a complete search session with all results"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    
                    # Insert search session
                    cursor.execute('''
                        INSERT INTO yrsn_search_sessions 
                        (session_id, query, search_source, timestamp, total_results, 
                         avg_y_score, avg_r_score, avg_s_score, avg_n_score, 
                         avg_context_risk, top_paper_title, research_domain)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (session_id) DO UPDATE SET
                        query = EXCLUDED.query,
                        search_source = EXCLUDED.search_source,
                        timestamp = EXCLUDED.timestamp,
                        total_results = EXCLUDED.total_results,
                        avg_y_score = EXCLUDED.avg_y_score,
                        avg_r_score = EXCLUDED.avg_r_score,
                        avg_s_score = EXCLUDED.avg_s_score,
                        avg_n_score = EXCLUDED.avg_n_score,
                        avg_context_risk = EXCLUDED.avg_context_risk,
                        top_paper_title = EXCLUDED.top_paper_title,
                        research_domain = EXCLUDED.research_domain
                    ''', (
                        session.session_id, session.query, session.search_source,
                        session.timestamp, session.total_results,
                        session.avg_y_score, session.avg_r_score, session.avg_s_score,
                        session.avg_n_score, session.avg_context_risk,
                        session.top_paper_title, session.research_domain
                    ))
                    
                    # Insert individual results
                    for result in results:
                        cursor.execute('''
                            INSERT INTO yrsn_search_results 
                            (session_id, paper_id, title, authors, abstract, source,
                             y_score, r_score, s_score, n_score, context_risk, semantic_relevance)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ''', (
                            session.session_id, result.paper_id, result.title,
                            json.dumps(result.authors), result.abstract, result.source,
                            result.y_score, result.r_score, result.s_score,
                            result.n_score, result.context_risk, result.semantic_relevance
                        ))
                    
                    # Update domain trends
                    today = datetime.now().date()
                    cursor.execute('''
                        INSERT INTO yrsn_domain_trends 
                        (domain, query, date, avg_quality, paper_count)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (domain, date) DO UPDATE SET
                        query = EXCLUDED.query,
                        avg_quality = (yrsn_domain_trends.avg_quality + EXCLUDED.avg_quality) / 2,
                        paper_count = yrsn_domain_trends.paper_count + EXCLUDED.paper_count
                    ''', (
                        session.research_domain, session.query, today,
                        session.avg_y_score, session.total_results
                    ))
            
            return session.session_id
            
        except Exception as e:
            raise Exception(f"Failed to log search session: {e}")
    
    def get_search_history(self, limit: int = 50) -> List[Dict]:
        """Get recent search history"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        SELECT session_id, query, search_source, timestamp, total_results,
                               avg_y_score, avg_context_risk, research_domain, top_paper_title
                        FROM yrsn_search_sessions 
                        ORDER BY timestamp DESC 
                        LIMIT %s
                    ''', (limit,))
                    
                    columns = ['session_id', 'query', 'search_source', 'timestamp', 'total_results',
                              'avg_y_score', 'avg_context_risk', 'research_domain', 'top_paper_title']
                    
                    results = []
                    for row in cursor.fetchall():
                        row_dict = dict(zip(columns, row))
                        # Convert timestamp to string for JSON serialization
                        if row_dict['timestamp']:
                            row_dict['timestamp'] = row_dict['timestamp'].isoformat()
                        results.append(row_dict)
                    
                    return results
        except Exception as e:
            raise Exception(f"Failed to get search history: {e}")
    
    def get_domain_trends(self, domain: str = None, days: int = 30) -> List[Dict]:
        """Get trending research domains and quality patterns"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    if domain:
                        cursor.execute('''
                            SELECT domain, date, avg_quality, paper_count, query
                            FROM yrsn_domain_trends 
                            WHERE domain = %s AND date >= CURRENT_DATE - INTERVAL '%s days'
                            ORDER BY date DESC
                        ''', (domain, days))
                    else:
                        cursor.execute('''
                            SELECT domain, date, avg_quality, paper_count, query
                            FROM yrsn_domain_trends 
                            WHERE date >= CURRENT_DATE - INTERVAL '%s days'
                            ORDER BY avg_quality DESC, date DESC
                        ''', (days,))
                    
                    columns = ['domain', 'date', 'avg_quality', 'paper_count', 'query']
                    results = []
                    for row in cursor.fetchall():
                        row_dict = dict(zip(columns, row))
                        # Convert date to string for JSON serialization
                        if row_dict['date']:
                            row_dict['date'] = row_dict['date'].isoformat()
                        results.append(row_dict)
                    
                    return results
        except Exception as e:
            raise Exception(f"Failed to get domain trends: {e}")
    
    def get_quality_analytics(self) -> Dict[str, Any]:
        """Get overall quality analytics across all searches"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    
                    # Overall statistics
                    cursor.execute('''
                        SELECT 
                            COUNT(*) as total_searches,
                            AVG(avg_y_score) as overall_avg_y,
                            AVG(avg_context_risk) as overall_avg_risk,
                            COUNT(DISTINCT research_domain) as unique_domains
                        FROM yrsn_search_sessions
                    ''')
                    row = cursor.fetchone()
                    overall_stats = dict(zip(['total_searches', 'overall_avg_y', 'overall_avg_risk', 'unique_domains'], row))
                    
                    # Top domains by quality
                    cursor.execute('''
                        SELECT research_domain, AVG(avg_y_score) as avg_quality, COUNT(*) as search_count
                        FROM yrsn_search_sessions 
                        WHERE research_domain IS NOT NULL
                        GROUP BY research_domain
                        ORDER BY avg_quality DESC
                        LIMIT 10
                    ''')
                    top_domains = []
                    for row in cursor.fetchall():
                        top_domains.append({
                            'domain': row[0],
                            'avg_quality': row[1], 
                            'search_count': row[2]
                        })
                    
                    # Recent quality trends
                    cursor.execute('''
                        SELECT DATE(timestamp) as date, AVG(avg_y_score) as daily_avg_quality
                        FROM yrsn_search_sessions 
                        WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days'
                        GROUP BY DATE(timestamp)
                        ORDER BY date DESC
                    ''')
                    daily_trends = []
                    for row in cursor.fetchall():
                        daily_trends.append({
                            'date': row[0].isoformat() if row[0] else None,
                            'avg_quality': row[1]
                        })
                    
                    return {
                        'overall_stats': overall_stats,
                        'top_domains': top_domains,
                        'daily_trends': daily_trends
                    }
        except Exception as e:
            raise Exception(f"Failed to get quality analytics: {e}")
    
    def infer_research_domain(self, query: str, paper_titles: List[str]) -> str:
        """Infer research domain from query and paper titles"""
        text = f"{query} {' '.join(paper_titles)}".lower()
        
        # Import domain keywords from centralized config
        try:
            from ui_config import DOMAIN_KEYWORDS
            domain_keywords = DOMAIN_KEYWORDS
        except ImportError:
            # Fallback to local definition if ui_config not available
            domain_keywords = {
                'Machine Learning': ['neural', 'learning', 'ai', 'artificial', 'deep', 'machine', 'algorithm'],
                'Signal Processing': ['signal', 'processing', 'filter', 'frequency', 'audio', 'image'],
                'Quantum Physics': ['quantum', 'physics', 'particle', 'entanglement', 'mechanics'],
                'Mathematics': ['mathematical', 'theorem', 'proof', 'algebra', 'geometry', 'calculus'],
                'Computer Science': ['computer', 'software', 'programming', 'algorithm', 'data'],
                'Biology/Medicine': ['bio', 'medical', 'health', 'disease', 'clinical', 'gene'],
                'Engineering': ['engineering', 'system', 'design', 'control', 'optimization'],
                'Statistics': ['statistical', 'probability', 'regression', 'analysis', 'data'],
                'Context Engineering': ['context', 'prompt', 'engineering', 'rag', 'retrieval', 'augmented', 'generation', 'llm', 'language', 'model', 'contextual', 'collapse', 'quality', 'control', 'information', 'curation']
            }
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return 'General Research'

def generate_session_id() -> str:
    """Generate unique session ID"""
    return f"yrsn_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(datetime.now()) % 10000}"