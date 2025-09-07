#!/usr/bin/env python3
"""
Rudy's Embedding Database Monitor
=================================

Comprehensive embedding database analysis tool for TidyLLM.
Generates detailed reports on embedding creation, storage, and processing status.

Usage:
    python rudy_test_embeddings.py                    # Full report
    python rudy_test_embeddings.py --summary          # Quick summary only  
    python rudy_test_embeddings.py --watch            # Continuous monitoring
    python rudy_test_embeddings.py --tables           # Show all embedding tables
"""

import sys
import time
import yaml
import argparse
from datetime import datetime, timedelta
from pathlib import Path

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    DB_AVAILABLE = True
except ImportError:
    print("[FAIL] psycopg2 not installed. Run: pip install psycopg2-binary")
    DB_AVAILABLE = False

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class EmbeddingDatabaseMonitor:
    """Monitor and analyze embedding database status."""
    
    def __init__(self, config_path=None):
        """Initialize with database configuration."""
        self.config_path = config_path or project_root / "tidyllm/admin/settings.yaml"
        self.db_config = self._load_db_config()
        self.conn = None
    
    def _load_db_config(self):
        """Load database configuration from settings."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config['postgres']
        except Exception as e:
            raise Exception(f"Failed to load database config: {e}")
    
    def connect(self):
        """Connect to PostgreSQL database."""
        if not DB_AVAILABLE:
            raise Exception("psycopg2 not available")
        
        try:
            self.conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['db_name'],
                user=self.db_config['db_user'],
                password=self.db_config['db_password'],
                sslmode=self.db_config.get('ssl_mode', 'require'),
                cursor_factory=RealDictCursor
            )
            return True
        except Exception as e:
            print(f"[FAIL] Database connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def get_embedding_tables(self):
        """Get list of all embedding-related tables."""
        cursor = self.conn.cursor()
        
        # Find tables with 'embed' or 'vector' in name
        cursor.execute("""
            SELECT table_name, 
                   (SELECT COUNT(*) FROM information_schema.columns 
                    WHERE table_name = t.table_name 
                    AND (column_name ILIKE '%embed%' OR column_name ILIKE '%vector%')) as embedding_cols
            FROM information_schema.tables t
            WHERE table_schema = 'public' 
            AND (table_name ILIKE '%embed%' OR table_name ILIKE '%vector%' OR table_name ILIKE '%chunk%')
            ORDER BY table_name;
        """)
        
        return cursor.fetchall()
    
    def get_document_chunks_analysis(self):
        """Analyze document_chunks table in detail."""
        cursor = self.conn.cursor()
        
        # Basic statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(DISTINCT doc_id) as unique_documents,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL) as chunks_with_embeddings,
                MIN(created_at) as earliest_date,
                MAX(created_at) as latest_date,
                MIN(DATE(created_at)) as earliest_day,
                MAX(DATE(created_at)) as latest_day
            FROM document_chunks;
        """)
        
        basic_stats = cursor.fetchone()
        
        # By embedding model
        cursor.execute("""
            SELECT 
                embedding_model,
                COUNT(*) as total_chunks,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL) as with_embeddings,
                ROUND(AVG(char_count)) as avg_char_count,
                ROUND(AVG(token_estimate)) as avg_token_estimate,
                MIN(created_at) as first_created,
                MAX(created_at) as last_created
            FROM document_chunks
            GROUP BY embedding_model
            ORDER BY total_chunks DESC;
        """)
        
        model_stats = cursor.fetchall()
        
        # By date
        cursor.execute("""
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as chunks_created,
                COUNT(DISTINCT doc_id) as docs_processed,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL) as embeddings_created,
                STRING_AGG(DISTINCT SUBSTRING(doc_id, 1, 40), ', ' ORDER BY SUBSTRING(doc_id, 1, 40)) as sample_docs
            FROM document_chunks
            GROUP BY DATE(created_at)
            ORDER BY date DESC;
        """)
        
        date_stats = cursor.fetchall()
        
        # Recent documents
        cursor.execute("""
            SELECT DISTINCT 
                doc_id,
                created_at,
                COUNT(*) as chunk_count,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL) as embedding_count,
                embedding_model
            FROM document_chunks
            GROUP BY doc_id, created_at, embedding_model
            ORDER BY created_at DESC
            LIMIT 15;
        """)
        
        recent_docs = cursor.fetchall()
        
        return {
            'basic': basic_stats,
            'by_model': model_stats,
            'by_date': date_stats,
            'recent_docs': recent_docs
        }
    
    def get_embedding_table_counts(self):
        """Get record counts for all embedding tables."""
        cursor = self.conn.cursor()
        
        embedding_tables = ['chunk_embeddings', 'paper_embeddings', 'document_chunks']
        results = {}
        
        for table in embedding_tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table};")
                count = cursor.fetchone()[0]
                results[table] = count
            except Exception as e:
                results[table] = f"ERROR: {str(e)[:50]}"
        
        return results
    
    def check_embedding_dimensions(self):
        """Check embedding dimensions and sample data."""
        cursor = self.conn.cursor()
        
        # Check if we can get embedding dimensions
        try:
            cursor.execute("""
                SELECT 
                    chunk_id,
                    embedding IS NOT NULL as has_embedding,
                    char_count,
                    token_estimate,
                    embedding_model,
                    created_at
                FROM document_chunks 
                WHERE embedding IS NOT NULL 
                LIMIT 5;
            """)
            
            samples = cursor.fetchall()
            return samples
        except Exception as e:
            return f"Error checking embeddings: {e}"
    
    def generate_summary_report(self):
        """Generate a quick summary report."""
        if not self.connect():
            return
        
        try:
            print("[SEARCH] RUDY'S EMBEDDING QUICK SUMMARY")
            print("=" * 50)
            
            # Get basic counts
            table_counts = self.get_embedding_table_counts()
            analysis = self.get_document_chunks_analysis()
            
            print(f"[STATS] RECORD COUNTS:")
            for table, count in table_counts.items():
                print(f"   {table:<20}: {count:>10}")
            
            basic = analysis['basic']
            print(f"\n[DATE] DATE RANGE:")
            print(f"   From: {basic['earliest_day']}")
            print(f"   To:   {basic['latest_day']}")
            
            print(f"\n[OK] EMBEDDING STATUS:")
            print(f"   Total chunks: {basic['total_chunks']:,}")
            print(f"   With embeddings: {basic['chunks_with_embeddings']:,}")
            print(f"   Percentage: {(basic['chunks_with_embeddings']/max(basic['total_chunks'],1)*100):.1f}%")
            
        finally:
            self.disconnect()
    
    def generate_full_report(self):
        """Generate comprehensive embedding database report."""
        if not self.connect():
            return
        
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("[SEARCH] RUDY'S COMPREHENSIVE EMBEDDING DATABASE REPORT")
            print("=" * 70)
            print(f"Generated: {now}")
            print(f"Database: {self.db_config['db_name']}@{self.db_config['host']}")
            print()
            
            # 1. Embedding Table Counts
            print("[STATS] EMBEDDING TABLE RECORD COUNTS:")
            print("-" * 50)
            table_counts = self.get_embedding_table_counts()
            
            total_embeddings = 0
            for table, count in table_counts.items():
                if isinstance(count, int):
                    if 'embedding' in table:
                        total_embeddings += count
                    print(f"   {table:<25}: {count:,} records")
                else:
                    print(f"   {table:<25}: {count}")
            
            print(f"   {'TOTAL EMBEDDINGS':<25}: {total_embeddings:,} records")
            print()
            
            # 2. Document Chunks Analysis
            analysis = self.get_document_chunks_analysis()
            basic = analysis['basic']
            
            print("[DATA] DOCUMENT_CHUNKS DETAILED ANALYSIS:")
            print("-" * 50)
            print(f"   Total chunks: {basic['total_chunks']:,}")
            print(f"   Unique documents: {basic['unique_documents']:,}")
            print(f"   Chunks with embeddings: {basic['chunks_with_embeddings']:,}")
            print(f"   Embedding percentage: {(basic['chunks_with_embeddings']/max(basic['total_chunks'],1)*100):.1f}%")
            print(f"   Date range: {basic['earliest_day']} to {basic['latest_day']}")
            print()
            
            # 3. By Embedding Model
            print("[MODEL] BY EMBEDDING MODEL:")
            print("-" * 50)
            for model in analysis['by_model']:
                print(f"   Model: {model['embedding_model']}")
                print(f"      Chunks: {model['total_chunks']:,} | With embeddings: {model['with_embeddings']:,}")
                print(f"      Avg chars: {model['avg_char_count'] or 0:.0f} | Avg tokens: {model['avg_token_estimate'] or 0:.0f}")
                print(f"      Period: {model['first_created'].date()} to {model['last_created'].date()}")
                print()
            
            # 4. By Date
            print("[DATE] PROCESSING BY DATE:")
            print("-" * 50)
            for date_stat in analysis['by_date']:
                print(f"   {date_stat['date']}: {date_stat['chunks_created']:>3} chunks, "
                      f"{date_stat['docs_processed']:>2} docs, {date_stat['embeddings_created']:>3} embeddings")
                if date_stat['sample_docs']:
                    docs = date_stat['sample_docs'][:60] + "..." if len(date_stat['sample_docs']) > 60 else date_stat['sample_docs']
                    print(f"      Documents: {docs}")
            print()
            
            # 5. Recent Documents
            print("[DOCS] RECENT DOCUMENTS:")
            print("-" * 50)
            for doc in analysis['recent_docs']:
                embed_status = f"{doc['embedding_count']}/{doc['chunk_count']} embeddings"
                print(f"   {doc['created_at']}: {doc['doc_id'][:50]}")
                print(f"      {doc['chunk_count']} chunks | {embed_status} | {doc['embedding_model']}")
            print()
            
            # 6. Sample Embeddings
            print("[EMBED] SAMPLE EMBEDDINGS:")
            print("-" * 50)
            samples = self.check_embedding_dimensions()
            if isinstance(samples, list) and samples:
                for sample in samples:
                    print(f"   {sample['chunk_id'][:40]}: "
                          f"{sample['char_count']} chars, {sample['token_estimate']} tokens")
                    print(f"      Model: {sample['embedding_model']} | Created: {sample['created_at']}")
            elif samples:
                print(f"   {samples}")
            else:
                print("   No embeddings found with vector data")
            print()
            
            # 7. Embedding Tables Discovery
            print("[SEARCH] ALL EMBEDDING-RELATED TABLES:")
            print("-" * 50)
            embedding_tables = self.get_embedding_tables()
            for table in embedding_tables:
                cursor = self.conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {table['table_name']};")
                count = cursor.fetchone()[0]
                print(f"   {table['table_name']:<25}: {count:,} records ({table['embedding_cols']} embedding columns)")
            
        except Exception as e:
            print(f"[FAIL] Error generating report: {e}")
        finally:
            self.disconnect()
    
    def watch_mode(self, interval=30):
        """Continuous monitoring mode."""
        print(f"[WATCH] RUDY'S EMBEDDING WATCH MODE (every {interval}s)")
        print("Press Ctrl+C to stop")
        print("=" * 50)
        
        try:
            while True:
                self.generate_summary_report()
                print(f"\n[TIME] Next update in {interval} seconds...")
                time.sleep(interval)
                print("\n" + "=" * 50)
        except KeyboardInterrupt:
            print("\n[BYE] Watch mode stopped")
    
    def show_tables(self):
        """Show all embedding-related tables."""
        if not self.connect():
            return
        
        try:
            print("[DATA] ALL EMBEDDING & VECTOR TABLES")
            print("=" * 50)
            
            embedding_tables = self.get_embedding_tables()
            if not embedding_tables:
                print("No embedding tables found")
                return
            
            for table in embedding_tables:
                cursor = self.conn.cursor()
                
                # Get record count
                cursor.execute(f"SELECT COUNT(*) FROM {table['table_name']};")
                count = cursor.fetchone()[0]
                
                # Get structure
                cursor.execute(f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = '{table['table_name']}'
                    AND (column_name ILIKE '%embed%' OR column_name ILIKE '%vector%')
                    ORDER BY ordinal_position;
                """)
                embedding_columns = cursor.fetchall()
                
                print(f"\n[TABLE]  {table['table_name']} ({count:,} records)")
                if embedding_columns:
                    for col in embedding_columns:
                        nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                        print(f"   └─ {col['column_name']:<20} {col['data_type']:<15} {nullable}")
                else:
                    print("   └─ No embedding columns found")
        finally:
            self.disconnect()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Rudy's Embedding Database Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rudy_test_embeddings.py                 # Full report
  python rudy_test_embeddings.py --summary       # Quick summary
  python rudy_test_embeddings.py --watch         # Monitor continuously
  python rudy_test_embeddings.py --tables        # Show all tables
        """
    )
    
    parser.add_argument('--summary', action='store_true', 
                       help='Generate quick summary only')
    parser.add_argument('--watch', action='store_true',
                       help='Continuous monitoring mode')
    parser.add_argument('--interval', type=int, default=30,
                       help='Watch mode interval in seconds (default: 30)')
    parser.add_argument('--tables', action='store_true',
                       help='Show all embedding-related tables')
    parser.add_argument('--config', type=str,
                       help='Path to settings.yaml file')
    
    args = parser.parse_args()
    
    if not DB_AVAILABLE:
        print("[FAIL] Database not available. Install with: pip install psycopg2-binary")
        return False
    
    try:
        monitor = EmbeddingDatabaseMonitor(config_path=args.config)
        
        if args.tables:
            monitor.show_tables()
        elif args.watch:
            monitor.watch_mode(args.interval)
        elif args.summary:
            monitor.generate_summary_report()
        else:
            monitor.generate_full_report()
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)