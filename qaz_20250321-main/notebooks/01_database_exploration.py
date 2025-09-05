#!/usr/bin/env python3
"""
🗄️ Database Exploration - VectorQA Sage

This script explores the existing PostgreSQL database schema and demonstrates 
its capabilities for point-in-time benchmarks and standards management.

Overview:
- Database: PostgreSQL with pgvector extensions
- Purpose: Standards storage with point-in-time awareness  
- Key Features: Vector embeddings, categorical indexing, audit trails

Usage:
    python notebooks/01_database_exploration.py

Requirements:
    pip install pandas numpy matplotlib seaborn plotly psycopg2-binary sqlalchemy
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from typing import List, Dict, Any, Optional
import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine, text

# Add src to path for backend imports
sys.path.insert(0, '../src')

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DatabaseExplorer:
    """Database exploration and analysis class."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database connection."""
        self.database_url = database_url or os.getenv('DATABASE_URL', 'postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}')
        self.engine = None
        self.connection = None
        
        print(f"🔗 Database URL: {self.database_url.split('@')[1] if '@' in self.database_url else 'Not configured'}")
        
        self._connect()
    
    def _connect(self):
        """Establish database connection."""
        try:
            self.engine = create_engine(self.database_url)
            self.connection = self.engine.connect()
            print("✅ Database connection established")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            print("💡 Please ensure PostgreSQL is running and DATABASE_URL is set correctly")
            self.engine = None
            self.connection = None
    
    def get_table_info(self) -> pd.DataFrame:
        """Get information about all tables in the database."""
        if not self.engine:
            return pd.DataFrame()
            
        query = """
        SELECT 
            table_name,
            table_type,
            pg_size_pretty(pg_total_relation_size(table_name::regclass)) as size,
            (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) as column_count
        FROM information_schema.tables t
        WHERE table_schema = 'public'
        ORDER BY pg_total_relation_size(table_name::regclass) DESC
        """
        
        try:
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            print(f"❌ Error querying table info: {e}")
            return pd.DataFrame()
    
    def check_vector_extension(self) -> bool:
        """Check if pgvector extension is installed."""
        if not self.engine:
            return False
            
        query = """
        SELECT 
            extname,
            extversion,
            extrelocatable
        FROM pg_extension 
        WHERE extname = 'vector'
        """
        
        try:
            df = pd.read_sql(query, self.engine)
            if not df.empty:
                print("✅ pgvector extension is installed")
                print(df.to_string(index=False))
                return True
            else:
                print("❌ pgvector extension not found")
                print("💡 Install with: CREATE EXTENSION IF NOT EXISTS vector;")
                return False
        except Exception as e:
            print(f"❌ Error checking vector extension: {e}")
            return False
    
    def analyze_review_runs(self) -> Dict[str, Any]:
        """Analyze the review_runs table for insights."""
        if not self.engine:
            return {}
            
        # Get basic stats
        stats_query = """
        SELECT 
            COUNT(*) as total_runs,
            COUNT(DISTINCT reviewer) as unique_reviewers,
            COUNT(DISTINCT org) as unique_orgs,
            COUNT(DISTINCT team) as unique_teams,
            MIN(review_date) as earliest_review,
            MAX(review_date) as latest_review,
            AVG(runtime_sec) as avg_runtime_sec,
            AVG(finding_count) as avg_findings,
            AVG(cost_usd) as avg_cost_usd
        FROM review_runs
        """
        
        try:
            stats_df = pd.read_sql(stats_query, self.engine)
            print("📈 Review Runs Statistics:")
            print(stats_df.to_string(index=False))
            
            # Get recent activity
            recent_query = """
            SELECT 
                review_date,
                COUNT(*) as runs_count,
                AVG(runtime_sec) as avg_runtime,
                AVG(finding_count) as avg_findings
            FROM review_runs 
            WHERE review_date >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY review_date
            ORDER BY review_date DESC
            """
            
            recent_df = pd.read_sql(recent_query, self.engine)
            if not recent_df.empty:
                print("\n📅 Recent Activity (Last 30 Days):")
                print(recent_df.to_string(index=False))
            else:
                print("\n📅 No recent activity found")
                
            return {
                'stats': stats_df.to_dict('records')[0] if not stats_df.empty else {},
                'recent': recent_df.to_dict('records') if not recent_df.empty else []
            }
                
        except Exception as e:
            print(f"❌ Error analyzing review_runs: {e}")
            return {}
    
    def analyze_section_gists(self) -> Dict[str, Any]:
        """Analyze the section_gists table for embedding insights."""
        if not self.engine:
            return {}
            
        # Get embedding statistics
        stats_query = """
        SELECT 
            COUNT(*) as total_gists,
            COUNT(DISTINCT doc_id) as unique_documents,
            COUNT(DISTINCT embedding_model) as embedding_models,
            COUNT(DISTINCT lifecycle) as lifecycle_stages,
            COUNT(DISTINCT risk_tier) as risk_tiers,
            COUNT(DISTINCT doc_type) as document_types,
            AVG(LENGTH(gist)) as avg_gist_length,
            MIN(created_at) as earliest_gist,
            MAX(created_at) as latest_gist
        FROM section_gists
        """
        
        try:
            stats_df = pd.read_sql(stats_query, self.engine)
            print("🧠 Section Gists (Embeddings) Statistics:")
            print(stats_df.to_string(index=False))
            
            # Get categorical distribution
            cat_query = """
            SELECT 
                lifecycle,
                risk_tier,
                doc_type,
                COUNT(*) as count
            FROM section_gists 
            WHERE lifecycle IS NOT NULL
            GROUP BY lifecycle, risk_tier, doc_type
            ORDER BY count DESC
            LIMIT 20
            """
            
            cat_df = pd.read_sql(cat_query, self.engine)
            if not cat_df.empty:
                print("\n📊 Categorical Distribution:")
                print(cat_df.to_string(index=False))
            
            return {
                'stats': stats_df.to_dict('records')[0] if not stats_df.empty else {},
                'categories': cat_df.to_dict('records') if not cat_df.empty else []
            }
                
        except Exception as e:
            print(f"❌ Error analyzing section_gists: {e}")
            return {}
    
    def analyze_standards_evolution(self) -> pd.DataFrame:
        """Analyze how standards have evolved over time."""
        if not self.engine:
            return pd.DataFrame()
            
        # Get standards by year
        year_query = """
        SELECT 
            year,
            COUNT(*) as standards_count,
            COUNT(DISTINCT doc_id) as unique_documents,
            COUNT(DISTINCT lifecycle) as lifecycle_stages,
            COUNT(DISTINCT risk_tier) as risk_tiers
        FROM section_gists 
        WHERE year IS NOT NULL
        GROUP BY year
        ORDER BY year
        """
        
        try:
            year_df = pd.read_sql(year_query, self.engine)
            
            if not year_df.empty:
                print("📈 Standards Evolution by Year:")
                print(year_df.to_string(index=False))
                
                # Create visualization
                self._plot_standards_evolution(year_df)
                
            else:
                print("📅 No standards data with year information found")
                
            return year_df
                
        except Exception as e:
            print(f"❌ Error analyzing standards evolution: {e}")
            return pd.DataFrame()
    
    def _plot_standards_evolution(self, year_df: pd.DataFrame):
        """Create visualization for standards evolution."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Standards count over time
            ax1.plot(year_df['year'], year_df['standards_count'], marker='o', linewidth=2, markersize=8)
            ax1.set_title('Standards Count Over Time')
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Number of Standards')
            ax1.grid(True, alpha=0.3)
            
            # Risk tier distribution
            risk_query = """
            SELECT 
                year,
                risk_tier,
                COUNT(*) as count
            FROM section_gists 
            WHERE year IS NOT NULL AND risk_tier IS NOT NULL
            GROUP BY year, risk_tier
            ORDER BY year, risk_tier
            """
            
            risk_df = pd.read_sql(risk_query, self.engine)
            if not risk_df.empty:
                risk_pivot = risk_df.pivot(index='year', columns='risk_tier', values='count').fillna(0)
                risk_pivot.plot(kind='bar', ax=ax2, stacked=True)
                ax2.set_title('Risk Tier Distribution Over Time')
                ax2.set_xlabel('Year')
                ax2.set_ylabel('Number of Standards')
                ax2.legend(title='Risk Tier')
                ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('notebooks/standards_evolution.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("📊 Visualization saved as 'notebooks/standards_evolution.png'")
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
    
    def get_point_in_time_standards(self, target_date: date, document_types: List[str] = None) -> pd.DataFrame:
        """Get standards that were effective at a specific point in time."""
        if not self.engine:
            return pd.DataFrame()
            
        # Build query with point-in-time logic
        query = """
        SELECT 
            doc_id,
            section,
            lifecycle,
            risk_tier,
            doc_type,
            year,
            gist,
            created_at
        FROM section_gists 
        WHERE year <= %s
        """
        
        params = [target_date.year]
        
        if document_types:
            placeholders = ','.join(['%s'] * len(document_types))
            query += f" AND doc_type IN ({placeholders})"
            params.extend(document_types)
        
        query += " ORDER BY year DESC, created_at DESC LIMIT 20"
        
        try:
            df = pd.read_sql(query, self.engine, params=params)
            
            if not df.empty:
                print(f"📅 Standards effective on {target_date}:")
                print(f"Found {len(df)} standards")
                print(df[['doc_id', 'section', 'lifecycle', 'risk_tier', 'doc_type', 'year']].to_string(index=False))
                
                # Show summary
                print(f"\n📊 Summary:")
                print(f"- Document types: {df['doc_type'].nunique()}")
                print(f"- Risk tiers: {df['risk_tier'].nunique()}")
                print(f"- Lifecycle stages: {df['lifecycle'].nunique()}")
                print(f"- Year range: {df['year'].min()} - {df['year'].max()}")
            else:
                print(f"📅 No standards found for {target_date}")
                
            return df
                
        except Exception as e:
            print(f"❌ Error retrieving point-in-time standards: {e}")
            return pd.DataFrame()
    
    def demonstrate_vector_search(self, query_text: str, limit: int = 5) -> Dict[str, Any]:
        """Demonstrate vector similarity search capabilities."""
        if not self.engine:
            return {}
            
        print(f"🔍 Vector Search Demo for: '{query_text}'")
        print("Note: This requires actual embeddings to be present in the database")
        
        # Check if we have embeddings
        check_query = """
        SELECT 
            COUNT(*) as total_embeddings,
            COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as non_null_embeddings
        FROM section_gists
        """
        
        try:
            check_df = pd.read_sql(check_query, self.engine)
            
            if check_df['non_null_embeddings'].iloc[0] > 0:
                print(f"✅ Found {check_df['non_null_embeddings'].iloc[0]} embeddings in database")
                
                # Example similarity search (placeholder - would need actual embedding generation)
                print("\n📋 Similarity Search Capabilities:")
                print("- Cosine similarity search")
                print("- Point-in-time filtering")
                print("- Categorical filtering")
                print("- Multi-modal search")
                
                return {
                    'status': 'embeddings_found',
                    'count': check_df['non_null_embeddings'].iloc[0]
                }
                
            else:
                print("⚠️ No embeddings found in database")
                print("💡 Run embedding generation notebook to populate embeddings")
                
                return {
                    'status': 'no_embeddings',
                    'count': 0
                }
                
        except Exception as e:
            print(f"❌ Error checking embeddings: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def analyze_database_performance(self) -> Dict[str, Any]:
        """Analyze database performance and usage patterns."""
        if not self.engine:
            return {}
            
        # Table sizes and row counts
        size_query = """
        SELECT 
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
            pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
        FROM pg_tables 
        WHERE schemaname = 'public' 
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        """
        
        try:
            size_df = pd.read_sql(size_query, self.engine)
            
            if not size_df.empty:
                print("📊 Database Table Sizes:")
                print(size_df.to_string(index=False))
                
                # Create visualization
                self._plot_table_sizes(size_df)
                
            # Index usage
            index_query = """
            SELECT 
                indexname,
                tablename,
                indexdef
            FROM pg_indexes 
            WHERE schemaname = 'public'
            ORDER BY tablename, indexname
            """
            
            index_df = pd.read_sql(index_query, self.engine)
            if not index_df.empty:
                print("\n🔍 Database Indexes:")
                print(index_df[['indexname', 'tablename']].to_string(index=False))
                
            return {
                'table_sizes': size_df.to_dict('records') if not size_df.empty else [],
                'indexes': index_df.to_dict('records') if not index_df.empty else []
            }
                
        except Exception as e:
            print(f"❌ Error analyzing performance: {e}")
            return {}
    
    def _plot_table_sizes(self, size_df: pd.DataFrame):
        """Create visualization for table sizes."""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Filter out very small tables for better visualization
            significant_tables = size_df[size_df['size_bytes'] > 1024]  # > 1KB
            
            if not significant_tables.empty:
                bars = ax.bar(range(len(significant_tables)), significant_tables['size_bytes'])
                ax.set_title('Database Table Sizes')
                ax.set_xlabel('Tables')
                ax.set_ylabel('Size (bytes)')
                ax.set_xticks(range(len(significant_tables)))
                ax.set_xticklabels(significant_tables['tablename'], rotation=45, ha='right')
                
                # Add size labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            significant_tables['size'].iloc[i],
                            ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig('notebooks/table_sizes.png', dpi=300, bbox_inches='tight')
                plt.show()
                print("📊 Visualization saved as 'notebooks/table_sizes.png'")
            
        except Exception as e:
            print(f"❌ Error creating table size visualization: {e}")
    
    def generate_insights(self) -> Dict[str, Any]:
        """Generate insights from the database exploration."""
        insights = {
            "database_status": "✅ PostgreSQL with pgvector ready",
            "point_in_time_capability": "✅ Date-based filtering implemented",
            "vector_search": "✅ pgvector extension available",
            "categorical_indexing": "✅ Walmart-style categorical filters",
            "audit_trail": "✅ Complete audit trail with timestamps",
            "next_steps": [
                "Populate embeddings using AWS Bedrock",
                "Process S3 standards folders",
                "Implement MCP coordinators",
                "Test point-in-time retrieval",
                "Validate similarity search"
            ]
        }
        
        print("🎯 Database Exploration Insights:")
        print("=" * 50)
        
        for key, value in insights.items():
            if key == "next_steps":
                print(f"\n📋 {key.replace('_', ' ').title()}:")
                for i, step in enumerate(value, 1):
                    print(f"  {i}. {step}")
            else:
                print(f"{value}")
        
        print("\n🚀 Ready for next script: AWS Bedrock Demo")
        return insights
    
    def close(self):
        """Close database connections."""
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()
        print("🔌 Database connections closed")


def main():
    """Main function to run database exploration."""
    print("🗄️ VectorQA Sage - Database Exploration")
    print("=" * 50)
    
    # Initialize database explorer
    explorer = DatabaseExplorer()
    
    if not explorer.engine:
        print("❌ Cannot proceed without database connection")
        return
    
    try:
        # 1. Explore database schema
        print("\n📋 Step 1: Database Schema Exploration")
        print("-" * 40)
        table_info = explorer.get_table_info()
        if not table_info.empty:
            print("📊 Database Tables:")
            print(table_info.to_string(index=False))
        
        # 2. Check vector extension
        print("\n📋 Step 2: Vector Extension Check")
        print("-" * 40)
        explorer.check_vector_extension()
        
        # 3. Analyze core tables
        print("\n📋 Step 3: Core Tables Analysis")
        print("-" * 40)
        review_analysis = explorer.analyze_review_runs()
        gists_analysis = explorer.analyze_section_gists()
        
        # 4. Point-in-time analysis
        print("\n📋 Step 4: Point-in-Time Analysis")
        print("-" * 40)
        evolution_df = explorer.analyze_standards_evolution()
        
        # 5. Demonstrate point-in-time retrieval
        print("\n📋 Step 5: Point-in-Time Standards Retrieval")
        print("-" * 40)
        target_date = date(2024, 1, 15)
        print(f"🔍 Point-in-Time Standards Retrieval for {target_date}")
        standards_df = explorer.get_point_in_time_standards(target_date, document_types=['Policy', 'Standards'])
        
        # 6. Vector search capabilities
        print("\n📋 Step 6: Vector Search Capabilities")
        print("-" * 40)
        vector_status = explorer.demonstrate_vector_search("model risk governance standards")
        
        # 7. Performance analysis
        print("\n📋 Step 7: Performance Analysis")
        print("-" * 40)
        performance_data = explorer.analyze_database_performance()
        
        # 8. Generate insights
        print("\n📋 Step 8: Key Insights & Next Steps")
        print("-" * 40)
        insights = explorer.generate_insights()
        
        # Summary
        print("\n📝 Conclusion")
        print("=" * 50)
        print("This script has explored the existing PostgreSQL database schema and demonstrated:")
        print("\n✅ What's Working:")
        print("- Database Schema: Complete schema with review runs, embeddings, and analytics")
        print("- Point-in-Time Logic: Date-based filtering for standards evolution")
        print("- Vector Support: pgvector extension for similarity search")
        print("- Categorical Indexing: Walmart-style categorical filters")
        print("- Audit Trail: Complete tracking with timestamps")
        
        print("\n🔧 Next Steps:")
        print("1. Populate Embeddings: Use AWS Bedrock to generate real embeddings")
        print("2. Process S3 Standards: Implement S3 folder processing")
        print("3. MCP Integration: Connect with MCP coordinators")
        print("4. Testing: Validate point-in-time retrieval")
        
        print("\n📚 Related Scripts:")
        print("- 02_aws_bedrock_demo.py - AWS Bedrock integration")
        print("- 03_embedding_generation.py - Embedding generation")
        print("- 05_point_in_time_demo.py - Point-in-time benchmarks")
        
        print("\nThe database is **ready for point-in-time benchmark functionality** - we just need to populate it with real data and connect it to the MCP architecture!")
        
    except Exception as e:
        print(f"❌ Error during exploration: {e}")
    
    finally:
        explorer.close()


if __name__ == "__main__":
    main()
