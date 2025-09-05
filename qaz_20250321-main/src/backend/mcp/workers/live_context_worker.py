#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Context Worker

Worker that integrates live database data with document analysis to provide
temporal context and current event correlation.
"""

import psycopg2
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from psycopg2.extras import RealDictCursor

from .base_worker import BaseWorker
from ..protocol.message_protocol import MCPMessage


class LiveContextWorker(BaseWorker):
    """Worker for integrating live database data with document analysis"""
    
    def __init__(self):
        super().__init__("live_context_worker", "live_data_integration")
        self.database_url = None
        self._load_database_config()
    
    def _load_database_config(self):
        """Load database configuration"""
        import os
        from dotenv import load_dotenv
        
        load_dotenv('src/backend/config/credentials.env')
        self.database_url = os.getenv('DATABASE_URL')
        
        if not self.database_url:
            self.logger.warning("DATABASE_URL not found in environment")
    
    def process_task(self, message: MCPMessage) -> Dict[str, Any]:
        """Process document analysis and integrate with live data"""
        payload = message.payload
        document_analysis = payload.get('document_analysis', {})
        
        if not self.database_url:
            return {
                'success': False,
                'error': 'Database connection not configured',
                'confidence_score': 0.0
            }
        
        try:
            # Extract key topics from document analysis
            key_topics = self._extract_key_topics(document_analysis)
            
            # Query live database for relevant events
            live_events = self._query_live_events(key_topics)
            
            # Query daily metrics
            daily_metrics = self._query_daily_metrics(key_topics)
            
            # Query recent review findings
            review_findings = self._query_review_findings(key_topics)
            
            # Provide temporal context
            temporal_context = self._analyze_temporal_context(
                document_analysis, live_events, daily_metrics, review_findings
            )
            
            return {
                'success': True,
                'confidence_score': 0.9,
                'processing': {
                    'key_topics': key_topics,
                    'live_events': live_events,
                    'daily_metrics': daily_metrics,
                    'review_findings': review_findings,
                    'temporal_context': temporal_context,
                    'integration_timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in live context processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'confidence_score': 0.0
            }
    
    def _extract_key_topics(self, document_analysis: Dict[str, Any]) -> List[str]:
        """Extract key topics from document analysis for database queries"""
        topics = []
        
        # Extract from claims (handle both string and dict formats)
        claims = document_analysis.get('claims', [])
        for claim in claims:
            if isinstance(claim, str):
                claim_text = claim.lower()
            else:
                claim_text = claim.get('text', '').lower()
            
            # Extract key terms (simplified - could use NLP)
            if 'revenue' in claim_text or 'sales' in claim_text:
                topics.append('revenue')
            if 'risk' in claim_text or 'compliance' in claim_text:
                topics.append('compliance')
            if 'performance' in claim_text or 'metrics' in claim_text:
                topics.append('performance')
            if 'process' in claim_text or 'workflow' in claim_text:
                topics.append('process')
        
        # Extract from evidence (handle both string and dict formats)
        evidence = document_analysis.get('evidence', [])
        for ev in evidence:
            if isinstance(ev, str):
                ev_text = ev.lower()
            else:
                ev_text = ev.get('text', '').lower()
            
            if 'data' in ev_text or 'analytics' in ev_text:
                topics.append('analytics')
        
        # Extract from references (handle both string and dict formats)
        references = document_analysis.get('references', [])
        for ref in references:
            if isinstance(ref, str):
                ref_text = ref.lower()
            else:
                ref_text = ref.get('text', '').lower()
            
            if 'api' in ref_text or 'integration' in ref_text:
                topics.append('integration')
        
        # Also check key_topics if provided directly
        if 'key_topics' in document_analysis:
            topics.extend(document_analysis['key_topics'])
        
        # Remove duplicates and return
        return list(set(topics)) if topics else ['general']
    
    def _query_live_events(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Query live events from events_raw table or generate mock data"""
        if not topics:
            return []
        
        try:
            # First try to query real database
            with psycopg2.connect(self.database_url) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Query recent events related to topics
                    query = """
                    SELECT ts, user_id, org, team, process, event, payload
                    FROM events_raw 
                    WHERE ts > NOW() - INTERVAL '30 days'
                    AND (
                        process = ANY(%s) 
                        OR event = ANY(%s)
                        OR payload::text ILIKE ANY(%s)
                    )
                    ORDER BY ts DESC
                    LIMIT 100
                    """
                    
                    # Create topic patterns for payload search
                    topic_patterns = [f'%{topic}%' for topic in topics]
                    
                    cur.execute(query, (topics, topics, topic_patterns))
                    events = cur.fetchall()
                    
                    if events:
                        return [dict(event) for event in events]
                    
        except Exception as e:
            self.logger.warning(f"Database query failed, using mock data: {e}")
        
        # Generate mock stock data if no real data found
        return self._generate_mock_stock_events(topics)
    
    def _generate_mock_stock_events(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Generate mock stock market events for testing"""
        import random
        from datetime import datetime, timedelta
        
        mock_events = []
        stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
        
        for i in range(random.randint(5, 15)):
            # Generate random timestamp in last 30 days
            days_ago = random.randint(0, 30)
            hours_ago = random.randint(0, 23)
            minutes_ago = random.randint(0, 59)
            
            timestamp = datetime.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
            
            # Generate mock event based on topics
            if 'revenue' in topics:
                event_type = random.choice(['revenue_beat', 'earnings_report', 'sales_update'])
                symbol = random.choice(stock_symbols)
                mock_events.append({
                    'ts': timestamp,
                    'user_id': f'analyst_{random.randint(1, 10)}',
                    'org': 'MOCK_FINANCIAL',
                    'team': 'equity_research',
                    'process': 'earnings_analysis',
                    'event': event_type,
                    'payload': {
                        'symbol': symbol,
                        'revenue_change': f"{random.choice(['+', '-'])}{random.randint(5, 25)}%",
                        'amount': f"${random.randint(10, 100)}M",
                        'note': f"Mock {event_type} for {symbol}"
                    }
                })
            
            if 'compliance' in topics:
                event_type = random.choice(['regulatory_update', 'compliance_alert', 'audit_result'])
                mock_events.append({
                    'ts': timestamp,
                    'user_id': f'compliance_{random.randint(1, 5)}',
                    'org': 'MOCK_REGULATORY',
                    'team': 'compliance',
                    'process': 'regulatory_monitoring',
                    'event': event_type,
                    'payload': {
                        'severity': random.choice(['low', 'medium', 'high']),
                        'regulation': random.choice(['SOX', 'GDPR', 'CCPA', 'Basel III']),
                        'status': random.choice(['pending', 'resolved', 'escalated']),
                        'note': f"Mock {event_type} - {random.choice(['SOX', 'GDPR', 'CCPA', 'Basel III'])} compliance"
                    }
                })
            
            if 'performance' in topics:
                event_type = random.choice(['performance_alert', 'metric_update', 'kpi_report'])
                mock_events.append({
                    'ts': timestamp,
                    'user_id': f'ops_{random.randint(1, 8)}',
                    'org': 'MOCK_OPERATIONS',
                    'team': 'operations',
                    'process': 'performance_monitoring',
                    'event': event_type,
                    'payload': {
                        'metric': random.choice(['response_time', 'throughput', 'error_rate', 'availability']),
                        'value': random.randint(80, 100),
                        'threshold': 90,
                        'status': random.choice(['normal', 'warning', 'critical']),
                        'note': f"Mock {event_type} - {random.choice(['response_time', 'throughput', 'error_rate', 'availability'])} at {random.randint(80, 100)}%"
                    }
                })
        
        # Sort by timestamp (most recent first)
        mock_events.sort(key=lambda x: x['ts'], reverse=True)
        return mock_events
    
    def _query_daily_metrics(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Query daily metrics from events_daily table or generate mock data"""
        if not topics:
            return []
        
        try:
            # First try to query real database
            with psycopg2.connect(self.database_url) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Query recent daily metrics
                    query = """
                    SELECT day, org, team, process, metric, value
                    FROM events_daily 
                    WHERE day > CURRENT_DATE - INTERVAL '30 days'
                    AND (
                        process = ANY(%s)
                        OR metric = ANY(%s)
                    )
                    ORDER BY day DESC, value DESC
                    LIMIT 50
                    """
                    
                    cur.execute(query, (topics, topics))
                    metrics = cur.fetchall()
                    
                    if metrics:
                        return [dict(metric) for metric in metrics]
                    
        except Exception as e:
            self.logger.warning(f"Database query failed, using mock data: {e}")
        
        # Generate mock daily metrics if no real data found
        return self._generate_mock_daily_metrics(topics)
    
    def _generate_mock_daily_metrics(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Generate mock daily metrics for testing"""
        import random
        from datetime import datetime, timedelta
        
        mock_metrics = []
        
        for i in range(random.randint(10, 25)):
            # Generate random date in last 30 days
            days_ago = random.randint(0, 30)
            date = datetime.now().date() - timedelta(days=days_ago)
            
            if 'revenue' in topics:
                mock_metrics.append({
                    'day': date,
                    'org': 'MOCK_FINANCIAL',
                    'team': 'sales',
                    'process': 'revenue_tracking',
                    'metric': 'daily_revenue',
                    'value': random.randint(100000, 500000)
                })
            
            if 'compliance' in topics:
                mock_metrics.append({
                    'day': date,
                    'org': 'MOCK_REGULATORY',
                    'team': 'compliance',
                    'process': 'compliance_monitoring',
                    'metric': 'compliance_score',
                    'value': random.randint(85, 100)
                })
            
            if 'performance' in topics:
                mock_metrics.append({
                    'day': date,
                    'org': 'MOCK_OPERATIONS',
                    'team': 'operations',
                    'process': 'performance_monitoring',
                    'metric': 'system_uptime',
                    'value': random.randint(95, 100)
                })
        
        # Sort by date (most recent first)
        mock_metrics.sort(key=lambda x: x['day'], reverse=True)
        return mock_metrics
    
    def _query_review_findings(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Query recent review findings or generate mock data"""
        if not topics:
            return []
        
        try:
            # First try to query real database
            with psycopg2.connect(self.database_url) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Query recent review findings
                    query = """
                    SELECT run_id, rule_id, stage, severity, status, evidence, rule_source
                    FROM review_findings 
                    WHERE rule_source = ANY(%s)
                    ORDER BY run_id DESC
                    LIMIT 50
                    """
                    
                    cur.execute(query, (topics,))
                    findings = cur.fetchall()
                    
                    if findings:
                        return [dict(finding) for finding in findings]
                    
        except Exception as e:
            self.logger.warning(f"Database query failed, using mock data: {e}")
        
        # Generate mock review findings if no real data found
        return self._generate_mock_review_findings(topics)
    
    def _generate_mock_review_findings(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Generate mock review findings for testing"""
        import random
        
        mock_findings = []
        
        for i in range(random.randint(3, 8)):
            if 'revenue' in topics:
                mock_findings.append({
                    'run_id': random.randint(1000, 9999),
                    'rule_id': f'REV_{random.randint(1, 10)}',
                    'stage': 'validation',
                    'severity': random.choice(['low', 'medium', 'high']),
                    'status': random.choice(['open', 'resolved', 'escalated']),
                    'evidence': {
                        'type': 'revenue_validation',
                        'amount': f"${random.randint(10000, 100000)}",
                        'threshold': "$50000",
                        'note': "Mock revenue validation finding"
                    },
                    'rule_source': 'revenue'
                })
            
            if 'compliance' in topics:
                mock_findings.append({
                    'run_id': random.randint(1000, 9999),
                    'rule_id': f'COMP_{random.randint(1, 15)}',
                    'stage': 'audit',
                    'severity': random.choice(['low', 'medium', 'high']),
                    'status': random.choice(['open', 'resolved', 'escalated']),
                    'evidence': {
                        'type': 'compliance_check',
                        'regulation': random.choice(['SOX', 'GDPR', 'CCPA']),
                        'compliance_score': random.randint(70, 100),
                        'note': f"Mock {random.choice(['SOX', 'GDPR', 'CCPA'])} compliance finding"
                    },
                    'rule_source': 'compliance'
                })
            
            if 'performance' in topics:
                mock_findings.append({
                    'run_id': random.randint(1000, 9999),
                    'rule_id': f'PERF_{random.randint(1, 12)}',
                    'stage': 'monitoring',
                    'severity': random.choice(['low', 'medium', 'high']),
                    'status': random.choice(['open', 'resolved', 'escalated']),
                    'evidence': {
                        'type': 'performance_check',
                        'metric': random.choice(['response_time', 'throughput', 'error_rate']),
                        'value': random.randint(80, 100),
                        'threshold': 90,
                        'note': f"Mock {random.choice(['response_time', 'throughput', 'error_rate'])} performance finding"
                    },
                    'rule_source': 'performance'
                })
        
        return mock_findings
    
    def _analyze_temporal_context(
        self, 
        document_analysis: Dict[str, Any],
        live_events: List[Dict[str, Any]],
        daily_metrics: List[Dict[str, Any]],
        review_findings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze temporal context between document and live data"""
        
        context = {
            'document_age': self._calculate_document_age(document_analysis),
            'event_frequency': self._analyze_event_frequency(live_events),
            'metric_trends': self._analyze_metric_trends(daily_metrics),
            'finding_patterns': self._analyze_finding_patterns(review_findings),
            'relevance_score': self._calculate_relevance_score(
                document_analysis, live_events, daily_metrics, review_findings
            ),
            'temporal_insights': self._generate_temporal_insights(
                document_analysis, live_events, daily_metrics, review_findings
            )
        }
        
        return context
    
    def _calculate_document_age(self, document_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate document age and temporal relevance"""
        # This would typically come from document metadata
        # For now, we'll use a placeholder
        return {
            'estimated_age_days': 30,  # Placeholder
            'temporal_relevance': 'recent',
            'context_freshness': 'high'
        }
    
    def _analyze_event_frequency(self, live_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze frequency and patterns in live events"""
        if not live_events:
            return {'total_events': 0, 'frequency': 'low', 'patterns': []}
        
        # Group events by process
        process_counts = {}
        for event in live_events:
            process = event.get('process', 'unknown')
            process_counts[process] = process_counts.get(process, 0) + 1
        
        return {
            'total_events': len(live_events),
            'frequency': 'high' if len(live_events) > 50 else 'medium' if len(live_events) > 10 else 'low',
            'top_processes': sorted(process_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'recent_activity': len([e for e in live_events if e.get('ts') > datetime.now() - timedelta(days=7)])
        }
    
    def _analyze_metric_trends(self, daily_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in daily metrics"""
        if not daily_metrics:
            return {'total_metrics': 0, 'trends': [], 'insights': []}
        
        # Group by metric and analyze trends
        metric_data = {}
        for metric in daily_metrics:
            metric_name = metric.get('metric', 'unknown')
            if metric_name not in metric_data:
                metric_data[metric_name] = []
            metric_data[metric_name].append(metric.get('value', 0))
        
        trends = []
        for metric_name, values in metric_data.items():
            if len(values) > 1:
                trend = 'increasing' if values[-1] > values[0] else 'decreasing' if values[-1] < values[0] else 'stable'
                trends.append({
                    'metric': metric_name,
                    'trend': trend,
                    'current_value': values[-1],
                    'change_percent': ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                })
        
        return {
            'total_metrics': len(daily_metrics),
            'trends': trends,
            'insights': [f"{t['metric']} is {t['trend']}" for t in trends[:3]]
        }
    
    def _analyze_finding_patterns(self, review_findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in review findings"""
        if not review_findings:
            return {'total_findings': 0, 'severity_distribution': {}, 'patterns': []}
        
        # Analyze severity distribution
        severity_counts = {}
        for finding in review_findings:
            severity = finding.get('severity', 'unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_findings': len(review_findings),
            'severity_distribution': severity_counts,
            'patterns': [
                f"Most common severity: {max(severity_counts.items(), key=lambda x: x[1])[0]}" if severity_counts else "No patterns found"
            ]
        }
    
    def _calculate_relevance_score(
        self,
        document_analysis: Dict[str, Any],
        live_events: List[Dict[str, Any]],
        daily_metrics: List[Dict[str, Any]],
        review_findings: List[Dict[str, Any]]
    ) -> float:
        """Calculate relevance score between document and live data"""
        score = 0.0
        
        # Base score from document analysis
        if document_analysis.get('claims'):
            score += 0.3
        if document_analysis.get('evidence'):
            score += 0.2
        
        # Live data relevance
        if live_events:
            score += min(0.3, len(live_events) / 100.0)
        if daily_metrics:
            score += min(0.1, len(daily_metrics) / 50.0)
        if review_findings:
            score += min(0.1, len(review_findings) / 50.0)
        
        return min(1.0, score)
    
    def _generate_temporal_insights(
        self,
        document_analysis: Dict[str, Any],
        live_events: List[Dict[str, Any]],
        daily_metrics: List[Dict[str, Any]],
        review_findings: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate temporal insights comparing document with live data"""
        insights = []
        
        # Check if we're using mock data
        using_mock_data = False
        if live_events and any('MOCK_' in str(event.get('org', '')) for event in live_events):
            using_mock_data = True
            insights.append("⚠️ Using MOCK data for demonstration purposes")
        
        # Document claims vs live events
        if document_analysis.get('claims') and live_events:
            if using_mock_data:
                insights.append(f"Document contains {len(document_analysis['claims'])} claims, with {len(live_events)} related MOCK events in the past 30 days")
            else:
                insights.append(f"Document contains {len(document_analysis['claims'])} claims, with {len(live_events)} related live events in the past 30 days")
        
        # Metric trends
        if daily_metrics:
            trends = self._analyze_metric_trends(daily_metrics)
            if trends.get('trends'):
                top_trend = trends['trends'][0]
                if using_mock_data:
                    insights.append(f"MOCK metric '{top_trend['metric']}' shows {top_trend['trend']} trend ({top_trend['change_percent']:.1f}% change)")
                else:
                    insights.append(f"Key metric '{top_trend['metric']}' shows {top_trend['trend']} trend ({top_trend['change_percent']:.1f}% change)")
        
        # Review findings correlation
        if review_findings:
            severity_dist = self._analyze_finding_patterns(review_findings)['severity_distribution']
            if severity_dist:
                most_common = max(severity_dist.items(), key=lambda x: x[1])
                if using_mock_data:
                    insights.append(f"MOCK review findings show {most_common[1]} {most_common[0]} severity issues")
                else:
                    insights.append(f"Recent review findings show {most_common[1]} {most_common[0]} severity issues")
        
        return insights if insights else ["No significant temporal patterns detected"]
