#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Mock Live Context System
Demonstrates the live context integration with mock stock data
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from backend.mcp.workers.live_context_worker import LiveContextWorker
from backend.mcp.protocol.message_protocol import (
    MCPMessage, TaskType, Priority, MessageType,
    create_coordinator_to_worker_message
)

def test_mock_live_context():
    """Test the mock live context system"""
    print("🚀 Mock Live Context Demonstration")
    print("=" * 50)
    
    # Initialize worker
    worker = LiveContextWorker()
    
    # Create mock document analysis
    mock_document_analysis = {
        'document_type': 'financial_report',
        'key_topics': ['revenue', 'compliance', 'performance'],
        'claims': [
            'Revenue increased by 15% year-over-year',
            'Compliance score improved to 95%',
            'System performance maintained 99.9% uptime'
        ],
        'extracted_text': 'This financial report shows strong revenue growth and improved compliance metrics.',
        'metadata': {
            'document_date': '2024-01-15',
            'document_source': 'quarterly_report.pdf'
        }
    }
    
    # Create message
    message = create_coordinator_to_worker_message(
        worker="live_context_worker",
        task_type=TaskType.PROCESSING,
        payload={'document_analysis': mock_document_analysis},
        context={}
    )
    
    print("📄 Mock Document Analysis:")
    print(f"  Document Type: {mock_document_analysis['document_type']}")
    print(f"  Key Topics: {', '.join(mock_document_analysis['key_topics'])}")
    print(f"  Claims: {len(mock_document_analysis['claims'])}")
    print()
    
    # Process with live context
    print("🔍 Processing with Live Context...")
    response = worker.execute(message)
    
    if response.payload['success']:
        result = response.payload['result']['processing']
        
        print("✅ Live Context Processing Complete!")
        print()
        
        # Display key topics
        print("🎯 Extracted Key Topics:")
        for topic in result.get('key_topics', []):
            print(f"  • {topic}")
        print()
        
        # Display live events (mock stock data)
        print("📈 Mock Live Events (Stock Data):")
        live_events = result.get('live_events', [])
        for i, event in enumerate(live_events[:5], 1):  # Show first 5
            print(f"  {i}. {event['event']} - {event['org']}")
            if 'payload' in event:
                payload = event['payload']
                if 'symbol' in payload:
                    print(f"     Symbol: {payload['symbol']}, Change: {payload.get('revenue_change', 'N/A')}")
                elif 'severity' in payload:
                    print(f"     Severity: {payload['severity']}, Regulation: {payload.get('regulation', 'N/A')}")
                elif 'metric' in payload:
                    print(f"     Metric: {payload['metric']}, Value: {payload.get('value', 'N/A')}%")
        print(f"  ... and {len(live_events) - 5} more events")
        print()
        
        # Display daily metrics
        print("📊 Mock Daily Metrics:")
        daily_metrics = result.get('daily_metrics', [])
        for i, metric in enumerate(daily_metrics[:3], 1):  # Show first 3
            print(f"  {i}. {metric['metric']} - {metric['org']}")
            print(f"     Value: {metric['value']}, Date: {metric['day']}")
        print(f"  ... and {len(daily_metrics) - 3} more metrics")
        print()
        
        # Display review findings
        print("🔍 Mock Review Findings:")
        review_findings = result.get('review_findings', [])
        for i, finding in enumerate(review_findings[:3], 1):  # Show first 3
            print(f"  {i}. {finding['rule_id']} - {finding['stage']}")
            print(f"     Severity: {finding['severity']}, Status: {finding['status']}")
        print(f"  ... and {len(review_findings) - 3} more findings")
        print()
        
        # Display temporal context
        print("⏰ Temporal Context Analysis:")
        temporal_context = result.get('temporal_context', {})
        print(f"  Document Age: {temporal_context.get('document_age', {}).get('age_days', 'N/A')} days")
        print(f"  Event Frequency: {temporal_context.get('event_frequency', {}).get('total_events', 'N/A')} events")
        print(f"  Relevance Score: {temporal_context.get('relevance_score', 'N/A'):.2f}")
        print()
        
        # Display temporal insights
        print("💡 Temporal Insights:")
        temporal_insights = temporal_context.get('temporal_insights', [])
        for insight in temporal_insights:
            print(f"  • {insight}")
        print()
        
        # Display performance metrics
        print("📊 Performance Metrics:")
        metrics = worker.get_performance_metrics()
        print(f"  Total Tasks: {metrics['total_tasks']}")
        print(f"  Success Rate: {metrics['success_rate']:.1f}%")
        print(f"  Average Processing Time: {metrics['average_processing_time']:.2f}s")
        
    else:
        print("❌ Live Context Processing Failed!")
        print(f"Error: {response.payload.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_mock_live_context()
