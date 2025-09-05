#!/usr/bin/env python3
"""
Generate HTML Dashboard for API Status
"""

import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def generate_html_dashboard():
    """Generate an HTML dashboard for API status."""
    
    # Import the dashboard
    from api_status_dashboard import APIStatusDashboard
    
    # Run the dashboard
    dashboard = APIStatusDashboard()
    working_count, total_count = dashboard.test_all_providers()
    report = dashboard.generate_report()
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VectorQA Sage - API Status Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .status-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .status-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border-left: 5px solid #28a745;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }}
        .status-card.error {{
            border-left-color: #dc3545;
        }}
        .status-card.warning {{
            border-left-color: #ffc107;
        }}
        .provider-name {{
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .status-working {{
            background: #28a745;
        }}
        .status-error {{
            background: #dc3545;
        }}
        .status-warning {{
            background: #ffc107;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .stat-label {{
            opacity: 0.9;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .credentials-list {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .credential-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #dee2e6;
        }}
        .refresh-btn {{
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            margin-bottom: 20px;
        }}
        .refresh-btn:hover {{
            background: #5a6fd8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 VectorQA Sage</h1>
            <p>API Status Dashboard</p>
        </div>
        
        <div class="content">
            <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Status</button>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{report['working_count']}</div>
                    <div class="stat-label">Working Providers</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{report['total_count']}</div>
                    <div class="stat-label">Total Providers</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{report['working_count']/report['total_count']*100:.0f}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
            </div>
            
            <div class="section">
                <h2>🤖 Provider Status</h2>
                <div class="status-grid">
    """
    
    # Add provider status cards
    for provider, result in report['results'].items():
        status = result['status']
        if status == "✅ Working":
            card_class = ""
            status_class = "status-working"
            status_text = "Working"
        elif status == "❌ Failed":
            card_class = "error"
            status_class = "status-error"
            status_text = "Failed"
        else:
            card_class = "warning"
            status_class = "status-warning"
            status_text = "Error"
        
        html_content += f"""
                    <div class="status-card {card_class}">
                        <div class="provider-name">
                            <span class="status-indicator {status_class}"></span>
                            {provider.upper()}
                        </div>
                        <div><strong>Status:</strong> {status_text}</div>
                        <div><strong>Details:</strong> {result['details'][:50]}...</div>
                        <div><strong>Tested:</strong> {result['timestamp'].strftime('%H:%M:%S')}</div>
                    </div>
        """
    
    html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>🔑 Credentials Status</h2>
                <div class="credentials-list">
    """
    
    # Add credentials status
    for provider, status in report['credentials'].items():
        status_class = "status-working" if status == "✅ Configured" else "status-error"
        html_content += f"""
                    <div class="credential-item">
                        <span>{provider}</span>
                        <span class="status-indicator {status_class}"></span>
                    </div>
        """
    
    html_content += f"""
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 Recommendations</h2>
                <div class="status-card">
                    <div class="provider-name">Primary Provider</div>
                    <div><strong>Recommended:</strong> {report['working_providers'][0].upper() if report['working_providers'] else 'None'}</div>
                    <div><strong>Fallbacks:</strong> {', '.join(report['working_providers'][1:]).upper() if len(report['working_providers']) > 1 else 'None'}</div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>VectorQA Sage - AI-Powered Document Analysis</p>
        </div>
    </div>
</body>
</html>
    """
    
    # Write to file
    with open('api_status_dashboard.html', 'w') as f:
        f.write(html_content)
    
    print("✅ HTML Dashboard generated: api_status_dashboard.html")
    print("🌐 Open the file in your browser to view the dashboard")
    
    return html_content

if __name__ == "__main__":
    generate_html_dashboard()
