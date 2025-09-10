# HeirOS Streamlit Control Center

A comprehensive web dashboard for managing TidyLLM-HeirOS hierarchical workflows, SPARSE agreements, and system monitoring.

## 🌟 Features

### 🏠 Dashboard Overview
- **Real-time system health** metrics (24h executions, success rates, running workflows)
- **Workflow performance** charts showing execution counts and success rates
- **Recent activity** feed with latest workflow executions
- **Interactive visualizations** using Plotly charts

### 🔧 Workflow Manager
- **Active workflow listing** with status, compliance levels, and metadata
- **Workflow creation wizard** with JSON structure editor
- **Execution controls** (start, pause, edit workflows)
- **Analytics dashboard** showing performance trends and success distributions

### 📜 SPARSE Agreements
- **Pending approvals** management with risk assessment
- **Active agreement** monitoring with expiry tracking
- **Agreement creation** wizard with conditions and actions
- **Usage statistics** and execution tracking

### 📊 System Analytics
- **Performance trends** over 30-day periods
- **Error analysis** with common failure patterns
- **Database health** monitoring with table statistics
- **Execution duration** analysis and optimization insights

### 🧩 Node Template Library
- **Template browsing** by category and usage
- **Template details** with descriptions and versions
- **One-click template** integration into workflows
- **Usage statistics** and popularity metrics

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL database with HeirOS tables
- TidyLLM-HeirOS system installed

### Installation

1. **Install requirements:**
```bash
pip install -r heiros_requirements.txt
```

2. **Launch dashboard:**
```bash
# Windows
launch_heiros_demo.bat

# Linux/Mac
streamlit run heiros_streamlit_demo.py --server.port=8501
```

3. **Open browser:**
Navigate to `http://localhost:8501`

## 📊 Dashboard Screenshots

### Main Dashboard
- System health metrics in real-time
- Top workflow performance charts
- Recent execution activity feed

### Workflow Manager
```
📋 Active Workflows | ➕ Create Workflow | 📊 Analytics
```
- Complete workflow lifecycle management
- Visual workflow designer (JSON-based)
- Performance analytics and insights

### SPARSE Agreements
```
⏳ Pending Approvals | ✅ Active Agreements | ➕ Create Agreement  
```
- Corporate compliance management
- Risk assessment and approval workflows
- Usage tracking and statistics

## 🗄️ Database Integration

The dashboard connects directly to your PostgreSQL HeirOS database:

```python
# Connection configuration (in heiros_streamlit_demo.py)
conn = psycopg2.connect(
    host='vectorqa-cluster.cluster-cu562e4m02nq.us-east-1.rds.amazonaws.com',
    database='vectorqa',
    user='vectorqa_user',
    password='Fujifuji500!',
    sslmode='require'
)
```

### Required Tables
- `heiros_workflows` - Workflow definitions
- `heiros_executions` - Execution history
- `heiros_sparse_agreements` - SPARSE agreements
- `heiros_node_templates` - Reusable templates
- `heiros_audit_trail` - Audit logging

## 📈 Analytics Capabilities

### Performance Metrics
- Workflow success rates and failure analysis
- Execution time trends and optimization opportunities
- Resource utilization and system load

### Business Intelligence
- SPARSE agreement usage patterns
- Template popularity and effectiveness
- User activity and workflow adoption

### Compliance Reporting
- Complete audit trails with timestamps
- Risk assessment tracking
- Regulatory compliance monitoring

## 🎨 Customization

### Theme Configuration
The dashboard uses Streamlit's theming system:

```python
# Custom CSS in main()
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
}
</style>
""", unsafe_allow_html=True)
```

### Query Customization
Modify SQL queries in the `load_query()` function to customize data views and analytics.

## 🔧 Configuration

### Environment Variables
```bash
export HEIROS_DB_HOST="your-postgres-host"
export HEIROS_DB_NAME="vectorqa"
export HEIROS_DB_USER="vectorqa_user"
export HEIROS_DB_PASSWORD="your-password"
```

### Streamlit Configuration
Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"

[server]
port = 8501
address = "localhost"
```

## 🚦 System Status

The sidebar shows real-time system status:
- 🟢 Database Connected
- 🔴 Database Disconnected
- Version information
- Environment details

## 📱 Mobile Responsive

The dashboard is optimized for desktop use but includes mobile-responsive elements:
- Collapsible sidebar
- Responsive column layouts
- Touch-friendly buttons

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check PostgreSQL credentials
   - Verify network connectivity
   - Ensure HeirOS tables exist

2. **Missing Data**
   - Verify workflow executions exist
   - Check table permissions
   - Review SQL query syntax

3. **Performance Issues**
   - Optimize database queries
   - Add indexes to frequently queried columns
   - Use query result caching

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   PostgreSQL     │    │   HeirOS        │
│   Frontend      │◄──►│   Database       │◄──►│   Engine        │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Plotly        │    │   SQL Queries    │    │   Workflow      │
│   Visualizations│    │   & Analytics    │    │   Execution     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🛠️ Development

### Adding New Features

1. **New Page:**
```python
def my_new_page():
    st.title("My New Feature")
    # Implementation here

# Add to sidebar navigation
page = st.sidebar.radio("Navigate:", [..., "🆕 My Feature"])
```

2. **New Visualization:**
```python
import plotly.express as px

fig = px.scatter(data, x='col1', y='col2', title='My Chart')
st.plotly_chart(fig, use_container_width=True)
```

3. **New Database Query:**
```python
def load_my_query():
    return """
    SELECT * FROM my_table
    WHERE conditions = %s;
    """
```

## 📚 Dependencies

### Core Requirements
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **psycopg2-binary**: PostgreSQL database adapter
- **plotly**: Interactive visualizations
- **pyyaml**: YAML configuration parsing

### Optional Enhancements
- **streamlit-aggrid**: Advanced data grids
- **streamlit-elements**: Additional UI components
- **streamlit-option-menu**: Enhanced navigation

## 🎯 Future Enhancements

### Planned Features
- **Visual workflow designer** (drag-and-drop interface)
- **Real-time execution monitoring** (WebSocket updates)
- **Advanced analytics** (predictive modeling)
- **Multi-tenant support** (organization isolation)
- **API integration** (RESTful endpoints)
- **Export capabilities** (PDF reports, CSV data)

### Integration Opportunities
- **MLflow integration** for experiment tracking
- **Grafana dashboards** for infrastructure monitoring  
- **Slack notifications** for workflow events
- **JIRA integration** for issue tracking

---

*TidyLLM-HeirOS Streamlit Control Center - Enterprise Workflow Management Made Simple*