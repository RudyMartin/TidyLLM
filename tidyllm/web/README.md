# AI Dropzone Manager Dashboard
===============================

Streamlit-based web dashboard for monitoring and managing the AI Dropzone Manager system.

## Quick Start
```bash
# Launch dashboard
python launch_dashboard.py

# Or directly with Streamlit
streamlit run tidyllm/web/ai_dropzone_dashboard.py
```

Dashboard will be available at: http://localhost:8501

## Features

### 📁 Drop Zone Monitoring
- Real-time file detection in drop zones
- Automatic bracket command mapping
- Processing status tracking
- Interactive file management

### 📊 System Metrics
- CPU, memory, and resource usage
- Processing rate and queue length
- Success rates and error tracking
- Performance trend analysis

### ⚙️ Worker Management
- Individual worker status monitoring
- Task assignment and progress tracking
- Resource usage per worker
- Worker restart and control

### 🎯 Manual Processing
- Upload files directly through UI
- Execute bracket commands manually
- Preview and analyze documents
- Batch processing operations

### 🚨 Alerts & Notifications
- Real-time system alerts
- Threshold-based warnings
- Error notifications
- Processing status updates

## File Structure
```
web/
├── ai_dropzone_dashboard.py    # Main dashboard application
├── components/                 # Reusable UI components
│   └── dropzone_monitor.py    # Drop zone monitoring widgets
├── pages/                      # Additional dashboard pages
│   └── system_monitor.py      # Advanced system monitoring
├── utils/                      # Helper utilities
│   └── dashboard_helpers.py   # Common functions and utilities
├── static/                     # Static assets
│   ├── css/                   # Custom stylesheets
│   ├── js/                    # JavaScript files
│   └── images/                # Images and icons
├── config.json                # Dashboard configuration
└── requirements.txt           # Python dependencies
```

## Configuration

Edit `web/config.json` to customize:
- Dashboard refresh intervals
- Alert thresholds
- Drop zone settings
- UI preferences
- API endpoints

## Drop Zone Integration

The dashboard monitors these drop zones:
- `tidyllm/drop_zones/mvr_analysis/` → [Process MVR]
- `tidyllm/drop_zones/financial_analysis/` → [Financial Analysis]
- `tidyllm/drop_zones/contract_review/` → [Contract Review]
- `tidyllm/drop_zones/compliance_check/` → [Compliance Check]
- `tidyllm/drop_zones/quality_check/` → [Quality Check]
- `tidyllm/drop_zones/data_extraction/` → [Data Extraction]

## Dependencies

Core requirements:
- streamlit>=1.28.0
- plotly>=5.17.0
- pandas>=2.1.0
- numpy>=1.24.0

Install with:
```bash
pip install -r tidyllm/web/requirements.txt
```

## Development

### Adding New Components
1. Create component in `components/`
2. Import in main dashboard
3. Add configuration to `config.json`

### Adding New Pages
1. Create page in `pages/`
2. Follow Streamlit multipage structure
3. Update navigation as needed

### Customizing UI
- Edit CSS in `static/css/`
- Update configuration in `config.json`
- Modify component styling in Python files

## Security Notes

- Dashboard runs on localhost by default
- File uploads are restricted by type and size
- No authentication enabled by default
- Configure `security` section in config for production

## Troubleshooting

### Common Issues
- **Port 8501 in use**: Change port in launch command
- **Import errors**: Install missing dependencies
- **Unicode errors**: Check console encoding settings
- **File permissions**: Ensure drop zone directories are writable

### Debug Mode
Set `show_debug_info: true` in config.json for additional diagnostic information.

---

**Status**: ✅ Dashboard is running at http://localhost:8501
**Last Updated**: 2024-01-15
**Version**: 1.0