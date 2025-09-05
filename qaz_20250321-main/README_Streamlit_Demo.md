# MVR Review - Streamlit Demo

## 🎯 Overview

This Streamlit-based demo provides a beautiful, interactive interface for Model Validation & Risk Assessment (MVR) that integrates with your actual MCP (Model Context Protocol) code. It offers comprehensive MVR analysis functionality with real backend integration.

## ✨ Features

### 🎨 Beautiful UI
- **Gradient Design**: Professional color scheme matching your brand
- **Progress Steps**: Clear 3-step workflow visualization
- **Interactive Cards**: Hover effects and smooth transitions
- **Responsive Layout**: Works on desktop and mobile

### 📁 File Management
- **Multi-file Upload**: Support for various document types
- **Real Classification**: Uses your MCP FileClassificationWorker
- **File Preview**: Shows file details and classifications
- **Supported Formats**: TXT, PDF, DOCX, XLSX, CSV, JSON, YAML, PY, SQL, XML

### 📊 Analysis Types
1. **Compliance Report** 📄
   - Risk control assessment
   - Data quality evaluation
   - Regulatory compliance check
   - Documentation review

2. **Consistency Report** 📈
   - Model drift analysis
   - Performance stability
   - Quality metrics
   - Trend visualization

3. **Challenge Report** 👥
   - Peer review simulation
   - Stress testing results
   - Edge case handling
   - Success rate assessment

### 🔧 Technical Integration
- **MCP Orchestrators**: Simple, Enhanced, Advanced
- **Multi-scope Workers**: FileClassification, TOCExtractor, BibliographyBuilder, ImageProcessing
- **Real-time Analysis**: Actual backend processing
- **Session Management**: State persistence across interactions

## 🚀 Quick Start

### Prerequisites
```bash
# Install Streamlit and dependencies
pip install -r requirements_streamlit.txt

# Or install manually
pip install streamlit plotly pandas
```

### Launch Options

#### Option 1: Direct Streamlit Command
```bash
streamlit run streamlit_mvr_demo.py --server.port 8501
```

#### Option 2: Using Launcher Script
```bash
python run_streamlit_demo.py
```

#### Option 3: Manual Launch
```bash
python -m streamlit run streamlit_mvr_demo.py
```

## 📋 Usage Guide

### Step 1: Upload Files
1. Click "Choose files" or drag files to the upload area
2. Select multiple files of different types
3. View automatic file classification results
4. Click "Continue to Step 2"

### Step 2: Select Report Type
1. Choose from three analysis types:
   - **Compliance**: Most popular, regulatory focus
   - **Consistency**: Performance and stability
   - **Challenge**: Advanced testing scenarios
2. Click your preferred report type

### Step 3: Review Results
1. Watch real-time analysis progress
2. View comprehensive results with metrics
3. Download detailed reports
4. Try different analysis types

## 🏗️ Architecture

### Frontend (Streamlit)
```
streamlit_mvr_demo.py
├── MVRDemo Class
│   ├── Session State Management
│   ├── UI Rendering
│   ├── File Upload Handling
│   └── Results Display
└── Integration Layer
    ├── MCP Component Loading
    ├── File Classification
    └── Analysis Processing
```

### Backend Integration
```
MCP Components
├── Orchestrators
│   ├── SimpleQAOrchestrator
│   ├── EnhancedQAOrchestrator
│   └── AdvancedQAOrchestrator
└── Workers
    ├── FileClassificationWorker
    ├── TOCExtractorWorker
    ├── BibliographyBuilderWorker
    └── ImageProcessingWorker
```

## 🎨 Customization

### Colors and Styling
The demo uses your brand colors:
- Primary: `#085280` (Dark Blue)
- Secondary: `#238196` (Medium Blue)
- Accent: `#C55422` (Orange)

### CSS Customization
Edit the CSS section in `streamlit_mvr_demo.py` to modify:
- Header styling
- Card appearances
- Progress indicators
- Metric displays

### Adding New Analysis Types
1. Add new report type in `step_2_pick_report()`
2. Create corresponding render method
3. Update `generate_mock_results()`
4. Add to results rendering logic

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom port
export STREAMLIT_SERVER_PORT=8501

# Optional: Disable telemetry
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### MCP Integration
The demo automatically detects MCP availability:
- ✅ **Available**: Full functionality with real backend
- ❌ **Unavailable**: Mock data with graceful degradation

## 📊 Performance

### Optimization Features
- **Lazy Loading**: MCP components loaded on demand
- **Session Caching**: Results cached in Streamlit session
- **Efficient File Handling**: Temporary file cleanup
- **Progress Indicators**: Real-time feedback

### Scalability
- **Multi-file Processing**: Handles multiple files efficiently
- **Background Processing**: Non-blocking analysis
- **Memory Management**: Proper cleanup of temporary files

## 🐛 Troubleshooting

### Common Issues

#### MCP Components Not Found
```
Error: MCP components not available
```
**Solution**: Ensure your MCP backend is properly installed and accessible.

#### File Upload Issues
```
Error: File classification failed
```
**Solution**: Check file format support and MCP worker availability.

#### Port Already in Use
```
Error: Address already in use
```
**Solution**: Use a different port or stop existing Streamlit instances.

### Debug Mode
```bash
# Enable debug logging
streamlit run streamlit_mvr_demo.py --logger.level=debug
```

## 🔄 Updates and Maintenance

### Adding New Features
1. **New File Types**: Update `file_uploader` type list
2. **New Analysis**: Add to report selection and rendering
3. **New Metrics**: Extend results generation and display
4. **New Workers**: Integrate additional MCP components

### Version Compatibility
- **Streamlit**: >= 1.28.0
- **Plotly**: >= 5.17.0
- **Pandas**: >= 2.0.0
- **Python**: >= 3.8

## 📈 Roadmap

### Planned Enhancements
- [ ] **Real-time Collaboration**: Multi-user support
- [ ] **Advanced Visualizations**: More interactive charts
- [ ] **Export Options**: PDF, Excel, PowerPoint reports
- [ ] **API Integration**: REST API for external systems
- [ ] **Custom Dashboards**: User-configurable layouts
- [ ] **Machine Learning**: AI-powered insights

### Integration Opportunities
- [ ] **Jupyter Integration**: Notebook export
- [ ] **GitHub Integration**: Repository analysis
- [ ] **Cloud Storage**: AWS S3, Google Cloud integration
- [ ] **CI/CD Pipeline**: Automated validation workflows

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

### Testing
```bash
# Run basic tests
python -m pytest tests/

# Run Streamlit tests
streamlit run streamlit_mvr_demo.py --test
```

## 📞 Support

### Getting Help
- **Documentation**: Check this README and code comments
- **Issues**: Report bugs via GitHub issues
- **Discussions**: Use GitHub discussions for questions
- **Email**: Contact the development team

### Community
- **Slack**: Join our development channel
- **Discord**: Community discussions
- **Twitter**: Follow for updates

---

**🎉 Enjoy using the MVR Review Streamlit Demo!**

*Built with ❤️ using Streamlit and your MCP architecture*

