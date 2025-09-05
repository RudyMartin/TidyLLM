# TidyLLM Whitepapers - Streamlit Demo

**Mathematical Decomposition & Context Engineering Research Showcase**

This demo showcases TidyLLM's whitepaper retrieval and analysis capabilities, specifically focused on our Y="+" vs C="-" research framework for Context Engineering.

## 🎯 Demo Purpose

Demonstrate how TidyLLM processes research papers about:
- Mathematical decomposition models (Y = R + S + N)
- Signal-noise separation techniques  
- Context collapse prevention in AI systems
- Residual risk analysis in ML models

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd tidyllm-whitepapers/streamlit_demo
pip install -r requirements.txt
```

### 2. Run Demo

**Auto-Launch Browser (Recommended):**
```bash
# Windows
launch.bat

# Linux/Mac  
./launch.sh
```

**Manual Launch:**
```bash
# With browser auto-launch
streamlit run app.py --server.headless=false

# Without browser auto-launch (default)
streamlit run app.py
```

### 3. Access Demo
Open browser to: `http://localhost:8501`

**Auto-Restart**: The launch script automatically restarts Streamlit when you edit any Python files, so changes appear immediately after browser refresh.

## 📊 Demo Features

### Core Functionality
- **Paper Search**: Search for mathematical decomposition research
- **Y Score Analysis**: Relevance scoring using Y="+" framework  
- **R+S+N Decomposition**: Break down paper content into:
  - **R (Relevant)**: Core systematic content
  - **S (Superfluous)**: Marginally systematic information  
  - **N (Noise)**: True noise and errors
- **Interactive Visualization**: Charts showing decomposition patterns

### Research Focus Areas
1. **Signal-Noise Decomposition Papers**
   - ArXiv: 1808.02578 - "Deep learning of dynamics and signal-noise decomposition"
   - ArXiv: 2508.13144 - "Signal and Noise: A Framework for Reducing Uncertainty"

2. **Orthogonal Decomposition Research**  
   - ArXiv: 2404.17290 - "Efficient Orthogonal Decomposition"
   - ArXiv: 2409.07242 - "Orthogonal Mode Decomposition"

3. **Context Engineering Applications**
   - Context collapse analysis from dbreuning.com
   - Mathematical frameworks for LLM reliability
   - Corporate compliance and risk management

## 🔧 Configuration

### Demo Papers
The demo includes pre-configured research papers with Y scores and R+S+N decomposition:
- High relevance papers (Y > 0.9)
- Medium relevance papers (Y 0.7-0.9)  
- Analysis of decomposition patterns

### Search Parameters
- **Query Terms**: Mathematical concepts to search
- **Paper Sources**: ArXiv, local files, or both
- **Analysis Depth**: 1-5 scale for processing intensity

## 📈 Mathematical Framework

### Y="+" vs C="-" Model
```
Y = Relevant Content (positive signal)
C = Context Collapse (negative effects)

Where C = R + S + N:
- R: Relevant but systematic items
- S: Superfluous marginally systematic content  
- N: True noise and errors
```

### Scoring System
- **Y Score**: Overall relevance (0.0 - 1.0)
- **R Component**: Core content relevance
- **S Component**: Background/supporting content
- **N Component**: Irrelevant noise content

## 🎨 User Interface

### Main Dashboard
- Search controls in sidebar
- Demo papers with decomposition charts
- Interactive analysis for each paper

### Analysis View  
- Detailed paper breakdown
- Progress bars for R+S+N components
- Key excerpts and insights
- Context Engineering relevance scoring

### System Capabilities
- Paper discovery across multiple sources
- Mathematical content extraction
- Citation network analysis  
- Export to multiple formats

## 💼 Business Value

### For Corporate Users
- **Compliance**: Systematic research validation
- **Risk Assessment**: Mathematical decomposition of uncertainty
- **Quality Control**: R+S+N analysis ensures content reliability
- **Decision Support**: Evidence-based research insights

### For Researchers
- **Literature Review**: Automated paper discovery and analysis
- **Mathematical Validation**: Y=R+S+N framework testing
- **Context Engineering**: Practical applications of decomposition theory
- **Citation Analysis**: Academic network mapping

## 🔍 Demo Scenarios

### Scenario 1: Context Collapse Research
Search for papers addressing the four types of context collapse:
1. Context Poisoning (misinformation absorption)
2. Context Distraction (information overload)  
3. Context Confusion (non-relevant data)
4. Context Clash (conflicting sources)

### Scenario 2: Mathematical Decomposition
Analyze papers using Y=R+S+N framework to validate:
- Signal processing techniques
- Noise separation methods
- Residual risk analysis
- Systematic vs random components

### Scenario 3: Corporate Application
Demonstrate how decomposition analysis supports:
- Model risk governance
- Regulatory compliance
- Quality assurance processes
- Evidence-based decision making

## 📁 File Structure
```
streamlit_demo/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies  
├── README.md          # This documentation
└── config/            # Configuration files (planned)
```

## 🚀 Next Steps

### Phase 1 Enhancements
- [ ] Connect to actual tidyllm-whitepapers backend
- [ ] Real ArXiv API integration
- [ ] Enhanced visualization with Plotly
- [ ] Export functionality (JSON, CSV, LaTeX)

### Phase 2 Features  
- [ ] Real-time paper monitoring
- [ ] Citation network visualization
- [ ] Collaborative analysis features
- [ ] Advanced mathematical parsing

### Phase 3 Integration
- [ ] HeirOS electrical system integration
- [ ] SPARSE agreement documentation
- [ ] Corporate compliance reporting
- [ ] MLflow experiment tracking

## 📞 Support

For questions about this demo or TidyLLM capabilities:
- Focus on mathematical decomposition research
- Y="+" vs C="-" framework applications  
- Context Engineering implementations
- Corporate research compliance needs

---

**Demo Status**: Ready for showcase  
**Target Audience**: Corporate research teams, compliance officers, AI researchers  
**Core Message**: TidyLLM transforms research paper analysis through mathematical decomposition

*"Turning research complexity into systematic insights - one decomposition at a time."*