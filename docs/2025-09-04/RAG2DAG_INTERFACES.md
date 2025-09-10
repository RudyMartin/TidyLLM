# RAG2DAG Interfaces
## User Interface Design for RAG2DAG Workflows

### Overview

RAG2DAG provides multiple interfaces to accommodate different user preferences and use cases:

1. **Drop Zone Interface** (Drag & Drop)
2. **CLI Interface** (Command Line)
3. **Python API Interface** (Programmatic)
4. **Web Interface** (Browser-based)
5. **HeirOS Chat Interface** (Conversational)

---

## 1. Drop Zone Interface (Primary)

### **The Simplest Experience**

**User Action:**
```
1. Drop files in: documents/
2. Drop query in: queries/research_question.txt
3. Results appear in: results/
```

**File Structure:**
```
tidyllm_workspace/
├── documents/          # Drop your files here
│   ├── paper1.pdf
│   ├── paper2.pdf
│   └── research.docx
├── queries/            # Drop your questions here
│   └── "What are the main themes.txt"
└── results/           # Automatic results appear here
    ├── workflow_analysis.json
    ├── extracted_content/
    │   ├── facts.json
    │   ├── quotes.json
    │   └── methods.json
    └── final_report.md
```

**Auto-Detection:**
- File watcher monitors `documents/` and `queries/`
- When both have content → RAG2DAG automatically triggers
- Progress shown in `results/workflow_status.json`

### **Visual Progress Indicator**
```
results/workflow_status.json:
{
  "status": "processing",
  "pattern": "Research Synthesis", 
  "progress": {
    "extract_facts": "completed",
    "extract_quotes": "in_progress", 
    "extract_methods": "pending",
    "synthesize": "pending",
    "generate": "pending"
  },
  "estimated_completion": "2024-12-07T10:45:00Z"
}
```

---

## 2. CLI Interface

### **Command Structure**
```bash
# Simple workflow creation
tidyllm rag2dag create \
  --query "What are the main findings?" \
  --files paper1.pdf paper2.pdf paper3.pdf \
  --output results/

# With configuration
tidyllm rag2dag create \
  --query "Compare methodologies" \
  --files *.pdf \
  --config speed \
  --parallel 5 \
  --output analysis/

# Interactive mode
tidyllm rag2dag interactive
```

### **Interactive CLI Flow**
```bash
$ tidyllm rag2dag interactive

RAG2DAG Interactive Workflow Builder
====================================

📁 Select files:
   1. Browse files
   2. Use current directory (*.pdf, *.docx)
   3. Specify pattern
> 2

Found 4 files: paper1.pdf, paper2.pdf, study.docx, report.pdf

❓ What's your question/goal?
> Compare the methodologies across these research papers

🔍 Analysis: Multi-source research pattern detected
   Complexity: 5/10, Estimated cost: 2.5x, Time: ~12 minutes

🎛️ Optimization level:
   1. Speed (Haiku, 8 parallel) - $2.50
   2. Balanced (Mixed models, 3 parallel) - $4.20  
   3. Quality (Sonnet, 2 parallel) - $8.10
> 2

✅ Creating workflow...
✅ DAG generated: 6 nodes, 3 parallel groups
✅ Starting execution...

Progress: [██████░░░░] 60% - Synthesis in progress
Results will be saved to: results/workflow_20241207_104530/
```

### **CLI Commands**

```bash
# Core commands
tidyllm rag2dag create          # Create new workflow
tidyllm rag2dag status          # Check running workflows  
tidyllm rag2dag list            # List all workflows
tidyllm rag2dag results <id>    # Get workflow results
tidyllm rag2dag stop <id>       # Stop running workflow

# Configuration
tidyllm rag2dag config          # Show current config
tidyllm rag2dag config --speed  # Switch to speed mode
tidyllm rag2dag config --custom # Interactive config editor

# Analysis
tidyllm rag2dag analyze <query> <files>  # Preview workflow (no execution)
tidyllm rag2dag patterns        # List available patterns
tidyllm rag2dag costs          # Show cost estimates
```

---

## 3. Python API Interface

### **Simple API**
```python
from tidyllm.rag2dag import RAG2DAG

# One-line workflow creation
results = RAG2DAG.create_and_run(
    query="What are the key insights?",
    files=["doc1.pdf", "doc2.pdf"],
    config="balanced"
)

print(results.final_report)
```

### **Advanced API**
```python
from tidyllm.rag2dag import RAG2DAGConverter, RAG2DAGConfig, DAGExecutor

# Custom configuration
config = RAG2DAGConfig.create_speed_config()
config.max_parallel_nodes = 6
config.aws_region = "us-west-2"

# Create workflow
converter = RAG2DAGConverter(config)
workflow = converter.create_workflow_from_query(
    query="Analyze compliance requirements",
    files=["reg1.pdf", "reg2.pdf", "policy.docx"],
    context={"industry": "finance", "urgency": "high"}
)

# Execute workflow
executor = DAGExecutor(config)
results = executor.execute_workflow(workflow['dag_nodes'])

# Access results
print(f"Pattern used: {workflow['pattern_name']}")
print(f"Total cost: ${results.total_cost:.2f}")
print(f"Execution time: {results.duration_seconds}s")
print(results.final_output)
```

### **Async API**
```python
import asyncio
from tidyllm.rag2dag import AsyncRAG2DAG

async def process_documents():
    async with AsyncRAG2DAG(config="quality") as rag2dag:
        
        # Start multiple workflows
        workflow1 = await rag2dag.create_workflow(
            "Summarize findings", ["research1.pdf", "research2.pdf"]
        )
        workflow2 = await rag2dag.create_workflow(  
            "Extract methodology", ["study1.pdf", "study2.pdf"]
        )
        
        # Execute in parallel
        results = await asyncio.gather(
            rag2dag.execute(workflow1),
            rag2dag.execute(workflow2)
        )
        
        return results

results = asyncio.run(process_documents())
```

---

## 4. Web Interface

### **Dashboard Layout**
```
┌─────────────────────────────────────────────────────────────────┐
│ TidyLLM RAG2DAG Dashboard                          [Settings]   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📁 DROP FILES HERE                     ❓ ENTER QUERY         │
│  ┌─────────────────────────┐            ┌──────────────────────┐ │
│  │ Drag & drop files       │            │ What do you want to  │ │
│  │ or click to browse      │            │ know about these     │ │
│  │                         │            │ documents?           │ │
│  │ [+] Browse Files        │            │                      │ │
│  └─────────────────────────┘            └──────────────────────┘ │
│                                                                 │
│  📊 WORKFLOW PREVIEW                                            │
│  Pattern: Research Synthesis | Complexity: 7/10 | Cost: $4.20  │
│  ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐                      │
│  │Ext 1│    │Ext 2│    │Ext 3│    │Ext 4│ → [Synthesis] → [Gen]│
│  └─────┘    └─────┘    └─────┘    └─────┘                      │
│                                                                 │
│  ⚙️ CONFIGURATION                           [CREATE WORKFLOW]   │
│  Optimization: ◉ Balanced ○ Speed ○ Quality                    │
│  Max Parallel: 3        Region: us-east-1                      │
└─────────────────────────────────────────────────────────────────┘
```

### **Workflow Monitoring**
```
┌─────────────────────────────────────────────────────────────────┐
│ Workflow: Research Analysis #20241207_104530        [94% Done] │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📊 EXECUTION PROGRESS                                          │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ ✅ Extract Facts      (Haiku)     [2.3s] $0.12              │ │
│  │ ✅ Extract Quotes     (Haiku)     [1.9s] $0.08              │ │ 
│  │ ✅ Extract Methods    (Haiku)     [2.1s] $0.11              │ │
│  │ ✅ Extract Findings   (Haiku)     [2.4s] $0.13              │ │
│  │ ✅ Synthesize         (Sonnet)    [8.7s] $1.24              │ │
│  │ 🔄 Generate Report    (Sonnet)    [3.2s] $0.45...           │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  💰 COST TRACKING                   ⏱️ TIME TRACKING           │
│  Current: $2.13 / $4.20 est         Elapsed: 18.4s / 720s est  │
│                                                                 │
│  📁 RESULTS                                                     │
│  ├── extracted_facts.json     ✅ Ready                         │
│  ├── extracted_quotes.json    ✅ Ready                         │
│  ├── extracted_methods.json   ✅ Ready                         │
│  ├── synthesis_analysis.json  ✅ Ready                         │
│  └── final_report.md         🔄 Generating...                 │
└─────────────────────────────────────────────────────────────────┘
```

### **Results View**
```
┌─────────────────────────────────────────────────────────────────┐
│ Research Analysis Results                           [Export]    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📄 FINAL REPORT                             [View Raw] [Edit]  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ # Research Analysis Summary                                 │ │
│  │                                                             │ │
│  │ ## Key Findings                                             │ │
│  │ Based on analysis of 4 research papers, the main themes... │ │
│  │                                                             │ │
│  │ ## Methodology Comparison                                   │ │
│  │ The studies employed three distinct approaches...           │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  🔍 EXTRACTED DATA                                              │
│  ├── 📊 Facts (247 items)         [View] [Download JSON]       │
│  ├── 💬 Quotes (89 items)         [View] [Download JSON]       │
│  ├── ⚗️ Methods (12 approaches)    [View] [Download JSON]       │
│  └── 🎯 Findings (34 conclusions) [View] [Download JSON]       │
│                                                                 │
│  📈 WORKFLOW METRICS                                            │
│  Total Cost: $2.89 | Execution Time: 42.7s | Pattern: Research │
│  Models Used: Haiku (4x), Sonnet (2x) | Parallel Efficiency: 73%│
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. HeirOS Chat Interface

### **Conversational Workflow Creation**

```
User: I need to analyze some compliance documents

HeirOS: I'd be happy to help you analyze compliance documents! 
        Let me ask a few questions to set up the optimal workflow:

        1. What type of compliance documents? (regulatory, policy, audit, etc.)
        2. How many documents do you have?
        3. What specifically do you need from them?

User: I have 6 regulatory documents and need to extract all compliance requirements

HeirOS: Perfect! For regulatory compliance extraction, I recommend:

        📋 Workflow: Compliance Extraction Pattern
        🎯 Goal: Extract structured compliance requirements  
        🔧 Configuration: Balanced (good speed + quality)
        💰 Estimated Cost: $3.40
        ⏱️ Estimated Time: 8-12 minutes

        The workflow will:
        1. Extract requirements from each document (parallel)
        2. Categorize by compliance type
        3. Cross-reference requirements
        4. Generate structured compliance matrix

        Would you like me to create this workflow?

User: Yes, create it

HeirOS: ✅ Workflow created! 

        📁 Please drop your 6 regulatory documents in: documents/
        🎯 I've automatically set the query to: "Extract all compliance requirements"
        
        Once you drop the files, processing will start automatically.
        I'll keep you updated on progress!

[Files dropped]

HeirOS: 🚀 Processing started!
        
        ✅ Document parsing complete
        🔄 Extracting requirements (Document 3/6)...
        
        Current progress: 45% complete
        Estimated completion: 6 minutes

[Later]

HeirOS: ✅ Analysis complete! Here's what I found:

        📊 Compliance Requirements Summary:
        • 127 total requirements identified
        • 23 critical requirements (immediate action needed)
        • 45 standard requirements (routine compliance)
        • 59 advisory requirements (best practices)

        📁 Detailed results saved to: results/compliance_analysis_20241207/
        
        Would you like me to:
        1. Show the critical requirements first?
        2. Generate an action plan?
        3. Create compliance tracking templates?

User: Show me the critical requirements

HeirOS: 🚨 Critical Requirements (23 items):

        1. **Data Protection Compliance** (Regulation EU-2024-1847)
           Deadline: 2024-12-31
           Action: Implement encryption protocols for customer data
           
        2. **Financial Reporting Standards** (Section 401.3)
           Deadline: 2025-01-15  
           Action: Update quarterly reporting format
           
        [... continues with all 23 items ...]

        💡 I can help you create action plans for any of these. 
        Just ask: "Create action plan for [requirement name]"
```

---

## 6. Interface Integration

### **Unified Experience**
All interfaces share common features:

1. **Real-time Progress**: Live updates across all interfaces
2. **Result Sharing**: URLs/IDs work across CLI, web, and API
3. **Configuration Sync**: Settings persist across interfaces
4. **Cost Tracking**: Consistent cost reporting
5. **Error Handling**: Graceful failures with helpful messages

### **Cross-Interface Commands**
```bash
# CLI creates workflow
$ tidyllm rag2dag create --query "analyze" --files *.pdf
> Workflow ID: wf_20241207_104530

# Web interface can monitor
https://tidyllm.local/workflow/wf_20241207_104530

# API can retrieve results
results = RAG2DAG.get_results("wf_20241207_104530")

# Drop zone can trigger same pattern
# Just drop same files + query txt file
```

### **Mobile-Friendly Web Interface**
```
┌─────────────────────┐
│ RAG2DAG            │
├─────────────────────┤
│ 📁 Upload Files     │
│ [Choose Files]      │
│                     │
│ ❓ Your Question    │
│ [Text Input]        │
│                     │
│ ⚙️ Quick Config      │
│ ○ Fast ○ Balanced   │
│ ○ Quality           │
│                     │
│ [Create Workflow]   │
│                     │
│ 📊 Active: 2        │
│ ✅ Done: 15         │
└─────────────────────┘
```

This multi-interface approach ensures RAG2DAG is accessible to everyone - from drag-and-drop users to developers to conversational AI users!