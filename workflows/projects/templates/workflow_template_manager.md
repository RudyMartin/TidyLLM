# Workflow Template Manager

## Overview
This document explains how to create, organize, and use step prompt templates in TidyLLM workflows.

## Template Organization Structure

```
your_project/
├── criteria/                          # Scoring and evaluation criteria
├── outputs/                           # Generated results storage
├── resources/                         # Reference materials
└── templates/                         # Step prompt templates
    ├── step_01_input_validation.md    # First step prompts
    ├── step_02_data_extraction.md     # Second step prompts
    ├── step_03_analysis.md            # Analysis step prompts
    ├── step_04_synthesis.md           # Synthesis step prompts
    ├── step_05_output_generation.md   # Final output prompts
    └── workflow_config.json           # Template configuration
```

## Template Naming Convention

**Format:** `step_XX_descriptive_name.md`

- **step_XX** - Sequential step number (01, 02, 03...)
- **descriptive_name** - Clear step purpose (validation, extraction, analysis)
- **.md** - Markdown format for rich formatting

## Template Structure Standard

Each template should include:

### 1. Header Section
```markdown
# Step X: Step Name

## Purpose
Clear description of what this step accomplishes
```

### 2. Prompt Template Section
```markdown
## Prompt Template
```
Main LLM prompt with placeholders for variables
```
```

### 3. Variables Section
```markdown
## Input Variables
- `{variable_name}` - Description of variable
- `{another_var}` - Another variable description
```

### 4. RAG Integration Points
```markdown
## RAG Integration Points
- Query types needed
- Knowledge bases to access
- Context enhancement requirements
```

### 5. Output Format
```markdown
## Expected Output Format
```json
{
  "structured_output": "example"
}
```
```

### 6. Success Criteria
```markdown
## Success Criteria
- Measurable success conditions
- Quality thresholds
- Completion requirements
```

## How to Create New Templates

### 1. **Using Flow Creator V3 Portal**
1. Go to http://localhost:8522
2. Navigate to "* Flow Designer" tab
3. Select "WIZARD: Template Wizard" mode
4. Use Step 3: "Flow Definition" to create templates

### 2. **Manual Creation**
1. Create new .md file in templates/ folder
2. Follow the standard structure above
3. Use existing templates as reference
4. Test with sample data

### 3. **Template Variables**
Use placeholder variables in curly braces:
- `{input_files}` - Input document list
- `{previous_step_output}` - Output from previous step
- `{domain_context}` - Domain-specific context
- `{quality_threshold}` - Quality requirements

## Template Orchestration

### Sequential Flow
Templates execute in order: 01 → 02 → 03 → 04 → 05

### Variable Passing
Output from step N becomes input to step N+1:
```
Step 1 Output → {previous_step_output} → Step 2 Input
```

### RAG Integration
Each step can query RAG systems for:
- Domain knowledge enhancement
- Historical data comparison
- Best practice guidance
- Quality validation

## Best Practices

### 1. **Clear Prompts**
- Use specific, actionable language
- Include examples when helpful
- Specify output format requirements

### 2. **Variable Management**
- Document all variables clearly
- Use consistent naming conventions
- Provide default values where appropriate

### 3. **Error Handling**
- Include validation criteria
- Specify fallback procedures
- Define error recovery steps

### 4. **Quality Control**
- Set confidence thresholds
- Include verification steps
- Enable human review points

## Example Workflow Configuration

Create `workflow_config.json` in templates/ folder:

```json
{
  "workflow_name": "Multi-Step Analysis Workflow",
  "version": "1.0",
  "steps": [
    {
      "step_number": 1,
      "template_file": "step_01_input_validation.md",
      "timeout_minutes": 5,
      "retry_attempts": 3,
      "rag_systems": ["ai_powered"]
    },
    {
      "step_number": 2,
      "template_file": "step_02_data_extraction.md",
      "timeout_minutes": 10,
      "retry_attempts": 2,
      "rag_systems": ["ai_powered", "intelligent"]
    },
    {
      "step_number": 3,
      "template_file": "step_03_analysis.md",
      "timeout_minutes": 15,
      "retry_attempts": 2,
      "rag_systems": ["ai_powered", "sme", "dspy"]
    }
  ],
  "global_variables": {
    "quality_threshold": 0.85,
    "confidence_minimum": 0.80,
    "domain": "financial_analysis"
  }
}
```

## Using Templates in Flow Creator V3

1. **Template Selection**: Portal auto-detects templates in templates/ folder
2. **Variable Mapping**: Configure variables in the workflow setup
3. **RAG Integration**: Select appropriate RAG systems per step
4. **Execution Monitoring**: Track step-by-step progress
5. **Output Collection**: Gather results in outputs/ folder

## Advanced Features

### Conditional Steps
Use conditional logic in templates:
```markdown
## Conditional Execution
If `{quality_score}` < 0.8, execute additional validation step
```

### Parallel Execution
Mark steps for parallel execution:
```markdown
## Execution Mode: PARALLEL
This step can run concurrently with step_03_analysis.md
```

### Dynamic Templates
Generate templates based on input:
```markdown
## Dynamic Content
Template content varies based on `{document_type}` variable
```

This template management system provides a structured, scalable approach to creating complex multi-step workflows with reusable prompt templates.