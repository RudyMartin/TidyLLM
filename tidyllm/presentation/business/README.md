# Business Presentation Materials

## Files

- `architecture.md` - Mermaid diagram and business documentation for TidyLLM gateway architecture
- `mermaid_to_png.py` - Utility to convert Mermaid diagrams to PNG images
- `README.md` - This file

## Quick Start

1. **View the architecture diagram on GitHub**: The `architecture.md` file contains a Mermaid diagram that renders automatically on GitHub.

2. **Convert to PNG for presentations**:
   ```bash
   # Install mermaid-cli (one-time setup)
   npm install -g @mermaid-js/mermaid-cli
   
   # Convert diagram to PNG
   python mermaid_to_png.py architecture.md
   
   # Or specify output filename
   python mermaid_to_png.py architecture.md -o business_architecture.png
   ```

3. **Auto-install mermaid-cli**:
   ```bash
   python mermaid_to_png.py architecture.md --install
   ```

## Architecture Overview

The current TidyLLM architecture uses a 4-gateway pipeline:

1. **BusinessContextGateway** - Compliance and regulatory validation
2. **WorkflowSolutionGateway** - Workflow selection and routing  
3. **AIProcessingGateway** - Prompt optimization and enhancement
4. **ModelExecutionGateway** - Model selection and execution

This design ensures compliance-first processing while optimizing for cost and quality.