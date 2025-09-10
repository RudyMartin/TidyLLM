# Unified Gateway Architecture - Revised Plan

## KEY INSIGHT
You're right! The current structure is confusing because:
1. Drop_zones are doing gateway-like work
2. HeirOS functionality overlaps with drop_zones
3. Three "gateways" aren't clearly gateways
4. Adding new drop zones is complicated

## NEW VISION

### Drop Zones = Primary Interface
**Drop zones ARE the main way users interact with the system**
- Each drop zone is essentially a specialized gateway
- Gateways become "processing engines" that drop zones use
- HeirOS becomes a "workflow optimizer" that cleans up messy workflows

```
tidyllm/
├── gateways/                       # All processing engines in ONE place
│   ├── __init__.py
│   ├── base_gateway.py            # Common interface
│   ├── dspy_gateway.py            # DSPy processing engine
│   ├── llm_gateway.py             # Corporate LLM access
│   └── heiros_gateway.py          # Workflow optimization engine
│
├── drop_zones/                     # User-facing interface
│   ├── __init__.py
│   ├── base_zone.py               # Base class for all zones
│   ├── processor.py               # Routes to appropriate gateway
│   ├── registry.py                # Zone registration system
│   │
│   ├── zones/                     # Pre-built drop zones
│   │   ├── document_zone.py       # PDF/DOCX processing
│   │   ├── data_zone.py          # CSV/JSON processing
│   │   ├── research_zone.py      # Research paper workflow
│   │   ├── compliance_zone.py    # Compliance checking
│   │   └── custom_zone.py        # Template for custom zones
│   │
│   └── workflows/                 # HeirOS integration
│       ├── optimizer.py          # Workflow optimization
│       ├── analyzer.py           # Workflow analysis
│       └── cleaner.py           # Clean up manual workflows
```

## ROLE CLARIFICATION

### 1. Drop Zones (Primary Interface)
- **Purpose**: User drops files, magic happens
- **Responsibility**: File monitoring, validation, routing
- **Uses**: One or more gateways for processing

### 2. Gateways (Processing Engines)
- **DSPyGateway**: Multi-model AI processing
- **LLMGateway**: Corporate-controlled LLM access
- **HeirOSGateway**: Workflow optimization & cleanup

### 3. HeirOS (Workflow Intelligence)
- **NOT** a primary gateway, but a **workflow optimizer**
- **Purpose**: Clean up messy manual workflows
- **Functions**:
  - Analyze existing workflows
  - Optimize processing paths
  - Suggest better configurations
  - Auto-correct workflow errors

## MAKING DROP ZONES EASY TO ADD

### Simple Zone Template
```python
from tidyllm.drop_zones.base_zone import BaseDropZone
from tidyllm.gateways import DSPyGateway

class MyCustomZone(BaseDropZone):
    """Custom drop zone for specific file type."""
    
    name = "my_custom_zone"
    file_patterns = ["*.custom", "*.special"]
    gateway = DSPyGateway  # or LLMGateway or HeirOSGateway
    
    def process_file(self, file_path):
        """Define custom processing logic."""
        # Pre-processing
        data = self.validate_and_read(file_path)
        
        # Gateway processing
        result = self.gateway.process(data, self.config)
        
        # Post-processing
        return self.save_results(result)
```

### YAML Configuration for Quick Zones
```yaml
drop_zones:
  - name: invoices
    type: quick  # Use pre-built logic
    patterns: ["*.pdf", "*.invoice"]
    gateway: llm  # Which gateway to use
    workflow:
      - extract_text
      - parse_invoice_fields
      - validate_totals
      - store_in_database
    
  - name: contracts
    type: custom  # Use custom Python class
    class: zones.ContractZone
    patterns: ["*.docx", "*.contract"]
    gateway: dspy
    heiros_optimize: true  # Let HeirOS optimize this workflow
```

## HEIROS AS WORKFLOW OPTIMIZER

### Current Problem with Manual Workflows
- Users create inefficient workflows
- Duplicate processing steps
- Poor error handling
- No optimization

### HeirOS Solution
```python
from tidyllm.drop_zones.workflows import HeirOSOptimizer

# Analyze existing workflow
optimizer = HeirOSOptimizer()
analysis = optimizer.analyze_workflow("invoices")

# Get recommendations
print(analysis.recommendations)
# Output: 
# - Combine extract_text and parse_fields steps
# - Add retry logic for database storage
# - Cache parsed results for similar files

# Auto-optimize
optimized_workflow = optimizer.optimize("invoices")
```

## IMPLEMENTATION STRATEGY

### Phase 1: Consolidate Gateways
```python
# tidyllm/gateways/__init__.py
from .dspy_gateway import DSPyGateway
from .llm_gateway import LLMGateway  
from .heiros_gateway import HeirOSGateway

# Registry for dynamic loading
GATEWAYS = {
    'dspy': DSPyGateway,
    'llm': LLMGateway,
    'heiros': HeirOSGateway
}

def get_gateway(name: str):
    """Get gateway by name."""
    return GATEWAYS[name]()
```

### Phase 2: Create Drop Zone Framework
```python
# tidyllm/drop_zones/base_zone.py
class BaseDropZone:
    """Base class for all drop zones."""
    
    def __init__(self, config):
        self.config = config
        self.gateway = self._init_gateway()
        self.workflow = self._init_workflow()
        
    def _init_gateway(self):
        """Initialize the configured gateway."""
        gateway_name = self.config.get('gateway', 'dspy')
        return get_gateway(gateway_name)
    
    def process(self, file_path):
        """Main processing method."""
        # 1. Validate file
        self.validate(file_path)
        
        # 2. Execute workflow
        result = self.workflow.execute(file_path)
        
        # 3. Let HeirOS analyze if enabled
        if self.config.get('heiros_optimize'):
            self._optimize_workflow(result)
        
        return result
```

### Phase 3: Zone Registry for Easy Addition
```python
# tidyllm/drop_zones/registry.py
class ZoneRegistry:
    """Registry for all drop zones."""
    
    def __init__(self):
        self.zones = {}
        self._load_builtin_zones()
        self._load_custom_zones()
    
    def register(self, zone_class):
        """Register a new drop zone."""
        self.zones[zone_class.name] = zone_class
        
    def create_zone(self, config):
        """Create zone from config."""
        if config['type'] == 'quick':
            return QuickZone(config)
        else:
            zone_class = self.zones[config['name']]
            return zone_class(config)
```

## BENEFITS OF THIS APPROACH

1. **Clear Separation**:
   - Drop Zones = User Interface
   - Gateways = Processing Engines
   - HeirOS = Workflow Optimizer

2. **Easy to Add Zones**:
   - Inherit from BaseDropZone
   - Or use YAML for quick zones
   - Automatic registration

3. **Consistent Pattern**:
   - All gateways in `/gateways`
   - All zones in `/drop_zones/zones`
   - Clear interfaces

4. **HeirOS Integration**:
   - Not forced, but available
   - Analyzes and optimizes workflows
   - Cleans up manual mess

5. **Scalable**:
   - Add new gateways easily
   - Add new zones easily
   - Mix and match as needed

## MIGRATION PATH

### Step 1: Move All Gateways
```bash
# Move and rename
tidyllm/gateways/
  ├── dspy_gateway.py      # From UnifiedDSPyWrapper
  ├── llm_gateway.py       # From deep nesting
  └── heiros_gateway.py    # New wrapper for HeirOS
```

### Step 2: Create Zone Framework
```bash
# New structure
tidyllm/drop_zones/
  ├── base_zone.py         # Base class
  ├── registry.py          # Registration
  └── zones/               # Pre-built zones
```

### Step 3: Integrate HeirOS as Optimizer
```bash
# HeirOS becomes a service
tidyllm/drop_zones/workflows/
  └── optimizer.py         # HeirOS optimization
```

## QUESTIONS & DECISIONS

1. **Should HeirOS remain separate or be fully integrated into drop_zones?**
   - Recommendation: Keep core HeirOS separate, integrate via HeirOSGateway

2. **How should zones specify their workflow?**
   - Option A: Python code (flexible)
   - Option B: YAML config (simple)
   - Recommendation: Support both

3. **Should we auto-optimize all workflows?**
   - Recommendation: Make it opt-in with `heiros_optimize: true`

4. **Gateway selection per zone or per file?**
   - Recommendation: Per zone default, with per-file override possible

## NEXT STEPS

1. **Approve this architecture**
2. **Start with gateway consolidation**
3. **Create BaseDropZone**
4. **Move one zone as proof of concept**
5. **Integrate HeirOS optimizer**
6. **Document the new pattern**

This makes drop_zones your primary work area, with clear patterns for adding new zones, while HeirOS becomes a powerful optimizer that cleans up the inevitable mess from manual workflow creation. Thoughts?