# Guidance on Model Selection

## What We Implemented

✅ **Provider-based architecture** using TidyLLM's Provider objects  
✅ **Dynamic model selection** via `bedrock()`, `claude()`, etc.  
✅ **Flow-based approach** using TidyLLM's verb system: `message | embed(provider)`  
✅ **Runtime provider switching** - can override provider per request  

## Clean Flow-Based Architecture

### Dynamic Provider Selection - No Hardcoded Models

```python
# Dynamic provider selection - no hardcoded models
from tidyllm import bedrock, claude

processor = EmbeddingProcessor(default_provider=bedrock())
embedding = processor.embed("text", provider=claude())  # Override provider
```

### TidyLLM Integration

```python
# Uses TidyLLM's verb system directly
from tidyllm.verbs import embed, llm_message

message = llm_message(text)
result = message | embed(provider)  # Provider determines model selection
```

## Benefits

- 🔄 **Dynamic**: Models selected at runtime based on Provider configuration
- 🏗️ **Configurable**: Provider objects handle model selection logic
- 🎯 **Flexible**: Can override provider per request for different models
- 📋 **Standards-compliant**: Uses TidyLLM's established Provider pattern

## Implementation Pattern

The facades now use **flow-based dynamic model selection** instead of hardcoded strings, allowing models to be selected dynamically through TidyLLM's Provider system.

### Core Principles

1. **No Hardcoded Models**: Model selection happens through Provider configuration
2. **Runtime Flexibility**: Switch providers/models per request as needed
3. **Flow-Based Design**: Use TidyLLM's pipe (`|`) syntax for clean composition
4. **Provider Abstraction**: Let Provider objects handle the model selection logic

### Example Usage

```python
from tidyllm.verbs import embed, llm_message
from tidyllm import bedrock, claude

# Default provider for all operations
processor = EmbeddingProcessor(default_provider=bedrock())

# Per-request provider override
result1 = processor.embed("text1", provider=claude())     # Uses Claude
result2 = processor.embed("text2", provider=bedrock())    # Uses Bedrock
result3 = processor.embed("text3")                        # Uses default (Bedrock)

# Flow-based composition with proper imports
message = llm_message("analyze this document")
result_frame = message | embed(claude())  # Provider determines model selection

# Extract embedding from DataTable result
if hasattr(result_frame, 'to_py'):
    embedding_data = result_frame.to_py()
    embedding = embedding_data[0][0] if embedding_data else []
```

## Implementation Details

### **✅ What The Current Implementation Gets Right:**

#### **1. Proper Import Structure:**
```python
from tidyllm.verbs import embed, llm_message
from tidyllm import bedrock, claude
```
- Import verbs from `tidyllm.verbs` module
- Import provider constructors from main `tidyllm` package
- Clean separation between verbs and providers

#### **2. Flow-Based Chaining Pattern:**
```python
message = llm_message(text)
result_frame = message | embed(provider)  # Correct chaining pattern
```
- Creates message object first
- Uses pipe (`|`) operator for chaining
- Provider object passed to verb, not hardcoded model string
- Returns DataTable result frame

#### **3. DataTable Result Handling:**
```python
if hasattr(result_frame, 'to_py'):
    embedding_data = result_frame.to_py()
    embedding = embedding_data[0][0] if embedding_data else []
```
- Check for DataTable interface with `hasattr()`
- Extract Python data with `.to_py()` method
- Handle nested array structure `[0][0]`
- Provide fallback for empty results

#### **4. Provider-Based Architecture:**
```python
# Constructor with default provider
processor = EmbeddingProcessor(default_provider=bedrock())

# Per-request provider override
result = processor.embed("text", provider=claude())
```
- Default provider set in constructor for consistency
- Per-request override maintains flexibility
- Provider objects handle model selection logic
- No hardcoded model strings anywhere

### **Critical Pattern Recognition:**

#### **DataTable Integration:**
TidyLLM verbs return DataTable objects, not raw arrays:
```python
# ❌ WRONG - Expecting raw embedding
embedding = message | embed(provider)  # This is a DataTable!

# ✅ CORRECT - Extract from DataTable
result_frame = message | embed(provider)
embedding = result_frame.to_py()[0][0]  # Extract actual embedding
```

#### **Provider vs Model Strings:**
```python
# ❌ OLD WAY - Hardcoded model strings
result = processor.embed("text", model="claude-3-sonnet")

# ✅ NEW WAY - Provider objects
result = processor.embed("text", provider=claude())
```

## Architectural Alignment

This approach aligns with TidyLLM's core architecture:

- **Provider Pattern**: Consistent with TidyLLM's abstraction layer
- **Verb System**: Uses established `message | verb(provider)` syntax
- **DataTable Integration**: Properly handles TidyLLM's data structures
- **Dynamic Configuration**: Models selected based on runtime context
- **Clean Separation**: Provider logic separate from processing logic

## Migration from Hardcoded Models

**❌ Old Approach:**
```python
# Hardcoded model strings - inflexible
embedding = processor.embed("text", model="claude-3-sonnet")
```

**✅ New Approach:**
```python
# Provider-based - flexible and configurable
embedding = processor.embed("text", provider=claude())
```

## **Real Implementation Code**

Here's the actual working code from `tidyllm/knowledge_systems/facades/embedding_processor.py`:

### **Class Constructor:**
```python
class EmbeddingProcessor:
    """Simple facade for embedding processing with automatic standardization"""
    
    def __init__(self, target_dimension: int = 1024, default_provider=None):
        """
        Initialize embedding processor with flow-based model selection
        
        Args:
            target_dimension: Target dimension for all embeddings
            default_provider: TidyLLM Provider object (e.g., bedrock()) - dynamic model selection
        """
        self.target_dimension = target_dimension
        self.default_provider = default_provider
        logger.info(f"EmbeddingProcessor initialized: {self.target_dimension}d target")
        logger.info("Using flow-based TidyLLM provider model selection")
```

### **Core Embed Method:**
```python
def embed(self, text: str, provider=None):
    """
    Generate embedding using TidyLLM's flow-based provider system
    
    Args:
        text: Text to embed  
        provider: TidyLLM Provider object (overrides default_provider)
    """
    # Use provided provider or fall back to default
    active_provider = provider or self.default_provider
    
    if not active_provider:
        raise ValueError("No provider specified and no default provider set")
    
    try:
        # Use TidyLLM's verb system with flow-based provider selection
        from tidyllm.verbs import embed, llm_message
        
        # Use TidyLLM's flow: message | embed(provider)
        message = llm_message(text)
        result_frame = message | embed(active_provider)
        
        # Extract embedding from DataTable result
        if hasattr(result_frame, 'to_py'):
            embedding_data = result_frame.to_py()
            embedding = embedding_data[0][0] if embedding_data and embedding_data[0] else []
            
            # Standardize to target dimension if needed
            if self.target_dimension and len(embedding) != self.target_dimension:
                embedding = self._standardize_embedding(embedding)
                
            return embedding
        else:
            # Fallback for direct embedding return
            return result_frame
            
    except Exception as e:
        logger.error(f"Embedding failed with provider {active_provider}: {e}")
        raise
```

### **Batch Processing Method:**
```python
def embed_batch(self, texts: List[str], provider=None) -> List:
    """Generate embeddings for multiple texts using flow-based provider"""
    return [self.embed(text, provider) for text in texts]
```

### **Quick Access Function:**
```python
def embed_text(text: str, provider=None) -> List[float]:
    """
    Quick function to embed text using TidyLLM's flow-based providers
    
    Args:
        text: Text to embed
        provider: TidyLLM Provider object (e.g., bedrock(), claude())
    """
    processor = EmbeddingProcessor(default_provider=provider)
    return processor.embed(text)
```

## **Usage Examples from Real Code:**

### **Basic Usage:**
```python
from tidyllm import bedrock, claude
from tidyllm.knowledge_systems.facades import EmbeddingProcessor

# Initialize with default provider
processor = EmbeddingProcessor(target_dimension=1024, default_provider=bedrock())

# Use default provider
embedding1 = processor.embed("Hello world")

# Override provider for specific request
embedding2 = processor.embed("Hello world", provider=claude())
```

### **Quick Function Usage:**
```python
from tidyllm.knowledge_systems.facades.embedding_processor import embed_text
from tidyllm import claude

# One-line embedding with provider
embedding = embed_text("Hello world", provider=claude())
```

This guidance ensures consistent, flexible model selection across all TidyLLM components while maintaining the established architectural patterns.