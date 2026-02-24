# TidyLLM Infrastructure Pattern

## The Pattern: Adapter/Delegate with Parent Detection

### How It Works

```
RAG Adapter
    ↓
Infrastructure Delegate (single point of access)
    ↓
Parent Detection (at initialization)
    ├── IF Parent Available → Use ResilientPoolManager, aws_service, etc.
    └── IF Standalone → Use simple fallback implementations
```

### Key Design Principles

1. **Single Delegate**: One delegate handles ALL infrastructure needs
2. **Parent First**: Always try to use parent infrastructure first
3. **Graceful Fallback**: Only create simple implementations if parent not available
4. **No Duplication**: Never reimplement what parent already provides

### The Code Pattern

```python
class InfrastructureDelegate:
    """
    Single delegate for all infrastructure access.

    PATTERN: Try parent infrastructure first, fallback if not available.
    WHY: Reuse enterprise features when deployed, work standalone when developing.
    """

    def __init__(self):
        # Try parent infrastructure first
        self._db_pool = self._init_db_pool()
        self._aws = self._init_aws()

    def _init_db_pool(self):
        """
        Initialize database pool.

        PRIORITY:
        1. Try parent's ResilientPoolManager (3-pool failover)
        2. Fallback to simple psycopg2 pool (basic but works)
        """
        try:
            from infrastructure.services.resilient_pool_manager import ResilientPoolManager
            return ResilientPoolManager()  # Enterprise features!
        except ImportError:
            import psycopg2.pool
            return psycopg2.pool.SimpleConnectionPool(1, 5, ...)  # Simple fallback
```

### Why This Pattern?

1. **No Duplication**: We don't reimplement ResilientPoolManager
2. **Progressive Enhancement**: Get enterprise features when available
3. **Developer Friendly**: Works in any environment
4. **Clean Boundaries**: Adapters don't know/care about infrastructure details

### Usage in RAG Adapters

```python
class AIPoweredRAGAdapter:
    def __init__(self):
        # One line - gets best available infrastructure
        self.infra = InfrastructureDelegate()

    def query(self, request):
        # Use infrastructure without caring about implementation
        conn = self.infra.get_db_connection()  # Could be ResilientPool or SimplePool
        # ... do work
        self.infra.return_connection(conn)
```

### What We're NOT Doing

❌ Multiple delegate files for each service
❌ Complex factory patterns
❌ Reimplementing parent services
❌ Runtime switching between implementations

### What We ARE Doing

✅ One delegate for all infrastructure
✅ Simple try/except for parent detection
✅ Direct usage, no factories
✅ Reusing parent services when available