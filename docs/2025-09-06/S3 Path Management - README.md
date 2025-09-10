# S3 Path Management - README

**Document Version**: 1.0  
**Created**: 2025-09-06  
**Status**: Official S3 Path Construction Guide  
**Priority**: MANDATORY FOR S3-FIRST ARCHITECTURE

---

## 📋 **Executive Summary**

TidyLLM uses an **S3-First Architecture** with structured path management. This document provides clear guidance on S3 path construction, terminology, and the S3PathBuilder utility for consistent object storage across all gateways.

### **Key Terminology Clarification**
- **`workflow_identifier`**: Unique name/ID for the stored object (replaces confusing "key" parameter)
- **`folder_prefix`**: Base S3 path prefix from settings.yaml (replaces "requirement" terminology)
- **`full_s3_path`**: Complete S3 object key (folder_prefix + workflow_identifier)

---

## 🎯 **S3 Path Construction Logic**

### **Current Pattern**
```python
# How S3 paths are constructed
folder_prefix = "workflows/"              # From settings.yaml
workflow_identifier = "user_123_approval" # Your unique object name
full_s3_path = f"{folder_prefix}{workflow_identifier}"

# Result: "workflows/user_123_approval"
s3_client.put_object(
    Bucket="your-tidyllm-bucket",
    Key=full_s3_path,  # "workflows/user_123_approval"
    Body=json.dumps(data)
)
```

### **Settings.yaml Alignment**
```yaml
# From your current settings.yaml
s3:
  prefixes:
    knowledge_base: "knowledge_base/"     # Base: knowledge_base/ + subfolder/object_id
    mvr_analysis: "mvr_analysis/"         # Base: mvr_analysis/ + subfolder/object_id
    workflows: "workflows/"               # Base: workflows/ + object_id  
    pages: "pages/"                       # Base: pages/ + domain/object_id
    embeddings: "embeddings/"             # Base: embeddings/ + type/object_id
    metadata: "metadata/"                 # Base: metadata/ + category/object_id
    temp: "temp/"                         # Base: temp/ + processing/object_id
```

---

## 🏗️ **S3PathBuilder Implementation**

### **Core S3PathBuilder Class**
```python
import json
from typing import Dict, Optional, Any
from dataclasses import dataclass

@dataclass
class S3PathComponents:
    """Structured representation of S3 path components"""
    content_type: str           # e.g., 'workflows', 'knowledge_base'
    object_identifier: str      # e.g., 'user_123_approval', 'sop_document_456'
    subfolder: Optional[str]    # e.g., 'approved', 'pending', 'reports'
    file_extension: Optional[str] = None  # e.g., '.json', '.txt'

class S3PathBuilder:
    """Build S3 paths consistently across TidyLLM"""
    
    def __init__(self, s3_settings: Dict[str, Any]):
        """
        Initialize with S3 settings from settings.yaml
        
        Args:
            s3_settings: The 's3' section from settings.yaml
        """
        self.base_prefixes = s3_settings.get('prefixes', {})
        self.bucket = s3_settings.get('bucket', 'default-bucket')
        self.region = s3_settings.get('region', 'us-east-1')
        
        # Environment-specific overrides
        self.environment = s3_settings.get('environments', {})
        
    def build_path(self, content_type: str, object_identifier: str, 
                   subfolder: Optional[str] = None, 
                   file_extension: Optional[str] = None) -> str:
        """
        Build full S3 path with clear parameter names
        
        Args:
            content_type: Type from settings.yaml prefixes ('workflows', 'knowledge_base', etc.)
            object_identifier: Unique identifier for the object
            subfolder: Optional sub-categorization within the content type
            file_extension: Optional file extension (will auto-add dot if missing)
            
        Returns:
            str: Full S3 object key
            
        Examples:
            build_path('workflows', 'user_approval_123') 
            → "workflows/user_approval_123"
            
            build_path('workflows', 'user_approval_123', 'approved', '.json')
            → "workflows/approved/user_approval_123.json"
            
            build_path('knowledge_base', 'sop_document', 'procedures', '.pdf')
            → "knowledge_base/procedures/sop_document.pdf"
        """
        if content_type not in self.base_prefixes:
            raise ValueError(f"Unknown content_type: {content_type}. Available: {list(self.base_prefixes.keys())}")
            
        # Get base prefix
        base_prefix = self.base_prefixes[content_type]
        
        # Build path components
        path_parts = [base_prefix.rstrip('/')]
        
        if subfolder:
            path_parts.append(subfolder.strip('/'))
            
        # Add object identifier
        if file_extension:
            if not file_extension.startswith('.'):
                file_extension = f".{file_extension}"
            object_name = f"{object_identifier}{file_extension}"
        else:
            object_name = object_identifier
            
        path_parts.append(object_name)
        
        return '/'.join(path_parts)
    
    def parse_path(self, s3_key: str) -> S3PathComponents:
        """
        Parse an S3 key back into components
        
        Args:
            s3_key: Full S3 object key
            
        Returns:
            S3PathComponents: Parsed components
            
        Example:
            parse_path("workflows/approved/user_123.json")
            → S3PathComponents(content_type='workflows', subfolder='approved', 
                              object_identifier='user_123', file_extension='.json')
        """
        # Find matching content type
        content_type = None
        remaining_path = s3_key
        
        for ct, prefix in self.base_prefixes.items():
            if s3_key.startswith(prefix):
                content_type = ct
                remaining_path = s3_key[len(prefix):]
                break
                
        if not content_type:
            raise ValueError(f"Could not determine content_type from S3 key: {s3_key}")
            
        # Parse remaining path
        parts = remaining_path.split('/')
        
        if len(parts) == 1:
            # No subfolder: "object_identifier.ext"
            subfolder = None
            filename = parts[0]
        else:
            # Has subfolder: "subfolder/object_identifier.ext"  
            subfolder = '/'.join(parts[:-1])
            filename = parts[-1]
            
        # Extract file extension
        if '.' in filename:
            object_identifier, file_extension = filename.rsplit('.', 1)
            file_extension = f".{file_extension}"
        else:
            object_identifier = filename
            file_extension = None
            
        return S3PathComponents(
            content_type=content_type,
            object_identifier=object_identifier,
            subfolder=subfolder,
            file_extension=file_extension
        )
    
    def get_environment_path(self, content_type: str, object_identifier: str, 
                           environment: str = 'production',
                           subfolder: Optional[str] = None) -> str:
        """
        Build path with environment prefix
        
        Args:
            content_type: Type from settings.yaml prefixes  
            object_identifier: Unique identifier
            environment: Environment ('development', 'staging', 'production')
            subfolder: Optional subfolder
            
        Returns:
            str: Environment-prefixed S3 path
            
        Example:
            get_environment_path('workflows', 'user_123', 'development')
            → "dev/workflows/user_123"
        """
        base_path = self.build_path(content_type, object_identifier, subfolder)
        
        env_config = self.environment.get(environment, {})
        prefix_override = env_config.get('prefix_override', '')
        
        if prefix_override:
            return f"{prefix_override}{base_path}"
        else:
            return base_path
```

---

## 💻 **Updated UnifiedSessionManager**

### **Enhanced Implementation**
```python
import boto3
import psycopg2
from typing import Dict, Any, Optional
from datetime import datetime
import json

class UnifiedSessionManager:
    """S3-First session manager with clear terminology"""
    
    def __init__(self, settings: Optional[Dict] = None):
        # Load settings
        if settings:
            self.settings = settings
        else:
            # Load from settings.yaml
            import yaml
            with open('tidyllm/admin/settings.yaml', 'r') as f:
                self.settings = yaml.safe_load(f)
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3', region_name=self.settings['s3']['region'])
        self.bucket_name = self.settings['s3']['bucket']
        
        # Initialize path builder
        self.path_builder = S3PathBuilder(self.settings['s3'])
        
        # Initialize database connection
        pg_config = self.settings['postgres']
        self.db_connection = psycopg2.connect(
            host=pg_config['host'],
            database=pg_config['db_name'], 
            user=pg_config['db_user'],
            password=pg_config['db_password'],
            sslmode=pg_config.get('ssl_mode', 'require')
        )
    
    def store_object_s3(self, data: Dict[str, Any], content_type: str, 
                       object_identifier: str, subfolder: Optional[str] = None) -> bool:
        """
        Store any object type in S3 using structured path construction
        
        Args:
            data: The data to store (will be JSON serialized)
            content_type: Type from settings.yaml prefixes ('workflows', 'knowledge_base', etc.)
            object_identifier: Unique identifier for this object
            subfolder: Optional sub-categorization
            
        Returns:
            bool: Success/failure of storage operation
            
        Examples:
            store_object_s3(workflow_data, 'workflows', 'user_123_approval')
            → Stores to: "workflows/user_123_approval"
            
            store_object_s3(sop_data, 'knowledge_base', 'procedure_456', 'sops')  
            → Stores to: "knowledge_base/sops/procedure_456"
        """
        try:
            s3_path = self.path_builder.build_path(content_type, object_identifier, subfolder, '.json')
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_path,
                Body=json.dumps(data, indent=2, default=str),
                ContentType='application/json'
            )
            
            print(f"✅ Stored object to S3: s3://{self.bucket_name}/{s3_path}")
            return True
            
        except Exception as e:
            print(f"❌ S3 storage failed: {e}")
            return False
    
    def get_object_s3(self, content_type: str, object_identifier: str, 
                      subfolder: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve object from S3 using structured path construction
        
        Args:
            content_type: Type used when storing the object
            object_identifier: Unique identifier used when storing 
            subfolder: Subfolder used when storing (must match)
            
        Returns:
            Dict or None: Retrieved data or None if not found
        """
        try:
            s3_path = self.path_builder.build_path(content_type, object_identifier, subfolder, '.json')
            
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_path
            )
            
            data = json.loads(response['Body'].read())
            print(f"✅ Retrieved object from S3: s3://{self.bucket_name}/{s3_path}")
            return data
            
        except Exception as e:
            print(f"❌ S3 retrieval failed: {e}")
            return None
    
    def list_objects_s3(self, content_type: str, subfolder: Optional[str] = None, 
                       max_keys: int = 100) -> List[str]:
        """
        List objects of a specific content type
        
        Args:
            content_type: Type to list ('workflows', 'knowledge_base', etc.)
            subfolder: Optional subfolder filter
            max_keys: Maximum number of keys to return
            
        Returns:
            List[str]: List of object identifiers (without path prefixes)
        """
        try:
            prefix = self.path_builder.build_path(content_type, '', subfolder).rstrip('/')
            if prefix and not prefix.endswith('/'):
                prefix += '/'
                
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            object_identifiers = []
            for obj in response.get('Contents', []):
                # Parse S3 key back to components
                try:
                    components = self.path_builder.parse_path(obj['Key'])
                    object_identifiers.append(components.object_identifier)
                except:
                    # If parsing fails, include raw key
                    object_identifiers.append(obj['Key'])
                    
            return object_identifiers
            
        except Exception as e:
            print(f"❌ S3 listing failed: {e}")
            return []
    
    # Convenience methods for common content types
    def store_workflow_s3(self, workflow_data: Dict[str, Any], workflow_identifier: str, 
                         subfolder: Optional[str] = None) -> bool:
        """Store workflow with clear naming"""
        return self.store_object_s3(workflow_data, 'workflows', workflow_identifier, subfolder)
    
    def get_workflow_s3(self, workflow_identifier: str, 
                       subfolder: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve workflow with clear naming"""
        return self.get_object_s3('workflows', workflow_identifier, subfolder)
    
    def store_knowledge_s3(self, knowledge_data: Dict[str, Any], knowledge_identifier: str,
                          knowledge_type: str = 'general') -> bool:
        """Store knowledge base item"""
        return self.store_object_s3(knowledge_data, 'knowledge_base', knowledge_identifier, knowledge_type)
    
    def get_knowledge_s3(self, knowledge_identifier: str, 
                        knowledge_type: str = 'general') -> Optional[Dict[str, Any]]:
        """Retrieve knowledge base item"""
        return self.get_object_s3('knowledge_base', knowledge_identifier, knowledge_type)
```

---

## 🔧 **Usage Examples**

### **Basic Object Storage**
```python
from scripts.start_unified_sessions import UnifiedSessionManager

# Initialize session manager
session_mgr = UnifiedSessionManager()

# Store a workflow
workflow_data = {
    "name": "Document Processing Workflow",
    "version": "1.0",
    "steps": [
        {"type": "upload", "config": {"max_size": "100MB"}},
        {"type": "validate", "config": {"schema": "document_v1"}}, 
        {"type": "process", "config": {"ai_model": "claude-3-sonnet"}}
    ]
}

# Store with clear identifier
success = session_mgr.store_workflow_s3(
    workflow_data, 
    workflow_identifier="document_processing_v1",
    subfolder="approved"
)
# Result: Stored to "workflows/approved/document_processing_v1.json"

# Retrieve the same workflow
retrieved = session_mgr.get_workflow_s3(
    workflow_identifier="document_processing_v1", 
    subfolder="approved"
)
```

### **Knowledge Base Storage**
```python
# Store SOP document
sop_data = {
    "title": "Model Validation Procedure",
    "version": "2.1", 
    "content": "Step-by-step validation process...",
    "compliance_frameworks": ["SOX", "Internal"]
}

session_mgr.store_knowledge_s3(
    sop_data,
    knowledge_identifier="model_validation_procedure_v2_1",
    knowledge_type="sops"
)
# Result: Stored to "knowledge_base/sops/model_validation_procedure_v2_1.json"
```

### **MVR Analysis Storage**
```python
# Store MVR analysis results
mvr_results = {
    "analysis_id": "mvr_2025_q1",
    "model_performance": {"accuracy": 0.94, "precision": 0.91},
    "validation_results": {"passed": True, "issues": []},
    "compliance_check": {"sox_compliant": True, "review_date": "2025-09-06"}
}

session_mgr.store_object_s3(
    mvr_results,
    content_type="mvr_analysis", 
    object_identifier="mvr_2025_q1_results",
    subfolder="reports"
)
# Result: Stored to "mvr_analysis/reports/mvr_2025_q1_results.json"
```

### **Environment-Specific Storage**
```python
# Store in development environment
dev_path = session_mgr.path_builder.get_environment_path(
    content_type="workflows",
    object_identifier="test_workflow_123",
    environment="development"
)
# Result: "dev/workflows/test_workflow_123" (if dev prefix configured)
```

---

## 🔍 **Path Analysis and Debugging**

### **Path Component Analysis**
```python
# Analyze any S3 path
s3_key = "knowledge_base/sops/model_validation_v2_1.json"
components = session_mgr.path_builder.parse_path(s3_key)

print(f"Content Type: {components.content_type}")        # "knowledge_base"
print(f"Object ID: {components.object_identifier}")      # "model_validation_v2_1"  
print(f"Subfolder: {components.subfolder}")              # "sops"
print(f"Extension: {components.file_extension}")         # ".json"
```

### **Path Validation**
```python
def validate_s3_paths():
    """Validate S3 path construction consistency"""
    
    test_cases = [
        ("workflows", "user_approval_123", None),
        ("workflows", "user_approval_123", "pending"),
        ("knowledge_base", "sop_document", "procedures"), 
        ("mvr_analysis", "analysis_456", "reports")
    ]
    
    for content_type, obj_id, subfolder in test_cases:
        # Build path
        path = session_mgr.path_builder.build_path(content_type, obj_id, subfolder, '.json')
        
        # Parse path back
        components = session_mgr.path_builder.parse_path(path)
        
        # Verify round-trip consistency
        assert components.content_type == content_type
        assert components.object_identifier == obj_id
        assert components.subfolder == subfolder
        assert components.file_extension == '.json'
        
        print(f"✅ Path validation passed: {path}")
```

---

## 📊 **Migration from Old Terminology**

### **Parameter Name Mapping**
| Old Parameter | New Parameter | Description |
|---------------|---------------|-------------|
| `key` | `object_identifier` | Unique identifier for the stored object |
| `requirement` | `content_type` | Type from settings.yaml prefixes |  
| (none) | `folder_prefix` | Base S3 path prefix (auto-derived from content_type) |
| (none) | `subfolder` | Optional sub-categorization within content type |

### **Method Name Updates**
| Old Method | New Method | Improvement |
|------------|------------|-------------|
| `store_workflow_s3(data, key)` | `store_workflow_s3(data, workflow_identifier, subfolder=None)` | Clear parameter names |
| `get_workflow_s3(key)` | `get_workflow_s3(workflow_identifier, subfolder=None)` | Consistent naming |
| (none) | `store_object_s3(data, content_type, object_identifier, subfolder=None)` | Generic method for any content type |

### **Migration Script**
```python
def migrate_old_s3_calls():
    """Update old S3 method calls to new terminology"""
    
    # Old way (confusing)
    # session_mgr.store_workflow_s3(workflow_data, "user_123.json")
    
    # New way (clear)
    session_mgr.store_workflow_s3(
        workflow_data, 
        workflow_identifier="user_123",
        subfolder="approved"  # Optional organization
    )
    
    # For non-workflow objects, use generic method
    session_mgr.store_object_s3(
        data=analysis_results,
        content_type="mvr_analysis", 
        object_identifier="analysis_456",
        subfolder="reports"
    )
```

---

## 🚨 **Best Practices**

### **Object Identifier Naming**
```python
# ✅ GOOD - Clear, descriptive identifiers
"user_approval_workflow_v2_1" 
"model_validation_procedure_2025"
"mvr_analysis_q1_results"
"compliance_report_sox_2025_09"

# ❌ BAD - Ambiguous or generic identifiers
"workflow123"
"document" 
"data_file"
"temp_analysis"
```

### **Subfolder Organization**
```python
# ✅ GOOD - Logical subfolder hierarchy
workflows/
├── approved/           # Production-ready workflows
├── pending/           # Awaiting approval  
├── archived/          # No longer active
└── templates/         # Reusable templates

knowledge_base/
├── sops/              # Standard Operating Procedures
├── checklists/        # Validation checklists  
├── modeling/          # Model documentation
└── compliance/        # Compliance frameworks

mvr_analysis/
├── raw/               # Raw analysis data
├── reports/           # Processed reports
└── embeddings/        # Generated embeddings
```

### **Error Handling**
```python
# Always handle S3 operations gracefully
try:
    success = session_mgr.store_workflow_s3(
        workflow_data,
        workflow_identifier="critical_workflow", 
        subfolder="production"
    )
    
    if not success:
        logger.error("Failed to store workflow - check S3 permissions and bucket access")
        # Implement fallback strategy
        
except Exception as e:
    logger.error(f"S3 storage exception: {e}")
    # Handle network issues, credential problems, etc.
```

---

## 🎯 **Quick Reference**

### **Common Content Types**
| Content Type | Prefix | Common Subfolders | Example Use |
|-------------|--------|------------------|-------------|
| `workflows` | `workflows/` | approved, pending, archived, templates | Business process workflows |
| `knowledge_base` | `knowledge_base/` | sops, checklists, modeling, compliance | Documentation and procedures |
| `mvr_analysis` | `mvr_analysis/` | raw, reports, embeddings | Model validation results |
| `pages` | `pages/` | domain, category | Web content or documentation pages |
| `embeddings` | `embeddings/` | tfidf, sentence | AI model embeddings |
| `metadata` | `metadata/` | processing, tracking | System metadata and logs |
| `temp` | `temp/` | uploads, processing | Temporary files and processing data |

### **S3PathBuilder Quick Usage**
```python
# Initialize (typically done once in UnifiedSessionManager)
path_builder = S3PathBuilder(settings['s3'])

# Build paths
workflow_path = path_builder.build_path('workflows', 'user_123', 'approved', '.json')
# Result: "workflows/approved/user_123.json"

# Parse paths  
components = path_builder.parse_path('workflows/approved/user_123.json')
# Result: S3PathComponents(content_type='workflows', object_identifier='user_123', ...)
```

---

**Document Location**: `/docs/2025-09-06/S3 Path Management - README.md`  
**Last Updated**: 2025-09-06  
**Status**: Official S3 Path Construction Guide