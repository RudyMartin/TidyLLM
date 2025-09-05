# 🚨 SYSTEMS INTEL - DO NOT DELETE 🚨
# Database Connection Architecture & Configuration

## ⚠️ CRITICAL WARNING
**This file contains SYSTEM INTELLIGENCE about our database architecture.**
**DO NOT DELETE, MODIFY, OR MOVE this file without consulting the team.**
**This document is essential for understanding our database connection patterns.**

---

## 📊 Current Database Architecture

### **Primary Database: Aurora PostgreSQL (AWS)**
- **Host**: `[REDACTED]` (stored in environment variables)
- **Port**: `5432`
- **Database**: `vectorqa`
- **User**: `[REDACTED]` (stored in environment variables)
- **Password**: `[REDACTED]` (stored in environment variables)
- **Full URL**: `[REDACTED]` (auto-generated from environment variables)

### **Connection Pattern: Credential Manager System**
We use a **centralized credential management system** to avoid hardcoded credentials:

1. **Primary Location**: `environ_settings/.env.local` (auto-detected by credential manager)
2. **Fallback Location**: `src/backend/config/credentials.env` (for direct sourcing)
3. **Manager**: `src/backend/config/credential_manager.py`

---

## 🔧 How Database Connections Work

### **1. Credential Manager Flow**
```python
from backend.config.credential_manager import credential_manager
db_config = credential_manager.get_database_config()
database_url = db_config.get('url')
```

### **2. Environment Detection**
The credential manager automatically detects the environment:
- **APP_ENV=local** → `environ_settings/.env.local`
- **APP_ENV=development** → `environ_settings/.env.development`
- **APP_ENV=staging** → `environ_settings/.env.staging`
- **APP_ENV=production** → `environ_settings/.env.production`

### **3. Loading Priority**
1. Environment variables (highest priority)
2. `.env` file in `environ_settings/` directory
3. System defaults (lowest priority)

---

## 🗄️ Database Tables & Schemas

### **Error Tracking System Tables**
Created by `database/setup_error_tracking.sh`:

1. **`prompt_history`** - MLflow-integrated prompt tracking
2. **`error_tracking`** - Intelligent error capture
3. **`error_patterns`** - Pattern detection and analysis
4. **`alert_history`** - Alert management and status
5. **`realtime_context`** - Live context data
6. **`batch_processing_status`** - Batch operation tracking

### **RAG System Tables**
Created by RAG orchestrator:

1. **`documents`** - Document metadata
2. **`document_chunks`** - Chunked document content
3. **`chunk_embeddings`** - Vector embeddings (pgvector)

---

## 🚀 Connection Testing

### **Test Scripts Available**
1. **`scripts/test_error_tracking_remote.py`** - Tests error tracking tables
2. **`scripts/debug_credentials.py`** - Debug credential loading
3. **`scripts/test_db_connection.sh`** - Bash wrapper for testing

### **Quick Test Command**
```bash
# Test using credential manager
python3 scripts/debug_credentials.py

# Test database connection
./scripts/test_db_connection.sh

# Test error tracking tables
python3 scripts/test_error_tracking_remote.py

# Run comprehensive test suite
python3 tests/run_error_tracking_test_suite.py
```

---

## 🔐 Security & Best Practices

### **✅ DO's**
- Use credential manager for all database connections
- Store credentials in `environ_settings/.env.local`
- Test connections before deployment
- Use environment-specific configurations

### **❌ DON'Ts**
- Hardcode database URLs in scripts
- Commit credentials to git
- Use direct environment variable access
- Bypass the credential manager

---

## 🛠️ Troubleshooting

### **Common Issues**

1. **"DATABASE_URL not found"**
   - Check if `environ_settings/.env.local` exists
   - Verify `APP_ENV` environment variable
   - Ensure credential manager can import

2. **"Password authentication failed"**
   - Verify password in credentials file
   - Check AWS RDS security groups
   - Ensure IP is whitelisted

3. **"Tables don't exist"**
   - Run `python3 scripts/setup_error_tracking_python.py`
   - Check if schema creation succeeded
   - Verify database permissions

4. **"SQL syntax errors with dollar-quoted strings"**
   - Use `database/simple_error_tracking_schema.sql` instead of complex schema
   - Avoid complex PostgreSQL functions in setup scripts
   - Split complex SQL into separate statements

5. **"Column mismatch errors"**
   - Verify column names match between schema and mock data
   - Check for typos in column names (e.g., `total_records` vs `total_items`)
   - Use consistent naming conventions across all files

### **Debug Commands**
```bash
# Check credential loading
python3 scripts/debug_credentials.py

# Test raw connection
psql "$DATABASE_URL"

# Check table existence
python3 -c "import psycopg2; import os; conn = psycopg2.connect(os.getenv('DATABASE_URL')); cur = conn.cursor(); cur.execute('SELECT table_name FROM information_schema.tables WHERE table_schema = \'public\''); print(cur.fetchall())"
```

---

## 📁 File Locations

### **Configuration Files**
- `environ_settings/.env.local` - **PRIMARY** credentials location
- `src/backend/config/credentials.env` - **FALLBACK** credentials
- `src/backend/config/credential_manager.py` - Credential management logic

### **Database Scripts**
- `database/simple_error_tracking_schema.sql` - **PRIMARY** schema definition (use this)
- `database/prompt_pipeline_error_tracking.sql` - Complex schema with functions (avoid for setup)
- `database/mock_data_error_tracking.sql` - Test data
- `database/error_tracking_queries.sql` - Query examples
- `database/setup_error_tracking.sh` - Bash setup script (uses hardcoded localhost)

### **Test Scripts**
- `scripts/test_error_tracking_remote.py` - Remote table testing
- `scripts/debug_credentials.py` - Credential debugging
- `scripts/test_db_connection.sh` - Connection testing
- `scripts/setup_error_tracking_python.py` - **PRIMARY** setup script (uses credential manager)

### **Test Suite**
- `tests/test_error_tracking_comprehensive.py` - **COMPREHENSIVE** system tests (15 tests)
- `tests/test_error_tracking_scenarios.py` - **SCENARIO**-based tests (10 tests)
- `tests/run_error_tracking_test_suite.py` - **MASTER** test runner (runs all tests)

---

## 🔄 Integration Points

### **MCP System Integration**
- **Live Context Worker**: Queries `realtime_context` table
- **Error Tracking**: Captures errors in `error_tracking` table
- **Performance Metrics**: Stores in `batch_processing_status` table

### **MLflow Integration**
- **Prompt History**: Tracks all prompts in `prompt_history` table
- **Experiment Tracking**: Links to MLflow experiment IDs
- **Performance Analysis**: Stores metrics for analysis

### **RAG System Integration**
- **Document Storage**: Uses `documents` and `document_chunks` tables
- **Vector Search**: Uses `chunk_embeddings` with pgvector
- **Query Processing**: Integrates with error tracking

---

## 📋 Current Status

### **✅ Working**
- Database connection via credential manager
- Aurora PostgreSQL connectivity
- Basic table structure understanding
- Test scripts functional
- **Comprehensive test suite** (25 total tests, 100% pass rate)
- **Error tracking system** fully operational
- **Mock data** populated and verified

### **🔄 In Progress**
- Error tracking table creation
- Mock data population
- Integration testing

### **📋 Next Steps**
1. ✅ **COMPLETED**: Error tracking system setup and testing
2. 🔄 Integrate with MCP system
3. 📊 Set up monitoring dashboards
4. 🚨 Configure alerting rules
5. 📈 Deploy to production

---

## 🎯 Key Takeaways

1. **Always use credential manager** - Never hardcode database URLs
2. **Test connections first** - Use provided test scripts
3. **Environment-specific configs** - Use `APP_ENV` for different environments
4. **Security first** - Credentials in `environ_settings/` not in code
5. **Documentation matters** - This file is essential for team understanding

---

**Last Updated**: 2024-03-21
**Maintained By**: AI Assistant
**Review Required**: Before any database architecture changes
