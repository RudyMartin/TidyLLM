# MLflow Tests

Comprehensive test suite for MLflow integration, session tracking, and activity monitoring in TidyLLM.

## Test Files

### 1. `test_session_tracking.py`
Tests user session parameter tracking across all chat modes.

**Features:**
- Tests `user_id`, `session_id`, `audit_reason` parameters
- Validates session consistency across chat modes
- Checks audit trail completeness
- Analyzes token usage tracking
- Integrates with MLflow viewer

**Usage:**
```bash
cd tidyllm/tests/mlflow
python test_session_tracking.py
```

**What it tests:**
- ✅ Session parameters in `direct` mode
- ✅ Session parameters in `dspy` mode
- ✅ Session parameters in `rag` mode
- ✅ Session parameters in `hybrid` mode
- ✅ MLflow service integration
- ✅ Parameter consistency validation

### 2. `test_activity_viewer.py`
Tests MLflow activity viewers and recent activity tracking.

**Features:**
- Tests `show_last_5_mlflow_records()` function
- Tests `show_missing_evidence()` function
- Generates test activity and checks MLflow visibility
- Analyzes session parameter coverage in MLflow

**Usage:**
```bash
cd tidyllm/tests/mlflow
python test_activity_viewer.py
```

**What it tests:**
- ✅ MLflow viewer functionality
- ✅ Evidence checker functionality
- ✅ Recent activity appears in MLflow
- ✅ Session parameters tracked correctly
- ✅ Token metrics logged properly

## Session Parameters Tracked

The system tracks comprehensive user session data:

### User Context
- `user_id` - Unique user identifier
- `session_id` - Session identifier for conversation tracking
- `audit_reason` - Purpose/reason for the request

### Request Parameters
- `model_id` - LLM model used
- `temperature` - Model temperature setting
- `max_tokens` - Maximum token limit
- `processing_time_ms` - Request processing time

### Token Metrics
- `input_tokens` - Input token count
- `output_tokens` - Output token count
- `total_tokens` - Total token usage

### Audit Trail
- `timestamp` - Request timestamp
- `success` - Request success status
- `error` - Error message if failed

## MLflow Infrastructure

### Viewer Functions
Located in `tidyllm/infrastructure/tools/`:

- **`mlflow_viewer.py`** - Shows last 5 MLflow records with full details
- **`mlflow_evidence_checker.py`** - Shows missing evidence for compliance

### Database Backend
- **Backend**: PostgreSQL on AWS RDS
- **Database**: vectorqa
- **Connection**: Managed through unified sessions

## Running All Tests

```bash
# Run session tracking tests
python test_session_tracking.py

# Run activity viewer tests
python test_activity_viewer.py

# Run both tests in sequence
python test_session_tracking.py && python test_activity_viewer.py
```

## Expected Results

### Successful Test Output
```
✅ Chat modes tested: 4/4
✅ Session tracking working: True
✅ MLflow service available: True
✅ Parameter coverage complete: True
✅ Activity viewers working: True
🎉 ALL MLFLOW TESTS PASSED!
```

### What Success Means
- All chat modes (direct, dspy, rag, hybrid) log to MLflow
- User session parameters are captured consistently
- Token usage is tracked accurately
- Audit trails are complete and compliant
- Recent activity is visible in MLflow viewers
- No missing evidence or logging gaps

## Troubleshooting

### Common Issues

**MLflow Connection Failed**
- Check PostgreSQL credentials in settings.yaml
- Verify AWS RDS connectivity
- Ensure MLflow service is configured

**Missing Session Parameters**
- Verify `log_llm_request` method is working
- Check audit trail generation in chat responses
- Ensure user_id is passed to chat functions

**No Recent Activity**
- Wait 3-5 seconds after chat for MLflow processing
- Check if MLflow service `is_available()` returns True
- Verify MLflow integration service is initialized

### Manual Verification

```python
# Check MLflow service status
from tidyllm.services import MLflowIntegrationService
service = MLflowIntegrationService()
print("Available:", service.is_available())
print("Connected:", service.is_connected)

# View recent activity
from tidyllm.infrastructure.tools import show_last_5_mlflow_records
show_last_5_mlflow_records()
```

## Integration with Chat Tests

These MLflow tests complement the chat tests in `tidyllm/review/chat/tests/`:
- Chat tests verify functionality
- MLflow tests verify logging and tracking
- Together they ensure complete enterprise compliance

## Next Steps

After running these tests successfully:
1. Monitor MLflow dashboard for ongoing activity
2. Set up automated MLflow monitoring
3. Configure cost tracking alerts
4. Implement MLflow experiment organization
5. Add performance metrics to MLflow logging