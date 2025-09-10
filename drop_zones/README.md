# TidyLLM Drop Zones - AI Dropzone Manager Triggers
=====================================================

## Directory Structure
Each folder automatically triggers the corresponding bracket command when documents are added:

```
drop_zones/
├── mvr_analysis/          → Triggers [Process MVR]
├── financial_analysis/    → Triggers [Financial Analysis]  
├── contract_review/       → Triggers [Contract Review]
├── compliance_check/      → Triggers [Compliance Check]
├── quality_check/         → Triggers [Quality Check]
├── data_extraction/       → Triggers [Data Extraction]
├── processing/            → Active processing files
├── completed/             → Successfully processed files
└── failed/                → Failed processing files
```

## How It Works
1. **Drop Document**: Place PDF/document in any workflow folder
2. **Auto Detection**: AI Dropzone Manager detects new file
3. **Document Intelligence**: Analyzes document type and complexity
4. **Worker Assignment**: Selects appropriate workers and templates
5. **Processing**: Executes analysis through CorporateLLMGateway
6. **Results**: Moves to completed/ with analysis results

## Example Usage
```bash
# Drop document for MVR analysis
cp your_mvr_report.pdf tidyllm/drop_zones/mvr_analysis/

# Drop contract for legal review  
cp contract.pdf tidyllm/drop_zones/contract_review/

# Monitor processing
python test_ai_dropzone_manager.py
```

## Security Features
- Only approved bracket commands from registry
- All AI calls through CorporateLLMGateway
- Complete audit trail of processing
- Sandboxed worker execution
- Template validation and approval required

## Monitoring
Check processing status:
- `processing/` - Currently being analyzed
- `completed/` - Analysis finished with results
- `failed/` - Processing errors with logs