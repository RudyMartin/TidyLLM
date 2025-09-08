# QA Control Drop Zone

## Expected File Structure

Drop an Excel file (.xlsx) with the following naming pattern:
- `REV00001_qa_control.xlsx`
- `REV12345_qa_control.xlsx`

## Required Excel Tabs

1. **core_checklist** - Core QA metrics (required)
   - Columns: metric_id, section, question, expected_answer, severity
   
2. **custom_checklist** - Custom QA items (optional)
   - Columns: metric_id, section, question, expected_answer, severity
   
3. **custom_prompts** - Custom analysis prompts (optional)
   - Columns: prompt_id, prompt_text, prompt_type, context

## Pipeline Flow

1. Drop Excel file here
2. System detects and processes through:
   - qa_ingest: Raw file copy
   - qa_extract: Parse Excel tabs to JSON
   - qa_embed: Generate embeddings
   - qa_index: Build search index
   - qa_analysis: Run QA analysis
   - qa_reports: Generate final report

## FLOW Agreement Usage

You can also trigger via FLOW Agreement:
```bash
python 1-enterprise.py "[QA Control]"
```
