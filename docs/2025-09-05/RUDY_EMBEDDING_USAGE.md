# Rudy's Embedding Database Monitor Usage

## 🎯 **What This Script Does**
Monitor PostgreSQL database for embedding generation progress in TidyLLM system.

## 📋 **Quick Usage Examples**

### **1. Quick Summary (Recommended)**
```bash
python rudy_test_embeddings.py --summary
```
**Output Example:**
```
[SEARCH] RUDY'S EMBEDDING QUICK SUMMARY
==================================================
[STATS] RECORD COUNTS:
   chunk_embeddings    :          0
   paper_embeddings    :          0  
   document_chunks     :        186

[DATE] DATE RANGE:
   From: 2025-08-24
   To:   2025-08-25

[OK] EMBEDDING STATUS:
   Total chunks: 186
   With embeddings: 2
   Percentage: 1.1%
```

### **2. Full Detailed Report**
```bash
python rudy_test_embeddings.py
```
Shows comprehensive analysis including:
- Record counts by table
- Processing by date
- Embedding models used
- Recent documents processed
- Sample embedding data

### **3. Continuous Monitoring**
```bash
python rudy_test_embeddings.py --watch
```
Updates every 30 seconds. Press Ctrl+C to stop.

### **4. Custom Update Interval**
```bash
python rudy_test_embeddings.py --watch --interval 10
```
Updates every 10 seconds.

## 📊 **What The Report Shows**

### **Key Metrics:**
- **Total Chunks**: Documents broken into processable chunks
- **With Embeddings**: Chunks that have vector embeddings generated
- **Percentage**: How much of your data has embeddings ready
- **Date Range**: When documents were processed

### **Status Indicators:**
- `[OK]` - Normal status
- `[FAIL]` - Error condition  
- `[STATS]` - Data counts
- `[DATE]` - Date information
- `[EMBED]` - Embedding-specific data

## 🔧 **Troubleshooting**

### **"Database connection failed"**
- Check if PostgreSQL credentials are correct in `tidyllm/admin/settings.yaml`
- Verify network connection to AWS RDS
- Confirm VPN/security group access if needed

### **"psycopg2 not installed"**
```bash
pip install psycopg2-binary
```

### **Zero embeddings generated**
- Check if embedding generation process is running
- Look for errors in document processing pipeline
- Verify embedding model configuration

## 📈 **Interpreting Results**

### **Good Status:**
- High percentage of chunks with embeddings (>80%)
- Recent document processing activity
- Multiple embedding models working

### **Issues to Watch:**
- 0% embeddings generated = processing pipeline not working
- Very old dates = system not processing new documents
- Errors in table counts = database schema issues

## 🚀 **Integration with TidyLLM**

Use this script to:
1. **Monitor Pipeline Health** - Check if documents are being processed
2. **Debug Embedding Issues** - See which models are working
3. **Track Processing Progress** - Monitor batch jobs
4. **Validate Data Quality** - Ensure embeddings are being created

## 💡 **Pro Tips**

- Run `--summary` before starting embedding jobs
- Use `--watch` during batch processing to monitor progress
- Compare before/after reports when troubleshooting
- Check date ranges to identify when processing stopped working

---
**Created for TidyLLM embedding monitoring and debugging**