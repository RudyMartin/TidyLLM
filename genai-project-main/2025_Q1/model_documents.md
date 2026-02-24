### **Optimized `model_documents` Table Creation Code**
This version includes:
- **Partitioning by `model_date` (yearly)**
- **Default `model_status` = 'active'`**
- **Indexing for fast queries**
- **pgvector support for embeddings (512 dimensions)
    supports amazon titan text v3 dim options 1024/512/256**

```sql
CREATE TABLE model_documents (
    id SERIAL PRIMARY KEY,
    model_id UUID NOT NULL,
    model_doc VARCHAR(50) NOT NULL CHECK (
        model_doc IN ('model_latex', 'model_assumption', 'model_issue', 
                      'model_usage', 'model_usage_adj', 'model_version', 'model_review')
    ),
    model_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_status VARCHAR(10) NOT NULL DEFAULT 'active' CHECK (model_status IN ('active', 'inactive')),
    embedding VECTOR(512) NOT NULL
) PARTITION BY RANGE (model_date);
```

---

### **Creating Yearly Partitions**
```sql
CREATE TABLE model_documents_2024 PARTITION OF model_documents
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE model_documents_2025 PARTITION OF model_documents
FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
```
✅ **New records automatically go to the correct yearly partition**  
✅ **`model_status` defaults to `'active'`, reducing manual inserts**  



This automatically sets:
- `model_status = 'active'`
- Routes data to `model_documents_2024`  

---

### **Indexing for Performance**
To optimize queries filtering by **model_id, model_date, and model_doc**:
```sql
CREATE INDEX idx_model_id_date_doc ON model_documents (model_id, model_date, model_doc);
CREATE INDEX idx_model_id ON model_documents (model_id);
CREATE INDEX idx_embedding ON model_documents USING hnsw (embedding vector_l2_ops);
```
### **Optimized `model_documents` Table Creation Code**
This version includes:
- **Partitioning by `model_date` (yearly)**
- **Default `model_status` = 'active'`**
- **Indexing for fast queries**
- **pgvector support for embeddings (512 dimensions)**

```sql
CREATE TABLE model_documents (
    id SERIAL PRIMARY KEY,
    model_id UUID NOT NULL,
    model_doc VARCHAR(50) NOT NULL CHECK (
        model_doc IN ('model_latex', 'model_assumption', 'model_issue', 
                      'model_usage', 'model_usage_adj', 'model_version', 'model_review')
    ),
    model_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_status VARCHAR(10) NOT NULL DEFAULT 'active' CHECK (model_status IN ('active', 'inactive')),
    embedding VECTOR(512) NOT NULL
) PARTITION BY RANGE (model_date);
```

---

### **Creating Yearly Partitions TRY THE  AUTOPARTITION OPTION AT BOTTOM FIRST**
```sql
CREATE TABLE model_documents_2024 PARTITION OF model_documents
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE model_documents_2025 PARTITION OF model_documents
FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
```
✅ **New records automatically go to the correct yearly partition**  
✅ **`model_status` defaults to `'active'`, reducing manual inserts**  

---

### **Insert Example (Now Without Explicit `model_status`)**
Since `model_status` defaults to `'active'`, you **don't need to specify it** in the `INSERT` statement:
```sql
INSERT INTO model_documents (model_id, model_doc, model_date, embedding)
VALUES (
    'model_301', 
    'model_assumption', 
    '2024-06-15', 
    '[0.1, -0.5, ..., 0.3]'::vector
);
```

This automatically sets:
- `model_status = 'active'`
- Routes data to `model_documents_2024`  

---

### **Indexing for Performance**
To optimize queries filtering by **model_id, model_date, and model_doc**:
```sql
CREATE INDEX idx_model_id_date_doc ON model_documents (model_id, model_date, model_doc);
CREATE INDEX idx_model_id ON model_documents (model_id);
CREATE INDEX idx_embedding ON model_documents USING hnsw (embedding vector_l2_ops);
```

Would you like a **trigger** to **auto-create yearly partitions** instead of manually adding them? 

Here's a **PostgreSQL trigger function and trigger** that automatically creates **yearly partitions** when new data is inserted into `model_documents`. This ensures **seamless inserts** without requiring manual partition creation.

---

### **1. Create the Function for Dynamic Partitioning**
```sql
CREATE OR REPLACE FUNCTION create_partition_if_not_exists()
RETURNS TRIGGER AS $$
DECLARE
    partition_name TEXT;
    partition_start DATE;
    partition_end DATE;
BEGIN
    -- Determine the partition range based on model_date
    partition_start := date_trunc('year', NEW.model_date);
    partition_end := partition_start + INTERVAL '1 year';
    partition_name := 'model_documents_' || to_char(partition_start, 'YYYY');

    -- Check if the partition already exists
    IF NOT EXISTS (
        SELECT 1 FROM pg_tables WHERE tablename = partition_name
    ) THEN
        -- Dynamically create the partition
        EXECUTE format(
            'CREATE TABLE %I PARTITION OF model_documents
            FOR VALUES FROM (%L) TO (%L)',
            partition_name, partition_start, partition_end
        );
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

---

### **2. Create the Trigger to Call the Function**
```sql
CREATE TRIGGER trigger_create_partition
BEFORE INSERT ON model_documents
FOR EACH ROW EXECUTE FUNCTION create_partition_if_not_exists();
```

---

### **How This Works**
- Before an **insert** happens, the trigger **checks if the corresponding yearly partition exists**.
- If **it doesn’t exist**, it **creates it dynamically**.
- The **new record is then inserted normally** into the newly created or existing partition.

---

### **Now, You Can Insert Data Without Worrying About Partitions!**
```sql
INSERT INTO model_documents (model_id, model_doc, model_date, embedding)
VALUES (
    'model_302', 
    'model_usage', 
    '2026-02-10', 
    '[0.2, -0.4, ..., 0.6]'::vector
);
```
✅ If `model_documents_2026` **doesn't exist**, it will be **automatically created**.  
✅ The record **inserts without error** into the correct partition.


