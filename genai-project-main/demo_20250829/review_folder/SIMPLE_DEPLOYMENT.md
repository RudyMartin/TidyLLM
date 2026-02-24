# Simple Air-Gapped Deployment Instructions

## Quick Start (3 Steps)

### Step 1: Update config.yaml
Edit `config.yaml` with your actual credentials:
```yaml
postgres:
  host: your-actual-rds-endpoint.amazonaws.com
  port: 5432
  database: your_database
  username: your_username
  password: your_password
```

### Step 2: Upload Files
Upload these files to your air-gapped environment:
- `config.yaml` (with your credentials)
- `simple_postgres_connection.py`
- `requirements_simple.txt`

### Step 3: Run
```bash
# Install dependencies (one time)
pip install psycopg2-binary pyyaml

# Test connection
python simple_postgres_connection.py

# Run a query
python simple_postgres_connection.py "SELECT * FROM your_table LIMIT 5"
```

## That's it! ✅

---

## Full File List for Complete Setup

If you need more features, upload these files:

### Core Files (Required)
```
config.yaml                      # Your configuration
simple_postgres_connection.py    # Basic connection script
requirements_simple.txt          # Python dependencies
```

### Advanced Features (Optional)
```
postgres_airgapped_manager.py    # Advanced connection manager
aws_airgapped_manager.py         # AWS services support
postgres_test.py                 # Comprehensive tests
```

## Common Commands

```bash
# Test connection
python simple_postgres_connection.py

# Run SELECT query
python simple_postgres_connection.py "SELECT count(*) FROM users"

# Run any SQL
python simple_postgres_connection.py "CREATE TABLE test (id INT)"
```

## Troubleshooting

**Connection refused**
- Check your RDS security group allows connections from your IP
- Verify the endpoint is correct in config.yaml

**Authentication failed**
- Double-check username/password in config.yaml
- Ensure user has connect permissions on the database

**SSL error**
- Try changing `ssl_mode` in config.yaml to `disable` for testing
- For production, keep it as `require`

## Security Notes

⚠️ **Important**: 
- Never commit config.yaml with real credentials to git
- Delete config.yaml after deployment if possible
- Use environment variables for production:

```python
# Alternative: Use environment variables instead of config.yaml
import os

connection = psycopg2.connect(
    host=os.environ['DB_HOST'],
    database=os.environ['DB_NAME'],
    user=os.environ['DB_USER'],
    password=os.environ['DB_PASSWORD']
)