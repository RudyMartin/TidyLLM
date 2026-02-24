# AWS CLI Playbook — S3 → Lambda (Hexagonal) JSON→Text Processor

**Goal:** When a `.json` file lands in `s3://nsc-mvp1/incoming/`, a Lambda extracts text and writes `text/<name>.txt`, then moves the original JSON to `processed/<name>.json`.
**Region:** `us-east-1` (change if needed).
**Function name:** `hex-arch-s3-processor` (change if you prefer).
**Bucket:** `nsc-mvp1` (already created).

---

## 0) One-time prerequisites (work in **AWS CloudShell**)

```bash
# Confirm CLI identity and default region
aws sts get-caller-identity
aws configure list

# We'll use these variables throughout
export REGION=us-east-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export BUCKET_NAME=nsc-mvp1
export FUNCTION_NAME=hex-arch-s3-processor
```

---

## 1) Prepare S3 prefixes (“folders”)

S3 doesn’t have real folders; we create prefixes.

```bash
aws s3api put-object --bucket $BUCKET_NAME --key incoming/
aws s3api put-object --bucket $BUCKET_NAME --key text/
aws s3api put-object --bucket $BUCKET_NAME --key processed/
aws s3api put-object --bucket $BUCKET_NAME --key error/
```

---

## 2) Create (or fix) the IAM role Lambda will assume

### 2.1 Trust policy (lets Lambda assume the role)

```bash
# Create role if missing; otherwise update trust policy
if ! aws iam get-role --role-name LambdaS3ExecutionRole >/dev/null 2>&1; then
  aws iam create-role \
    --role-name LambdaS3ExecutionRole \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": { "Service": "lambda.amazonaws.com" },
        "Action": "sts:AssumeRole"
      }]
    }'
else
  aws iam update-assume-role-policy \
    --role-name LambdaS3ExecutionRole \
    --policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": { "Service": "lambda.amazonaws.com" },
        "Action": "sts:AssumeRole"
      }]
    }'
fi
```

### 2.2 Attach execution permissions

```bash
# CloudWatch logging
aws iam attach-role-policy \
  --role-name LambdaS3ExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# S3 access (scoped to your bucket)
aws iam put-role-policy \
  --role-name LambdaS3ExecutionRole \
  --policy-name LambdaS3BucketAccess \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [{
      \"Effect\": \"Allow\",
      \"Action\": [
        \"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\",\"s3:ListBucket\",\"s3:CopyObject\",\"s3:PutObjectAcl\"
      ],
      \"Resource\": [
        \"arn:aws:s3:::$BUCKET_NAME\",
        \"arn:aws:s3:::$BUCKET_NAME/*\"
      ]
    }]
  }"

# (Optional) If your bucket uses a customer-managed KMS key, add KMS perms:
# aws iam put-role-policy --role-name LambdaS3ExecutionRole --policy-name LambdaKMSAccess --policy-document '{
#   "Version": "2012-10-17",
#   "Statement": [{
#     "Effect": "Allow",
#     "Action": ["kms:Decrypt","kms:Encrypt","kms:GenerateDataKey","kms:DescribeKey"],
#     "Resource": "arn:aws:kms:us-east-1:<ACCOUNT_ID>:key/<YOUR_KEY_ID>"
#   }]
# }'
```

> Give IAM \~10 seconds to propagate before creating the function:

```bash
sleep 10
```

---

## 3) Build the Lambda code package (with fixed imports)

We’ll scaffold the package **in CloudShell**, add `__init__.py`, and use **absolute imports** (`from app.adapters...`).

```bash
rm -rf app hex_arch_s3_lambda.zip
mkdir -p app/domain app/adapters

# Make packages explicit
: > app/__init__.py
: > app/domain/__init__.py
: > app/adapters/__init__.py

# --- domain/ports.py
cat > app/domain/ports.py <<'PY'
from typing import Protocol

class StoragePort(Protocol):
    def read_object(self, key: str) -> bytes: ...
    def write_object(self, key: str, data: bytes) -> None: ...
    def move_object(self, src_key: str, dst_key: str) -> None: ...

class QueuePort(Protocol):
    def enqueue(self, payload: dict) -> None: ...

class LoggerPort(Protocol):
    def info(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...
PY

# --- domain/content_extractor.py
cat > app/domain/content_extractor.py <<'PY'
import json

def json_to_text(raw: bytes) -> str:
    obj = json.loads(raw.decode("utf-8"))
    parts = []
    def walk(x):
        if isinstance(x, dict):
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
        elif isinstance(x, str):
            parts.append(x.strip())
    walk(obj)
    return "\n".join(p for p in parts if p)
PY

# --- domain/use_cases.py
cat > app/domain/use_cases.py <<'PY'
from app.domain.ports import StoragePort, QueuePort, LoggerPort
from app.domain.content_extractor import json_to_text
from typing import Optional

def process_new_json(
    *, storage: StoragePort, queue: Optional[QueuePort], logger: LoggerPort,
    bucket: str, key: str, text_prefix: str, processed_prefix: str, error_prefix: str
) -> None:
    logger.info(f"Start processing s3://{bucket}/{key}")
    try:
        raw = storage.read_object(key)
        text = json_to_text(raw)
        base = key.split("/")[-1].removesuffix(".json")
        text_key = f\"{text_prefix}{base}.txt\"
        storage.write_object(text_key, text.encode(\"utf-8\"))
        processed_key = f\"{processed_prefix}{base}.json\"
        storage.move_object(key, processed_key)
        if queue:
            queue.enqueue({\"type\": \"document_processed\", \"json_key\": processed_key, \"text_key\": text_key})
        logger.info(f\"Done: text->{text_key}, moved->{processed_key}\")
    except Exception as e:
        logger.error(f\"Failed on {key}: {e}\")
        error_key = f\"{error_prefix}{key.split('/')[-1]}\"
        try:
            storage.move_object(key, error_key)
        except Exception:
            logger.error(\"Also failed to move to error/; leaving original in place.\")
        raise
PY

# --- adapters/s3_storage.py
cat > app/adapters/s3_storage.py <<'PY'
import boto3

class S3StorageAdapter:
    def __init__(self, bucket: str):
        self.s3 = boto3.client("s3")
        self.bucket = bucket

    def read_object(self, key: str) -> bytes:
        resp = self.s3.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()

    def write_object(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)

    def move_object(self, src_key: str, dst_key: str) -> None:
        self.s3.copy({"Bucket": self.bucket, "Key": src_key}, self.bucket, dst_key)
        self.s3.delete_object(Bucket=self.bucket, Key=src_key)
PY

# --- adapters/sqs_queue.py (optional)
cat > app/adapters/sqs_queue.py <<'PY'
import boto3, json

class SQSAdapter:
    def __init__(self, queue_url: str):
        self.sqs = boto3.client("sqs")
        self.queue_url = queue_url

    def enqueue(self, payload: dict) -> None:
        self.sqs.send_message(QueueUrl=self.queue_url, MessageBody=json.dumps(payload))
PY

# --- adapters/logger.py
cat > app/adapters/logger.py <<'PY'
class CWLoggerAdapter:
    def info(self, msg: str) -> None:
        print(f"[INFO] {msg}")

    def error(self, msg: str) -> None:
        print(f"[ERROR] {msg}")
PY

# --- lambda_handler.py (USE ABSOLUTE IMPORTS)
cat > app/lambda_handler.py <<'PY'
import os
from app.adapters.s3_storage import S3StorageAdapter
from app.adapters.sqs_queue import SQSAdapter
from app.adapters.logger import CWLoggerAdapter
from app.domain.use_cases import process_new_json

BUCKET = os.environ["BUCKET_NAME"]
TEXT_PREFIX = os.environ.get("TEXT_PREFIX", "text/")
PROCESSED_PREFIX = os.environ.get("PROCESSED_PREFIX", "processed/")
ERROR_PREFIX = os.environ.get("ERROR_PREFIX", "error/")
SQS_URL = os.environ.get("SQS_URL")

storage = S3StorageAdapter(bucket=BUCKET)
queue = SQSAdapter(SQS_URL) if SQS_URL else None
logger = CWLoggerAdapter()

def handler(event, context):
    for record in event.get("Records", []):
        s3 = record["s3"]
        bucket = s3["bucket"]["name"]
        key = s3["object"]["key"]
        if bucket != BUCKET or not key.endswith(".json"):
            logger.info(f"Skipping {bucket}/{key}")
            continue
        process_new_json(
            storage=storage, queue=queue, logger=logger,
            bucket=bucket, key=key,
            text_prefix=TEXT_PREFIX,
            processed_prefix=PROCESSED_PREFIX,
            error_prefix=ERROR_PREFIX
        )
    return {"status": "ok"}
PY

# Zip and verify
zip -r hex_arch_s3_lambda.zip app
unzip -l hex_arch_s3_lambda.zip | sed -n '1,50p'
```

---

## 4) Create the Lambda (or update if it already exists)

```bash
# Create
aws lambda create-function \
  --function-name "$FUNCTION_NAME" \
  --runtime python3.11 \
  --role arn:aws:iam::$ACCOUNT_ID:role/LambdaS3ExecutionRole \
  --handler app.lambda_handler.handler \
  --zip-file fileb://hex_arch_s3_lambda.zip \
  --region $REGION \
|| \
# Update (if it already exists)
aws lambda update-function-code \
  --function-name "$FUNCTION_NAME" \
  --zip-file fileb://hex_arch_s3_lambda.zip \
  --region $REGION
```

---

## 5) Set required environment variables + runtime settings

```bash
aws lambda update-function-configuration \
  --function-name "$FUNCTION_NAME" \
  --environment "Variables={BUCKET_NAME=$BUCKET_NAME,TEXT_PREFIX=text/,PROCESSED_PREFIX=processed/,ERROR_PREFIX=error/}" \
  --memory-size 256 \
  --timeout 30 \
  --region $REGION

# Verify
aws lambda get-function \
  --function-name "$FUNCTION_NAME" \
  --region "$REGION" \
  --query 'Configuration.[FunctionName,Handler,Runtime,Environment]'
```

---

## 6) Wire S3 → Lambda (invoke permission + bucket notification)

> ⚠️ **Note:** `put-bucket-notification-configuration` **overwrites** existing notifications on the bucket.

```bash
# Allow S3 to invoke Lambda (ignore error if statement-id already exists)
aws lambda add-permission \
  --function-name "$FUNCTION_NAME" \
  --principal s3.amazonaws.com \
  --statement-id s3invoke-allow \
  --action "lambda:InvokeFunction" \
  --source-arn arn:aws:s3:::$BUCKET_NAME \
  --region $REGION 2>/dev/null || true

# Attach bucket notification for incoming/*.json (Put events)
aws s3api put-bucket-notification-configuration \
  --bucket $BUCKET_NAME \
  --notification-configuration "{
    \"LambdaFunctionConfigurations\": [
      {
        \"LambdaFunctionArn\": \"arn:aws:lambda:$REGION:$ACCOUNT_ID:function:$FUNCTION_NAME\",
        \"Events\": [\"s3:ObjectCreated:Put\"],
        \"Filter\": {
          \"Key\": { \"FilterRules\": [
            { \"Name\": \"prefix\", \"Value\": \"incoming/\" },
            { \"Name\": \"suffix\", \"Value\": \".json\" }
          ]}
        }
      }
    ]
  }" \
  --region $REGION

# Verify
aws s3api get-bucket-notification-configuration --bucket $BUCKET_NAME --region $REGION
```

---

## 7) Test the flow

```bash
# Upload a sample JSON to trigger Lambda
echo '{"hello":"world","nested":{"foo":"bar"}}' > test.json
aws s3 cp test.json s3://$BUCKET_NAME/incoming/test.json --region $REGION

# First run creates the log group; then tail logs
aws logs tail /aws/lambda/$FUNCTION_NAME --since 5m --follow --region $REGION
# (Press Ctrl+C to stop following)

# Verify outputs
aws s3 ls s3://$BUCKET_NAME/text/ --region $REGION
aws s3 ls s3://$BUCKET_NAME/processed/ --region $REGION
aws s3 ls s3://$BUCKET_NAME/error/ --region $REGION
```

**Expected:**

* `text/test.txt` contains:

  ```
  world
  bar
  ```
* `processed/test.json` exists.

---

## 8) Common troubleshooting

* **ImportModuleError: No module named 'adapters'**
  → Ensure the ZIP has `app/` at root, **with** `__init__.py` files, and **absolute imports** like `from app.adapters...`.
  → Handler must be `app.lambda_handler.handler`.

* **Environment = null** in `get-function`:
  → You didn’t set env vars; run step 5.

* **S3 upload doesn’t trigger Lambda**:
  → Confirm bucket notification (step 6) matches `prefix=incoming/` and `suffix=.json`.
  → Ensure region matches everywhere (`us-east-1`).
  → Confirm Lambda resource policy includes S3 principal:

  ```bash
  aws lambda get-policy --function-name $FUNCTION_NAME --region $REGION
  ```

* **No logs**:
  → First invocation creates the log group. Upload a test file, then:

  ```bash
  aws logs tail /aws/lambda/$FUNCTION_NAME --since 5m --region $REGION
  ```

* **KMS-encrypted bucket**:
  → Add KMS permissions to the role (see step 2.2 optional block) and ensure the KMS key policy allows the role to use it.

---

## 9) Optional: Add downstream SQS notification

```bash
# Create queue
QUEUE_URL=$(aws sqs create-queue --queue-name doc-processed-queue --query QueueUrl --output text)

# Add env var to Lambda
aws lambda update-function-configuration \
  --function-name "$FUNCTION_NAME" \
  --environment "Variables={BUCKET_NAME=$BUCKET_NAME,TEXT_PREFIX=text/,PROCESSED_PREFIX=processed/,ERROR_PREFIX=error/,SQS_URL=$QUEUE_URL}" \
  --region $REGION

# Role permission to send to SQS
aws iam put-role-policy \
  --role-name LambdaS3ExecutionRole \
  --policy-name LambdaSQSSend \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [{
      \"Effect\": \"Allow\",
      \"Action\": [\"sqs:SendMessage\"],
      \"Resource\": \"arn:aws:sqs:$REGION:$ACCOUNT_ID:doc-processed-queue\"
    }]
  }"
```

---

## 10) Update cycle

```bash
# After editing code under app/:
zip -r hex_arch_s3_lambda.zip app

aws lambda update-function-code \
  --function-name "$FUNCTION_NAME" \
  --zip-file fileb://hex_arch_s3_lambda.zip \
  --region $REGION
```

---

**Hand this to the developer** as-is. It’s linear, idempotent where possible, and includes the exact fixes we needed:

* Correct trust policy and role policies
* Proper package structure and imports
* Environment variables set
* S3 notification + permission
* Clear test + troubleshooting steps
