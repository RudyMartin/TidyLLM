Awesome. Here’s a **clean, copy-paste Terraform stack** that provisions everything for the S3 → Lambda JSON→Text pipeline. It zips the local `app/` directory, creates the Lambda + IAM role/policies, wires S3 bucket notifications, and (optionally) creates the visible S3 “folders”.

> Assumes your bucket already exists: `nsc-mvp1` in `us-east-1`.
> Terraform will **look up** the bucket; it won’t try to recreate it.

---

# 🧱 Directory layout

```
your-project/
├─ app/                      # your Lambda code (same as we deployed)
│  ├─ __init__.py
│  ├─ lambda_handler.py
│  ├─ adapters/...
│  └─ domain/...
├─ main.tf
├─ variables.tf
└─ outputs.tf
```

If you don’t have `app/` locally, you can rebuild it quickly with the shell snippet you used earlier (or ask me and I’ll drop it again).

---

# main.tf

```hcl
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = ">= 2.4"
    }
  }
}

provider "aws" {
  region = var.region
}

# Who am I?
data "aws_caller_identity" "current" {}

# Use an existing bucket by name
data "aws_s3_bucket" "bucket" {
  bucket = var.bucket_name
}

# Package the Lambda code from ./app into a zip
data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = "${path.module}/app"
  output_path = "${path.module}/hex_arch_s3_lambda.zip"
}

# IAM role: trust policy for Lambda
resource "aws_iam_role" "lambda_role" {
  name = "${var.function_name}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Allow",
      Principal = { Service = "lambda.amazonaws.com" },
      Action   = "sts:AssumeRole"
    }]
  })
}

# Attach basic execution (CloudWatch logs)
resource "aws_iam_role_policy_attachment" "basic_logs" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# S3 access limited to the one bucket
resource "aws_iam_policy" "s3_access" {
  name        = "${var.function_name}-s3-access"
  description = "Read/Write/Move objects in ${var.bucket_name}"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Allow",
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:CopyObject"
      ],
      Resource = [
        "arn:aws:s3:::${var.bucket_name}",
        "arn:aws:s3:::${var.bucket_name}/*"
      ]
    }]
  })
}

resource "aws_iam_role_policy_attachment" "s3_access_attach" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = aws_iam_policy.s3_access.arn
}

# (Optional) add KMS perms if your bucket uses a CMK
# resource "aws_iam_policy" "kms_access" { ... }
# resource "aws_iam_role_policy_attachment" "kms_access_attach" { ... }

# Lambda function
resource "aws_lambda_function" "fn" {
  function_name = var.function_name
  role          = aws_iam_role.lambda_role.arn
  handler       = "app.lambda_handler.handler"
  runtime       = "python3.11"
  filename      = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256

  memory_size   = 256
  timeout       = 30

  environment {
    variables = {
      BUCKET_NAME      = var.bucket_name
      TEXT_PREFIX      = var.text_prefix
      PROCESSED_PREFIX = var.processed_prefix
      ERROR_PREFIX     = var.error_prefix
      # SQS_URL        = "https://sqs.${var.region}.amazonaws.com/${data.aws_caller_identity.current.account_id}/your-queue" # optional
    }
  }
}

# Allow S3 to invoke the Lambda
resource "aws_lambda_permission" "allow_s3" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.fn.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = data.aws_s3_bucket.bucket.arn
}

# S3 -> Lambda notifications for incoming/*.json (PUT)
resource "aws_s3_bucket_notification" "notify" {
  bucket = data.aws_s3_bucket.bucket.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.fn.arn
    events              = ["s3:ObjectCreated:Put"]
    filter_prefix       = var.trigger_prefix
    filter_suffix       = ".json"
  }

  depends_on = [aws_lambda_permission.allow_s3]
}

# (Optional) create visible "folders" via zero-byte objects
resource "aws_s3_object" "prefix_incoming" {
  count  = var.create_prefixes ? 1 : 0
  bucket = var.bucket_name
  key    = "${var.trigger_prefix}"
  content = ""
}

resource "aws_s3_object" "prefix_text" {
  count  = var.create_prefixes ? 1 : 0
  bucket = var.bucket_name
  key    = "${var.text_prefix}"
  content = ""
}

resource "aws_s3_object" "prefix_processed" {
  count  = var.create_prefixes ? 1 : 0
  bucket = var.bucket_name
  key    = "${var.processed_prefix}"
  content = ""
}

resource "aws_s3_object" "prefix_error" {
  count  = var.create_prefixes ? 1 : 0
  bucket = var.bucket_name
  key    = "${var.error_prefix}"
  content = ""
}
```

---

# variables.tf

```hcl
variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "bucket_name" {
  description = "Existing S3 bucket that will trigger Lambda"
  type        = string
  default     = "nsc-mvp1"
}

variable "function_name" {
  description = "Lambda function name"
  type        = string
  default     = "hex-arch-s3-processor"
}

variable "trigger_prefix" {
  description = "S3 prefix to watch for JSON files"
  type        = string
  default     = "incoming/"
}

variable "text_prefix" {
  description = "Prefix where extracted text files are written"
  type        = string
  default     = "text/"
}

variable "processed_prefix" {
  description = "Prefix where original JSONs are moved"
  type        = string
  default     = "processed/"
}

variable "error_prefix" {
  description = "Prefix for failures"
  type        = string
  default     = "error/"
}

variable "create_prefixes" {
  description = "Create zero-byte objects so prefixes show in the console"
  type        = bool
  default     = true
}
```

---

# outputs.tf

```hcl
output "function_name" {
  value = aws_lambda_function.fn.function_name
}

output "function_arn" {
  value = aws_lambda_function.fn.arn
}

output "role_arn" {
  value = aws_iam_role.lambda_role.arn
}

output "bucket_arn" {
  value = data.aws_s3_bucket.bucket.arn
}
```

---

# 📦 Lambda code reminder (goes in `app/`)

Ensure your `app/` code matches the working version (absolute imports + URL-decoding). If you need, here are the key files/paths:

* `app/lambda_handler.py` (imports from `app.adapters...`, URL-decodes S3 key with `unquote_plus`)
* `app/adapters/{s3_storage.py,sqs_queue.py,logger.py}`
* `app/domain/{ports.py,content_extractor.py,use_cases.py}`
* Add `__init__.py` in `app/`, `app/adapters/`, `app/domain/`

(If you want, I can re-print those files again to paste.)

---

# 🚀 Apply

```bash
terraform init
terraform plan
terraform apply
```

> ⚠️ Heads up: Terraform will **manage the bucket’s notification config**. If you previously added S3 → Lambda notifications via CLI/Console, Terraform will replace them with what’s in `aws_s3_bucket_notification.notify`.

---

# ✅ Test

```bash
echo '{"hello":"world","nested":{"foo":"bar"}}' > test.json
aws s3 cp test.json s3://nsc-mvp1/incoming/test.json --region us-east-1

# After first invoke creates the log group:
aws logs tail /aws/lambda/hex-arch-s3-processor --since 5m --follow --region us-east-1
# Ctrl+C to stop

aws s3 ls s3://nsc-mvp1/text/
aws s3 ls s3://nsc-mvp1/processed/
```

**Expected:** `text/test.txt` with:

```
world
bar
```

and `processed/test.json` exists.

---

# 🔧 Notes / Options

* **KMS-encrypted bucket?** Add a KMS policy + role attachment (commented in `main.tf`) and ensure your key policy allows the role ARN.
* **Downstream SQS?** Add `SQS_URL` to `environment.variables` and give the role `sqs:SendMessage` on that queue.
* **Code updates:** Re-run `terraform apply`; the `archive_file` hash detects changes and updates the Lambda.

If you want me to include the **exact `app/` code files** again here so your dev can paste them next to the Terraform, say the word and I’ll drop them inline.
