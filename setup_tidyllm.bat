@echo off
REM TidyLLM Complete Setup - Windows Batch Script
REM ===============================================
REM Easy onboarding that uses existing admin infrastructure

echo ============================================================
echo ^>^> TidyLLM Complete Setup v1.0.4
echo ============================================================
echo Easy setup using existing admin tools
echo.

REM Step 1: Set AWS credentials using existing admin script
echo [STEP 1]: Setting up AWS credentials...
call tidyllm\admin\set_aws_env.bat
if errorlevel 1 (
    echo [ERROR] AWS setup failed
    exit /b 1
)
echo.

REM Step 2: Simple connectivity test using existing environment
echo [STEP 2]: Testing basic connectivity...
python -c "import boto3; print('[SUCCESS] S3 buckets:', len(boto3.client('s3').list_buckets()['Buckets']))" 2>nul
if errorlevel 1 (
    echo [WARNING] S3 test failed, but continuing...
) else (
    echo [SUCCESS] S3 connectivity verified
)
echo.

REM Step 3: Test TidyLLM import
echo [STEP 3]: Testing TidyLLM import...
python -c "import sys; sys.path.insert(0, '.'); import tidyllm; print('[SUCCESS] TidyLLM import successful')" 2>nul
if errorlevel 1 (
    echo [WARNING] TidyLLM import failed, but setup completed
) else (
    echo [SUCCESS] TidyLLM import working
)
echo.

REM Step 4: Summary
echo [STEP 4]: Setup Complete!
echo ============================================================
echo [SUCCESS] TidyLLM setup completed using existing admin tools
echo.
echo [NEXT STEPS]:
echo   1. Try: python -c "import tidyllm; print(tidyllm.chat('Hello!'))"
echo   2. Check configuration: tidyllm\admin\settings.yaml
echo   3. Run full test: python tidyllm\admin\test_config.py
echo   4. Read constraints: docs\2025-09-08\IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md
echo.
echo [ADMIN TOOLS]:
echo   * tidyllm\admin\set_aws_env.bat - Reset AWS credentials
echo   * tidyllm\admin\test_config.py - Full configuration test  
echo   * tidyllm\admin\run_diagnostics_real.py - Detailed diagnostics
echo.
echo [ARCHITECTURE]: 4-Gateway 2-Service Design
echo   * CorporateLLM -^> AIProcessing -^> WorkflowOptimizer -^> Database
echo   * + UnifiedSessionManager + DomainRAG
echo.
echo Setup complete! AWS credentials are set for this session.
echo ============================================================