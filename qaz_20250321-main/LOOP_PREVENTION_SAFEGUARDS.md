# 🔄 LOOP PREVENTION SAFEGUARDS (n=5 MAX RETRIES)

**Critical Safety Measure**: Prevent infinite loops during fix execution by limiting retry attempts to **n=5** maximum per step.

---

## 🚨 **LOOP PREVENTION RULES**

### **Rule 1: Maximum Retry Limit**
- **n=5**: Maximum attempts for any single fix step
- **After 5 failures**: STOP and escalate to manual intervention
- **No infinite loops**: Each step must have exit conditions

### **Rule 2: Progress Tracking**
- **Count attempts**: Track how many times each step has been tried
- **Document failures**: Record what failed and why
- **Exit on n=5**: Hard stop after 5 unsuccessful attempts

### **Rule 3: Escalation Protocol**
- **Attempt 1-3**: Automated retry with logging
- **Attempt 4-5**: Manual intervention recommended
- **After n=5**: STOP, document issue, seek help

---

## 🔄 **RETRY-SAFE TODO TEMPLATES**

### **Safe Import Test Template**
```bash
#!/bin/bash
# Template with n=5 loop prevention

test_import() {
    local module="$1"
    local max_attempts=5
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt/$max_attempts: Testing import of $module"
        
        if python -c "import sys; sys.path.append('src'); from $module import *; print('✅ Import successful')"; then
            echo "✅ SUCCESS: $module imported successfully on attempt $attempt"
            return 0
        else
            echo "❌ FAILED: $module import failed on attempt $attempt"
            attempt=$((attempt + 1))
            
            if [ $attempt -gt $max_attempts ]; then
                echo "🚨 CRITICAL: $module failed after $max_attempts attempts - STOPPING"
                echo "Manual intervention required for: $module"
                return 1
            fi
            
            echo "Waiting 2 seconds before retry..."
            sleep 2
        fi
    done
}

# Usage:
# test_import "backend.mcp.orchestrators.enhanced_qa_orchestrator"
```

### **Safe File Operation Template**
```bash
#!/bin/bash
# File operations with retry limit

safe_file_operation() {
    local operation="$1"
    local file="$2" 
    local max_attempts=5
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt/$max_attempts: $operation on $file"
        
        case $operation in
            "create")
                if touch "$file" 2>/dev/null; then
                    echo "✅ SUCCESS: Created $file on attempt $attempt"
                    return 0
                fi
                ;;
            "delete")
                if rm "$file" 2>/dev/null; then
                    echo "✅ SUCCESS: Deleted $file on attempt $attempt" 
                    return 0
                fi
                ;;
            "modify")
                # Add your modification logic here
                if [ -w "$file" ]; then
                    echo "✅ SUCCESS: Modified $file on attempt $attempt"
                    return 0
                fi
                ;;
        esac
        
        echo "❌ FAILED: $operation on $file failed on attempt $attempt"
        attempt=$((attempt + 1))
        
        if [ $attempt -gt $max_attempts ]; then
            echo "🚨 CRITICAL: $operation on $file failed after $max_attempts attempts - STOPPING"
            return 1
        fi
        
        sleep 1
    done
}
```

---

## 📋 **RETRY-LIMITED TODO STEPS**

### **Phase 1A: Setup with Loop Prevention**
```bash
# STEP 1A.2: Create datamart directory (max 5 attempts)
create_datamart_dir() {
    local max_attempts=5
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt/$max_attempts: Creating datamart directory"
        
        if mkdir -p src/backend/datamart && touch src/backend/datamart/__init__.py; then
            echo "✅ SUCCESS: Datamart directory created on attempt $attempt"
            return 0
        else
            echo "❌ FAILED: Directory creation failed on attempt $attempt"
            attempt=$((attempt + 1))
            
            if [ $attempt -gt $max_attempts ]; then
                echo "🚨 STOP: Directory creation failed after $max_attempts attempts"
                echo "Check permissions and disk space manually"
                return 1
            fi
            
            sleep 1
        fi
    done
}
```

### **Phase 1B: DataMart Extraction with Retry Limit**
```python
# STEP 1B.3: Test new manager with retry logic
def test_datamart_with_retries():
    """Test DataMart with n=5 retry limit"""
    max_attempts = 5
    
    for attempt in range(1, max_attempts + 1):
        print(f"Attempt {attempt}/{max_attempts}: Testing DataMart import")
        
        try:
            from src.backend.datamart.core_manager import DataMartManager, DataMartMode
            dm = DataMartManager(DataMartMode.SIMPLE)
            
            if dm.initialize_datamart():
                print(f"✅ SUCCESS: DataMart works on attempt {attempt}")
                return True
            else:
                print(f"❌ FAILED: DataMart initialization failed on attempt {attempt}")
                
        except Exception as e:
            print(f"❌ FAILED: DataMart import failed on attempt {attempt}: {e}")
        
        if attempt == max_attempts:
            print(f"🚨 STOP: DataMart failed after {max_attempts} attempts")
            print("Manual intervention required - check file paths and syntax")
            return False
        
        print("Waiting 2 seconds before retry...")
        time.sleep(2)
    
    return False

# Usage: 
# if not test_datamart_with_retries():
#     exit(1)  # Stop execution if DataMart doesn't work after 5 attempts
```

### **Phase 1E: Import Updates with Retry Safety**
```python
# STEP 1E.1: Update enhanced_datamart_manager.py with retry logic
def update_imports_with_retries(file_path, old_import, new_import):
    """Update import statements with retry protection"""
    max_attempts = 5
    
    for attempt in range(1, max_attempts + 1):
        print(f"Attempt {attempt}/{max_attempts}: Updating imports in {file_path}")
        
        try:
            # Read file
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Replace import
            if old_import in content:
                new_content = content.replace(old_import, new_import)
                
                # Write file
                with open(file_path, 'w') as f:
                    f.write(new_content)
                
                # Test import works
                import sys
                sys.path.append('src')
                exec(f"from {file_path.replace('/', '.').replace('.py', '')} import *")
                
                print(f"✅ SUCCESS: Import updated in {file_path} on attempt {attempt}")
                return True
            else:
                print(f"❌ FAILED: Old import not found in {file_path} on attempt {attempt}")
                
        except Exception as e:
            print(f"❌ FAILED: Import update failed on attempt {attempt}: {e}")
            
            # Restore original file if it exists
            if attempt < max_attempts:
                try:
                    # Restore from git if possible
                    os.system(f"git checkout HEAD -- {file_path}")
                    print("Restored original file for retry")
                except:
                    pass
        
        if attempt == max_attempts:
            print(f"🚨 STOP: Import update failed after {max_attempts} attempts")
            print("Manual intervention required - check file syntax and paths")
            return False
            
        print("Waiting 2 seconds before retry...")
        time.sleep(2)
    
    return False
```

---

## 🚨 **CRITICAL FAILURE DETECTION**

### **Early Exit Conditions (Prevent Infinite Loops)**
```python
def check_critical_failure_conditions():
    """Check if we should exit early to prevent loops"""
    
    failure_conditions = [
        {
            'name': 'Git repository corrupted',
            'check': lambda: os.system('git status') != 0,
            'action': 'Stop immediately - git repo needs manual repair'
        },
        {
            'name': 'Python path issues', 
            'check': lambda: not os.path.exists('src'),
            'action': 'Stop immediately - src directory missing'
        },
        {
            'name': 'Permission issues',
            'check': lambda: not os.access('src', os.W_OK),
            'action': 'Stop immediately - no write permissions'
        },
        {
            'name': 'Disk space critical',
            'check': lambda: shutil.disk_usage('.').free < 100 * 1024 * 1024,  # 100MB
            'action': 'Stop immediately - insufficient disk space'
        }
    ]
    
    for condition in failure_conditions:
        if condition['check']():
            print(f"🚨 CRITICAL FAILURE: {condition['name']}")
            print(f"ACTION REQUIRED: {condition['action']}")
            return True
    
    return False

# Check before each major phase
if check_critical_failure_conditions():
    print("🚨 STOPPING EXECUTION - Critical failure detected")
    exit(1)
```

---

## 🔄 **MODIFIED EXECUTION PHASES WITH n=5 LIMITS**

### **Phase 1: Foundation (with Retry Limits)**
- [ ] **1A.1**: Create git branch (max 5 attempts) 
- [ ] **1A.2**: Create datamart directory (max 5 attempts)
- [ ] **1A.3**: Test package import (max 5 attempts)
- [ ] **1B.1**: Extract DataMartManager (max 5 attempts) 
- [ ] **1B.2**: Test extracted manager (max 5 attempts)
- [ ] **1C.1**: Update Advanced QA (max 5 attempts)
- [ ] **1C.2**: Test Advanced QA works (max 5 attempts)

**PHASE 1 EXIT CONDITIONS:**
- ✅ **Success**: All steps pass within retry limits → Proceed to Phase 2
- ❌ **Failure**: Any step fails after n=5 attempts → STOP, manual intervention required

### **Phase 2: Enhanced QA (with Retry Limits)**
- [ ] **2A.1**: Test Enhanced QA import (max 5 attempts)
- [ ] **2B.1**: Create coordinators if needed (max 5 attempts)
- [ ] **2B.2**: Test Enhanced QA functionality (max 5 attempts)

**PHASE 2 EXIT CONDITIONS:**
- ✅ **Success**: Enhanced QA works → Proceed to Phase 3
- ❌ **Failure**: Enhanced QA fails after n=5 attempts → STOP

### **Phase 3: Demo File (with Retry Limits)**
- [ ] **3A.1**: Create demo file (max 5 attempts)
- [ ] **3A.2**: Test demo import (max 5 attempts) 
- [ ] **3A.3**: Test launcher finds demo (max 5 attempts)

**PHASE 3 EXIT CONDITIONS:**
- ✅ **Success**: Demo works → Proceed to Phase 4
- ❌ **Failure**: Demo fails after n=5 attempts → STOP

### **Phase 4: Consolidation (with Retry Limits)**
- [ ] **4A.1**: Consolidate architecture (max 5 attempts)
- [ ] **4A.2**: Test consolidated system (max 5 attempts)

**PHASE 4 EXIT CONDITIONS:**
- ✅ **Success**: All tests pass → COMPLETE
- ❌ **Failure**: Consolidation fails after n=5 attempts → STOP (but Phases 1-3 still work)

---

## 📊 **LOOP PREVENTION TRACKING**

### **Attempt Counter Template**
```bash
# Create attempt tracking file
echo "# Attempt Tracking Log" > attempt_log.txt
echo "Started: $(date)" >> attempt_log.txt

log_attempt() {
    local step="$1"
    local attempt="$2" 
    local status="$3"
    local message="$4"
    
    echo "[$step] Attempt $attempt: $status - $message" >> attempt_log.txt
    
    if [ $attempt -eq 5 ] && [ "$status" = "FAILED" ]; then
        echo "[$step] CRITICAL: Maximum attempts reached - STOPPING" >> attempt_log.txt
        echo "🚨 CRITICAL: $step failed after 5 attempts"
        echo "Check attempt_log.txt for details"
        return 1
    fi
    
    return 0
}

# Usage:
# log_attempt "Phase1-DataMart" 3 "FAILED" "Import error in core_manager.py"
```

---

## 🎯 **SUCCESS WITH SAFEGUARDS**

### **Final Validation (n=5 Protected)**
```bash
#!/bin/bash
# Final system test with loop protection

final_validation() {
    local max_attempts=5
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "Final validation attempt $attempt/$max_attempts"
        
        if bash tests/run_all_fix_tests.sh; then
            echo "🎉 SUCCESS: All fixes complete and tested on attempt $attempt"
            echo "System is production ready!"
            return 0
        else
            echo "❌ FAILED: Final validation failed on attempt $attempt"
            attempt=$((attempt + 1))
            
            if [ $attempt -gt $max_attempts ]; then
                echo "🚨 CRITICAL: Final validation failed after $max_attempts attempts"
                echo "System may be partially fixed but not fully validated"
                echo "Manual review required"
                return 1
            fi
            
            echo "Waiting 5 seconds before final retry..."
            sleep 5
        fi
    done
}

# Run final validation
final_validation
```

---

**🛡️ KEY SAFETY BENEFIT**: With n=5 retry limits, the fix process will **never get stuck in infinite loops**, will **fail fast** on persistent issues, and will **clearly indicate** when manual intervention is needed.