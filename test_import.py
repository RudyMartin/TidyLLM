#!/usr/bin/env python3
import sys
import traceback

print("Testing tidyllm import step by step...")

try:
    print("Step 1: Import logging")
    import logging
    logging.basicConfig(level=logging.DEBUG)
    print("[OK] Logging imported")
    
    print("Step 2: Import tidyllm.api directly")
    from tidyllm.api import chat
    print("[OK] tidyllm.api imported")
    
    print("Step 3: Test api function")
    result = chat("test")
    print(f"[OK] API function works: {result[:50]}...")
    
    print("Step 4: Import full tidyllm")
    import tidyllm
    print(f"[OK] tidyllm imported, available: {[x for x in dir(tidyllm) if not x.startswith('_')]}")
    
    if hasattr(tidyllm, 'chat'):
        print("[OK] tidyllm.chat is available")
    else:
        print("[ERROR] tidyllm.chat is NOT available")
        
except Exception as e:
    print(f"[ERROR] Error: {e}")
    traceback.print_exc()