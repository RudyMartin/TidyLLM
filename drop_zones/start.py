#!/usr/bin/env python3
"""
Drop Zones Launcher - Unified Implementation
============================================

Simple launcher that uses the official unified drop zones implementation.
All operations go through UnifiedSessionManager architecture.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Launch unified drop zones system"""
    
    print("=" * 60)
    print("TIDYLLM DROP ZONES - UNIFIED IMPLEMENTATION")
    print("Using UnifiedSessionManager (Official Architecture)")
    print("=" * 60)
    
    try:
        # Import and launch unified drop zones
        from scripts.unified_drop_zones import main as unified_main
        unified_main()
        
    except ImportError as e:
        print(f"❌ Could not import unified drop zones: {e}")
        print("\nEnsure you're running from the project root directory:")
        print("  python drop_zones/start.py")
        print("\nOr use the direct script:")
        print("  python scripts/unified_drop_zones.py")
        return 1
    
    except KeyboardInterrupt:
        print("\n🛑 Drop zones monitoring stopped by user")
        return 0
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())