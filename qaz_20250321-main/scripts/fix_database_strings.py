#!/usr/bin/env python3
"""
Database String Fixer

Fixes database connection strings that were affected by the pre-flight filtering process.
Restores proper database connectivity while maintaining security.
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Any

class DatabaseStringFixer:
    """Fix database connection strings affected by filtering"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.fixed_files = []
        self.errors = []
        
        # Database connection patterns that need fixing
        self.db_patterns = [
            # Pattern: postgresql://user:password@localhost:5432/database
            (r'postgresql://user:password@localhost:5432/database', 
             'postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}'),
            
            # Pattern: postgresql://user:password@localhost:5432/vectorqa
            (r'postgresql://user:password@localhost:5432/vectorqa',
             'postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/vectorqa'),
        ]
    
    def fix_database_strings(self, dry_run: bool = True) -> bool:
        """Fix database connection strings in affected files"""
        print("🔧 DATABASE STRING FIXER")
        print("=" * 60)
        
        if dry_run:
            print("🔍 DRY RUN MODE - No files will be modified")
        else:
            print("🚀 FIXING MODE - Files will be modified")
        
        print()
        
        # Files that need database string fixes
        files_to_fix = [
            "notebooks/01_database_exploration.py",
            "notebooks/05_point_in_time_demo.py", 
            "notebooks/06_model_risk_governance_workflow.py",
            "notebooks/07_mcp_implementation_demo.py",
            "notebooks/08_mcp_backend_sequence_demo.py",
            "scripts/create_migration_bundle.py",
            "scripts/pre_flight_cleanup.py"
        ]
        
        total_fixes = 0
        
        for file_path in files_to_fix:
            full_path = self.project_root / file_path
            if full_path.exists():
                fixes = self._fix_file(full_path, dry_run)
                total_fixes += fixes
            else:
                print(f"❌ {file_path}: File not found")
                self.errors.append(f"File not found: {file_path}")
        
        # Create environment template
        self._create_environment_template()
        
        # Print summary
        self._print_summary(total_fixes, dry_run)
        
        return len(self.errors) == 0
    
    def _fix_file(self, file_path: Path, dry_run: bool) -> int:
        """Fix database strings in a single file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            original_content = content
            fixes_made = 0
            
            # Apply database string fixes
            for pattern, replacement in self.db_patterns:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    fixes_made += 1
            
            # Check if any changes were made
            if content != original_content:
                if dry_run:
                    print(f"🔍 {file_path.name}: Would fix {fixes_made} database strings")
                    self._show_changes(original_content, content)
                else:
                    with open(file_path, 'w') as f:
                        f.write(content)
                    print(f"✅ {file_path.name}: Fixed {fixes_made} database strings")
                    self.fixed_files.append(str(file_path))
            else:
                print(f"ℹ️  {file_path.name}: No database strings to fix")
            
            return fixes_made
            
        except Exception as e:
            print(f"❌ {file_path.name}: Error processing file - {e}")
            self.errors.append(f"Error processing {file_path}: {e}")
            return 0
    
    def _show_changes(self, original: str, modified: str):
        """Show what changes would be made (dry run)"""
        original_lines = original.split('\n')
        modified_lines = modified.split('\n')
        
        for i, (orig_line, mod_line) in enumerate(zip(original_lines, modified_lines), 1):
            if orig_line != mod_line and 'postgresql://' in orig_line:
                print(f"    Line {i}:")
                print(f"      - {orig_line.strip()}")
                print(f"      + {mod_line.strip()}")
    
    def _create_environment_template(self):
        """Create environment template for database configuration"""
        print("\n📝 Creating Environment Template...")
        
        template_content = """# Database Environment Configuration
# Copy this to .env.local and fill in your actual database credentials

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=vectorqa
DB_USER=your_username
DB_PASSWORD=your_password

# Full Database URL (auto-generated from above)
DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}

# AWS Configuration (if using RDS)
AWS_REGION=us-east-1
AWS_RDS_ENDPOINT=your-rds-endpoint.amazonaws.com

# Optional: Direct RDS connection
# DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@${AWS_RDS_ENDPOINT}:5432/${DB_NAME}
"""
        
        template_path = self.project_root / "environ_settings" / ".env.template"
        
        try:
            with open(template_path, 'w') as f:
                f.write(template_content)
            print(f"✅ Created environment template: {template_path}")
        except Exception as e:
            print(f"❌ Error creating template: {e}")
            self.errors.append(f"Error creating template: {e}")
    
    def _print_summary(self, total_fixes: int, dry_run: bool):
        """Print summary of fixes"""
        print("\n" + "=" * 60)
        print("📊 DATABASE STRING FIX SUMMARY")
        print("=" * 60)
        
        if dry_run:
            print(f"🔍 DRY RUN: Would fix {total_fixes} database strings")
            print("💡 Run with --fix to apply changes")
        else:
            print(f"✅ FIXED: {total_fixes} database strings")
            print(f"📁 Files modified: {len(self.fixed_files)}")
        
        if self.fixed_files:
            print("\n📝 Modified Files:")
            for file_path in self.fixed_files:
                print(f"  • {file_path}")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        
        print("\n💡 Next Steps:")
        print("  1. Copy environ_settings/.env.template to environ_settings/.env.local")
        print("  2. Fill in your actual database credentials")
        print("  3. Set DATABASE_URL environment variable")
        print("  4. Test database connectivity with: python3 scripts/check_database_connectivity.py")
    
    def create_database_setup_guide(self):
        """Create a database setup guide"""
        guide_content = """# Database Setup Guide

## 🔗 Database Configuration

The database connection strings have been updated to use environment variables for security.

### 1. Environment Configuration

Copy the template and configure your database:

```bash
# Copy template
cp environ_settings/.env.template environ_settings/.env.local

# Edit with your credentials
nano environ_settings/.env.local
```

### 2. Database Options

#### Option A: Local PostgreSQL
```bash
# Install PostgreSQL locally
brew install postgresql  # macOS
sudo apt-get install postgresql postgresql-contrib  # Ubuntu

# Start PostgreSQL
brew services start postgresql  # macOS
sudo systemctl start postgresql  # Ubuntu

# Create database
createdb vectorqa
```

#### Option B: Docker PostgreSQL
```bash
# Run PostgreSQL in Docker
docker run --name vectorqa-postgres \\
  -e POSTGRES_DB=vectorqa \\
  -e POSTGRES_USER=vectorqa \\
  -e POSTGRES_PASSWORD=your_password \\
  -p 5432:5432 \\
  -d postgres:15
```

#### Option C: AWS RDS
```bash
# Use AWS RDS PostgreSQL instance
# Update .env.local with your RDS endpoint
DB_HOST=your-rds-endpoint.amazonaws.com
```

### 3. Test Connection

```bash
# Test database connectivity
python3 scripts/check_database_connectivity.py

# Run database exploration
python3 notebooks/01_database_exploration.py
```

### 4. Environment Variables

The following environment variables are used:

- `DB_HOST`: Database host (default: localhost)
- `DB_PORT`: Database port (default: 5432)
- `DB_NAME`: Database name (default: vectorqa)
- `DB_USER`: Database username
- `DB_PASSWORD`: Database password
- `DATABASE_URL`: Full connection string (auto-generated)

### 5. Security Notes

- Never commit `.env.local` to version control
- Use strong passwords for database users
- Consider using AWS Secrets Manager for production
- Enable SSL for remote database connections

## 🚀 Quick Start

1. **Setup Database:**
   ```bash
   # Local PostgreSQL
   createdb vectorqa
   ```

2. **Configure Environment:**
   ```bash
   cp environ_settings/.env.template environ_settings/.env.local
   # Edit .env.local with your credentials
   ```

3. **Test Connection:**
   ```bash
   python3 scripts/check_database_connectivity.py
   ```

4. **Run Application:**
   ```bash
   python3 run_qa_demo.py
   ```
"""
        
        guide_path = self.project_root / "DATABASE_SETUP_GUIDE.md"
        
        try:
            with open(guide_path, 'w') as f:
                f.write(guide_content)
            print(f"✅ Created database setup guide: {guide_path}")
        except Exception as e:
            print(f"❌ Error creating guide: {e}")
            self.errors.append(f"Error creating guide: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database String Fixer")
    parser.add_argument("--fix", action="store_true", help="Actually fix files (default: dry run)")
    parser.add_argument("--create-guide", action="store_true", help="Create database setup guide")
    
    args = parser.parse_args()
    
    fixer = DatabaseStringFixer()
    
    # Fix database strings
    success = fixer.fix_database_strings(dry_run=not args.fix)
    
    # Create setup guide if requested
    if args.create_guide:
        fixer.create_database_setup_guide()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
