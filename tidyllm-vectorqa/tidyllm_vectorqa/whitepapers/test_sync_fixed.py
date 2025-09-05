#!/usr/bin/env python3
"""Test S3 sync with fixed bucket handling"""

from dotenv import load_dotenv
load_dotenv()

from paper_repository import get_paper_repository

repo = get_paper_repository()
print('Testing S3 sync with nsc-mvp1...')
print('=' * 50)

result = repo.sync_to_s3('nsc-mvp1', 'papers/')

print(f"Success: {result.get('success', False)}")
print(f"Message: {result.get('message', 'No message')}")

if result.get('errors'):
    print(f"\nErrors ({len(result['errors'])}):")
    for error in result['errors'][:3]:  # Show first 3 errors
        print(f"  - {error[:100]}...")
else:
    print(f"Uploaded: {result.get('uploaded_count', 0)} papers")
    print(f"Total size: {result.get('total_size_mb', 0):.2f} MB")