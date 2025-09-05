# Database Layer

## Overview
The database layer manages data persistence, schemas, and data-related operations for VectorQA Sage.

## Structure
- `schemas/` - Database schema definitions
- `migrations/` - Database migration scripts
- `scripts/` - Database utility scripts

## Components

### Schemas (`schemas/`)
- Database schema definitions
- Data models and relationships
- Validation schemas

### Migrations (`migrations/`)
- Database migration scripts
- Schema evolution management
- Version control for database changes

### Scripts (`scripts/`)
- Database utility scripts
- Data seeding scripts
- Maintenance scripts

## Features
- Schema management for structured data organization
- Migration support for safe database evolution
- Data integrity through validation and constraint management

## Development
This layer is prepared for future database integration. Currently, the application uses file-based storage, but the structure is in place for database migration when needed.
