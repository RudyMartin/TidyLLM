# Data Extraction Workflow Template
*Structured data extraction and processing workflow*

## Document Data Structure Analysis
```
Document Data Structure Identification:

Analyze document structure and identify extractable data elements:

Source Document: {document_content}

Structure Analysis:
1. **Data Format Identification**:
   - Structured vs. unstructured content
   - Tables, forms, and data grids
   - Key-value pairs and field mappings
   - Hierarchical data relationships

2. **Data Element Catalog**:
   - Identifiable data fields and types
   - Required vs. optional elements
   - Data validation rules and formats
   - Cross-reference and lookup values

3. **Extraction Complexity Assessment**:
   - Simple direct extraction fields
   - Complex pattern matching requirements
   - Calculated or derived values
   - Multi-document data consolidation

4. **Quality and Consistency Check**:
   - Data completeness assessment
   - Format consistency evaluation
   - Error and anomaly identification
   - Validation rule compliance

Output structured data inventory with extraction complexity ratings.
```

## Structured Data Extraction
```
Structured Data Extraction Process:

Extract specific data elements in structured format for downstream processing:

Document for Extraction: {document_content}

Data Extraction:
1. **Primary Data Elements**:
   - Names, addresses, contact information
   - Dates, times, and durations
   - Numbers, amounts, and quantities
   - Identifiers, codes, and references

2. **Secondary Data Elements**:
   - Relationships and associations
   - Status and classification codes
   - Comments and descriptive text
   - Metadata and document properties

3. **Calculated Fields**:
   - Derived values and computations
   - Aggregations and summaries
   - Ratios and percentages
   - Trend and change calculations

4. **Validation and Quality Control**:
   - Data type validation
   - Range and logic checks
   - Cross-field consistency validation
   - Error flagging and exception handling

Output as structured JSON/XML with validated data elements and quality indicators.
```

## Data Transformation and Standardization
```
Data Transformation and Standardization Process:

Transform extracted data into standardized formats for system integration:

Extracted Data: {document_content}

Data Transformation:
1. **Format Standardization**:
   - Date/time format normalization
   - Address and name standardization
   - Currency and numeric formatting
   - Code and identifier mapping

2. **Data Enrichment**:
   - Lookup and reference data addition
   - Validation against external sources
   - Missing data inference and completion
   - Data quality scoring and flagging

3. **Business Rule Application**:
   - Business logic validation
   - Workflow routing decisions
   - Exception handling rules
   - Compliance and regulatory checks

4. **Integration Readiness**:
   - Target system format preparation
   - Data mapping and transformation
   - Error handling and rollback procedures
   - Audit trail and logging requirements

Format as integration-ready dataset with transformation documentation.
```

## Data Quality and Validation Report
```
Data Quality Assessment and Validation Report:

Comprehensive data quality analysis and validation results:

Data Analysis Subject: {document_content}

Quality Assessment:
1. **Data Completeness**:
   - Required field population rates
   - Optional field completion statistics
   - Missing data pattern analysis
   - Data availability scoring

2. **Data Accuracy**:
   - Format validation results
   - Business rule compliance
   - Cross-reference validation
   - External source verification

3. **Data Consistency**:
   - Internal consistency checks
   - Cross-document consistency
   - Historical pattern validation
   - Anomaly and outlier detection

4. **Data Usability**:
   - Fitness for business purpose
   - Integration readiness assessment
   - Processing requirements satisfaction
   - User accessibility and format

Present as data quality scorecard with recommendations for data improvement and handling.
```