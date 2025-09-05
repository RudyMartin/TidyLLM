# Section View Workflow Prompts

## Document Structure Analysis
```
Section Identification Prompt:

Analyze the document and break it down into logical sections for interactive browsing:

Document: {document_content}

Identify and extract:

1. **Header Information**
   - Document title
   - Reference numbers (REV ID, case numbers)
   - Dates and timestamps
   - Document type and version

2. **Main Sections**
   - Primary content areas
   - Data tables or structured information
   - Key findings or observations
   - Recommendations or conclusions

3. **Supporting Information**  
   - Appendices
   - References
   - Metadata
   - Footnotes or annotations

4. **Navigation Structure**
   - Create a table of contents
   - Section hierarchies
   - Cross-references between sections

Format as structured JSON for interactive navigation.
```

## Interactive Browse Preparation
```
Content Summarization Prompt:

For each section identified, create concise summaries for interactive browsing:

Section Content: {section_content}
Section Type: {section_type}

Generate:

1. **Quick Summary** (1 sentence overview)
2. **Key Points** (3-5 bullet points)
3. **Important Data** (specific numbers, dates, names)
4. **Related Sections** (cross-references)
5. **Action Items** (if any)

The summaries should allow users to quickly understand content without reading full sections.
```

## Section Deep-dive Analysis
```
Detailed Section Analysis Prompt:

User has selected to view detailed analysis of this section:

Section: {section_title}
Content: {section_content}
Context: {related_sections}

Provide comprehensive analysis:

1. **Content Analysis**
   - What this section contains
   - Purpose within the document
   - Key information extracted

2. **Data Validation**
   - Data quality assessment
   - Completeness check
   - Consistency with other sections

3. **Risk Indicators**
   - Potential issues identified
   - Red flags or concerns
   - Areas requiring attention

4. **Business Impact**
   - How this section affects overall assessment
   - Decision-making implications
   - Compliance considerations

Present in a format suitable for interactive display with expandable details.
```