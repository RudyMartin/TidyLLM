# Code Review Report Template

## Executive Summary
**Review ID:** {review_id}
**System:** {system_name}
**Reviewer:** {reviewer}
**Date:** {review_date}
**Overall Status:** {overall_status}
**Overall Score:** {overall_score}/1.0

## Assessment Overview

### Tier Scores
- **Tier 1 (Critical):** {tier1_score}/1.0 - {tier1_status}
- **Tier 2 (Important):** {tier2_score}/1.0 - {tier2_status}
- **Tier 3 (Standard):** {tier3_score}/1.0 - {tier3_status}

### Compliance Status
- **Architectural:** {architectural_status}
- **Security:** {security_status}
- **Performance:** {performance_status}
- **Regulatory:** {regulatory_status}

## Detailed Findings

### Critical Issues (Tier 1)
{tier1_issues}

### Important Issues (Tier 2)
{tier2_issues}

### Standard Issues (Tier 3)
{tier3_issues}

## Action Plan
{action_plan}

## Recommendations
{recommendations}

## Compliance References
- **Architecture Specs:** V2 Clean Architecture, Hexagonal Architecture
- **Security Specs:** S3-First Architecture, AWS Secrets Manager
- **Performance Specs:** MLflow Integration, Optimization Patterns
- **Regulatory Specs:** SR-11-7, Basel-III, SOX-404

---
**Output Requirements:**
- JSON Report: `{workflow_path}/outputs/{review_id}_report.json`
- PDF Report: `{workflow_path}/outputs/{review_id}_report.pdf`