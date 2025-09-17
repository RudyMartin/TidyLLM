"""
Compliance Service - Contains ALL business logic for compliance decisions
This is where authority tiers, precedence, and compliance rules live
"""

from typing import List, Dict, Any
from dataclasses import dataclass

from ..ports.outbound import ComplianceRepositoryPort, ComplianceRule


@dataclass
class ComplianceContext:
    """Context for compliance decisions"""
    domain: str
    risk_level: str
    regulatory_framework: str
    organization_type: str


class ComplianceService:
    """
    Domain service containing all compliance business logic.
    NO infrastructure concerns - pure business rules.
    """

    def __init__(self, compliance_repository: ComplianceRepositoryPort):
        """Initialize with repository port only"""
        self.repository = compliance_repository

    async def determine_authority_tier(self, context: ComplianceContext) -> int:
        """
        BUSINESS LOGIC: Determine which authority tier applies
        This logic was incorrectly in adapters before
        """
        # Regulatory tier for high-risk or financial services
        if context.risk_level == 'high' or context.organization_type == 'financial':
            return 1  # Regulatory

        # SOP tier for medium risk or established processes
        if context.risk_level == 'medium' or context.regulatory_framework:
            return 2  # SOP

        # Technical tier for everything else
        return 3  # Technical

    async def evaluate_compliance(self, query: str, context: ComplianceContext) -> Dict[str, Any]:
        """
        BUSINESS LOGIC: Evaluate compliance based on rules and context
        """
        # Determine authority tier based on business rules
        tier = await self.determine_authority_tier(context)

        # Get rules from repository (no business logic in adapter)
        rules = await self.repository.find_by_authority_tier(tier)

        # Apply business logic to evaluate compliance
        applicable_rules = self._filter_applicable_rules(rules, context)
        compliance_score = self._calculate_compliance_score(applicable_rules, query)
        recommendations = self._generate_recommendations(applicable_rules, compliance_score)

        return {
            'authority_tier': tier,
            'compliance_score': compliance_score,
            'applicable_rules': [r.rule_text for r in applicable_rules],
            'recommendations': recommendations,
            'decision': self._make_decision(compliance_score)
        }

    def _filter_applicable_rules(self, rules: List[ComplianceRule], context: ComplianceContext) -> List[ComplianceRule]:
        """
        BUSINESS LOGIC: Filter rules based on context
        """
        applicable = []
        for rule in rules:
            # Complex business logic for rule applicability
            if self._is_rule_applicable(rule, context):
                applicable.append(rule)

        # Sort by precedence (business rule)
        return sorted(applicable, key=lambda r: r.precedence, reverse=True)

    def _is_rule_applicable(self, rule: ComplianceRule, context: ComplianceContext) -> bool:
        """
        BUSINESS LOGIC: Determine if a rule applies to context
        """
        # This is where complex business rules would go
        # For now, simple domain matching
        return context.domain.lower() in rule.rule_text.lower()

    def _calculate_compliance_score(self, rules: List[ComplianceRule], query: str) -> float:
        """
        BUSINESS LOGIC: Calculate compliance score
        """
        if not rules:
            return 0.0

        # Weight rules by precedence
        total_weight = sum(r.precedence for r in rules)
        if total_weight == 0:
            return 0.5

        score = 0.0
        for rule in rules:
            # Business logic for scoring
            rule_score = self._score_rule_match(rule, query)
            weighted_score = rule_score * (rule.precedence / total_weight)
            score += weighted_score

        return min(1.0, max(0.0, score))

    def _score_rule_match(self, rule: ComplianceRule, query: str) -> float:
        """
        BUSINESS LOGIC: Score how well a rule matches query
        """
        # Simplified scoring - would be more complex in production
        query_lower = query.lower()
        rule_lower = rule.rule_text.lower()

        if query_lower in rule_lower or rule_lower in query_lower:
            return 1.0

        # Check for keyword matches
        query_words = set(query_lower.split())
        rule_words = set(rule_lower.split())
        overlap = len(query_words & rule_words)

        if overlap > 0:
            return overlap / max(len(query_words), len(rule_words))

        return 0.0

    def _generate_recommendations(self, rules: List[ComplianceRule], score: float) -> List[str]:
        """
        BUSINESS LOGIC: Generate recommendations based on rules and score
        """
        recommendations = []

        if score < 0.5:
            recommendations.append("Review compliance requirements with legal team")
            recommendations.append("Consider additional controls to meet regulations")

        if score < 0.8:
            recommendations.append("Document compliance rationale thoroughly")
            recommendations.append("Schedule periodic compliance reviews")

        if len(rules) < 3:
            recommendations.append("Expand compliance rule coverage for this domain")

        # Add rule-specific recommendations
        for rule in rules[:3]:  # Top 3 rules by precedence
            if rule.authority_tier == 1:
                recommendations.append(f"Ensure regulatory requirement met: {rule.id}")

        return recommendations

    def _make_decision(self, score: float) -> str:
        """
        BUSINESS LOGIC: Make compliance decision based on score
        """
        if score >= 0.8:
            return "COMPLIANT"
        elif score >= 0.5:
            return "CONDITIONALLY_COMPLIANT"
        else:
            return "NON_COMPLIANT"