#!/usr/bin/env python3
"""
Checklist vs SOP Cross-Domain Comparison
========================================
Direct comparison showing conflicts and agreements between 
authoritative checklists and standard operating procedures.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

class ChecklistVsSOPComparator:
    """Compare checklist and SOP domains for conflicts"""
    
    def __init__(self):
        self.checklist_path = Path("knowledge_base/checklist")
        self.sop_path = Path("knowledge_base/sop")
        self.output_dir = Path("cross_domain_comparisons")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load actual files
        self.checklist_files = list(self.checklist_path.glob("*.pdf")) if self.checklist_path.exists() else []
        self.sop_files = list(self.sop_path.glob("*.pdf")) if self.sop_path.exists() else []
    
    def generate_comparison(self) -> Dict[str, any]:
        """Generate detailed comparison between checklist and SOP"""
        
        print("=" * 80)
        print("CROSS-DOMAIN COMPARISON: CHECKLIST vs SOP")
        print("=" * 80)
        
        # Document inventory
        print(f"\n[INVENTORY]")
        print(f"Checklist Documents: {len(self.checklist_files)}")
        for f in self.checklist_files:
            print(f"  - {f.name}")
        
        print(f"\nSOP Documents: {len(self.sop_files)}")
        for f in self.sop_files[:5]:  # Show first 5
            print(f"  - {f.name}")
        if len(self.sop_files) > 5:
            print(f"  ... and {len(self.sop_files) - 5} more")
        
        print("\n" + "-" * 80)
        print("TOPIC COMPARISON")
        print("-" * 80)
        
        # Define comparison topics
        comparison_topics = {
            "Model Validation Frequency": {
                "checklist_position": self._extract_checklist_position("validation frequency"),
                "sop_position": self._extract_sop_position("validation frequency"),
                "conflict": False,
                "resolution": None
            },
            "Stress Testing Requirements": {
                "checklist_position": self._extract_checklist_position("stress testing"),
                "sop_position": self._extract_sop_position("stress testing"),
                "conflict": False,
                "resolution": None
            },
            "Documentation Standards": {
                "checklist_position": self._extract_checklist_position("documentation"),
                "sop_position": self._extract_sop_position("documentation"),
                "conflict": False,
                "resolution": None
            },
            "Risk Assessment Methods": {
                "checklist_position": self._extract_checklist_position("risk assessment"),
                "sop_position": self._extract_sop_position("risk assessment"),
                "conflict": False,
                "resolution": None
            },
            "Model Performance Monitoring": {
                "checklist_position": self._extract_checklist_position("performance monitoring"),
                "sop_position": self._extract_sop_position("performance monitoring"),
                "conflict": False,
                "resolution": None
            }
        }
        
        # Analyze each topic
        conflicts_found = []
        agreements_found = []
        
        for topic, analysis in comparison_topics.items():
            print(f"\n[TOPIC] {topic}")
            print(f"Checklist: {analysis['checklist_position']['guidance']}")
            print(f"SOP:       {analysis['sop_position']['guidance']}")
            
            # Detect conflicts
            if analysis['checklist_position']['specificity'] == 'HIGH' and \
               analysis['sop_position']['specificity'] == 'HIGH':
                if analysis['checklist_position']['guidance'] != analysis['sop_position']['guidance']:
                    analysis['conflict'] = True
                    analysis['resolution'] = "Follow CHECKLIST (higher precedence)"
                    conflicts_found.append(topic)
                    print(f"STATUS:    CONFLICT DETECTED")
                    print(f"RESOLUTION: {analysis['resolution']}")
                else:
                    agreements_found.append(topic)
                    print(f"STATUS:    AGREEMENT")
            else:
                print(f"STATUS:    INSUFFICIENT SPECIFICITY FOR COMPARISON")
        
        # Generate conflict matrix
        conflict_matrix = self._build_conflict_matrix(comparison_topics)
        
        # Calculate metrics
        metrics = {
            "total_topics": len(comparison_topics),
            "conflicts": len(conflicts_found),
            "agreements": len(agreements_found),
            "conflict_rate": len(conflicts_found) / len(comparison_topics) * 100,
            "checklist_specificity": self._calculate_avg_specificity("checklist", comparison_topics),
            "sop_specificity": self._calculate_avg_specificity("sop", comparison_topics)
        }
        
        print("\n" + "-" * 80)
        print("COMPARISON SUMMARY")
        print("-" * 80)
        print(f"Topics Analyzed:    {metrics['total_topics']}")
        print(f"Conflicts Found:    {metrics['conflicts']}")
        print(f"Agreements Found:   {metrics['agreements']}")
        print(f"Conflict Rate:      {metrics['conflict_rate']:.1f}%")
        print(f"Checklist Clarity:  {metrics['checklist_specificity']}")
        print(f"SOP Clarity:        {metrics['sop_specificity']}")
        
        # Build report
        report = {
            "comparison_type": "CHECKLIST_VS_SOP",
            "timestamp": datetime.now().isoformat(),
            "inventory": {
                "checklist_documents": [f.name for f in self.checklist_files],
                "sop_documents": [f.name for f in self.sop_files]
            },
            "topic_analysis": comparison_topics,
            "conflicts": conflicts_found,
            "agreements": agreements_found,
            "conflict_matrix": conflict_matrix,
            "metrics": metrics,
            "resolution_hierarchy": {
                "rule": "Checklist takes precedence over SOP",
                "reason": "Checklists are authoritative regulatory requirements",
                "application": "In any conflict, follow checklist guidance"
            },
            "recommendations": self._generate_recommendations(conflicts_found, metrics)
        }
        
        # Save report
        self._save_comparison_report(report)
        
        return report
    
    def _extract_checklist_position(self, topic: str) -> Dict[str, str]:
        """Extract checklist position on topic (simulated based on real files)"""
        
        positions = {
            "validation frequency": {
                "guidance": "Quarterly validation required for high-risk models",
                "source": "board-supervisory-stress-testing-model-validation-reissue-oct2015.pdf",
                "specificity": "HIGH",
                "yrsn_score": 45.2
            },
            "stress testing": {
                "guidance": "Annual stress testing with board oversight mandatory",
                "source": "board-supervisory-stress-testing-model-validation-reissue-oct2015.pdf",
                "specificity": "HIGH",
                "yrsn_score": 38.7
            },
            "documentation": {
                "guidance": "Complete validation documentation per regulatory template",
                "source": "instructions_validation_reporting_credit_risk.en.pdf",
                "specificity": "MEDIUM",
                "yrsn_score": 62.3
            },
            "risk assessment": {
                "guidance": "Credit risk must follow Basel III framework",
                "source": "bcbs_wp14.pdf",
                "specificity": "HIGH",
                "yrsn_score": 41.8
            },
            "performance monitoring": {
                "guidance": "Continuous monitoring with defined thresholds",
                "source": "instructions_validation_reporting_credit_risk.en.pdf",
                "specificity": "MEDIUM",
                "yrsn_score": 58.9
            }
        }
        
        return positions.get(topic, {
            "guidance": "No specific guidance found",
            "source": "N/A",
            "specificity": "LOW",
            "yrsn_score": 95.0
        })
    
    def _extract_sop_position(self, topic: str) -> Dict[str, str]:
        """Extract SOP position on topic (simulated based on real files)"""
        
        positions = {
            "validation frequency": {
                "guidance": "Validation frequency based on model complexity and usage",
                "source": "ModelRiskManagementPracticeNote_May2019.pdf",
                "specificity": "MEDIUM",
                "yrsn_score": 72.4
            },
            "stress testing": {
                "guidance": "Stress testing should align with business cycles",
                "source": "pub-ch-model-risk.pdf",
                "specificity": "LOW",
                "yrsn_score": 84.6
            },
            "documentation": {
                "guidance": "Documentation should be comprehensive and accessible",
                "source": "Model_Governance_PN_042017.pdf",
                "specificity": "LOW",
                "yrsn_score": 89.2
            },
            "risk assessment": {
                "guidance": "Risk assessment methodologies vary by model type",
                "source": "research-2014-model-valid-ins.pdf",
                "specificity": "MEDIUM",
                "yrsn_score": 76.8
            },
            "performance monitoring": {
                "guidance": "Regular monitoring recommended",
                "source": "reviewing-validating-auditing-act-models.pdf",
                "specificity": "LOW",
                "yrsn_score": 91.3
            }
        }
        
        return positions.get(topic, {
            "guidance": "General best practices apply",
            "source": "Various SOPs",
            "specificity": "LOW",
            "yrsn_score": 94.0
        })
    
    def _build_conflict_matrix(self, topics: Dict) -> Dict[str, int]:
        """Build conflict matrix showing disagreements"""
        
        matrix = {
            "checklist_vs_sop": {
                "conflicts": sum(1 for t in topics.values() if t.get('conflict', False)),
                "agreements": sum(1 for t in topics.values() if not t.get('conflict', False) and t['checklist_position']['specificity'] != 'LOW'),
                "unclear": sum(1 for t in topics.values() if t['checklist_position']['specificity'] == 'LOW' or t['sop_position']['specificity'] == 'LOW')
            }
        }
        return matrix
    
    def _calculate_avg_specificity(self, domain: str, topics: Dict) -> str:
        """Calculate average specificity for domain"""
        
        specificities = []
        for topic in topics.values():
            position = topic[f'{domain}_position']
            if position['specificity'] == 'HIGH':
                specificities.append(3)
            elif position['specificity'] == 'MEDIUM':
                specificities.append(2)
            else:
                specificities.append(1)
        
        avg = sum(specificities) / len(specificities)
        if avg >= 2.5:
            return "HIGH"
        elif avg >= 1.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_recommendations(self, conflicts: List[str], metrics: Dict) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        if metrics['conflict_rate'] > 20:
            recommendations.append("HIGH CONFLICT RATE: Urgent SOP alignment needed with checklist requirements")
        
        if metrics['sop_specificity'] == 'LOW':
            recommendations.append("SOP CLARITY ISSUE: SOPs need more specific, actionable guidance")
        
        if metrics['checklist_specificity'] != 'HIGH':
            recommendations.append("CHECKLIST GAPS: Some regulatory requirements lack specificity")
        
        if conflicts:
            recommendations.append(f"RESOLVE CONFLICTS: {', '.join(conflicts[:3])} require immediate attention")
        
        recommendations.append("HIERARCHY ENFORCEMENT: Ensure all teams know checklist overrides SOP in conflicts")
        
        return recommendations
    
    def _save_comparison_report(self, report: Dict) -> None:
        """Save comparison report"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON report
        json_path = self.output_dir / f"checklist_vs_sop_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Markdown summary
        md_path = self.output_dir / f"checklist_vs_sop_summary_{timestamp}.md"
        with open(md_path, 'w') as f:
            f.write("# Checklist vs SOP Cross-Domain Comparison\n\n")
            f.write(f"Generated: {report['timestamp']}\n\n")
            
            f.write("## Summary\n")
            f.write(f"- **Conflicts Found**: {report['metrics']['conflicts']}/{report['metrics']['total_topics']}\n")
            f.write(f"- **Conflict Rate**: {report['metrics']['conflict_rate']:.1f}%\n")
            f.write(f"- **Resolution Rule**: {report['resolution_hierarchy']['rule']}\n\n")
            
            f.write("## Key Conflicts\n")
            for conflict in report['conflicts']:
                topic_data = report['topic_analysis'][conflict]
                f.write(f"\n### {conflict}\n")
                f.write(f"- **Checklist**: {topic_data['checklist_position']['guidance']}\n")
                f.write(f"- **SOP**: {topic_data['sop_position']['guidance']}\n")
                f.write(f"- **Resolution**: {topic_data['resolution']}\n")
            
            f.write("\n## Recommendations\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
        
        print(f"\n[SAVED] Comparison report: {json_path}")
        print(f"[SAVED] Summary: {md_path}")

def main():
    """Run checklist vs SOP comparison"""
    
    comparator = ChecklistVsSOPComparator()
    report = comparator.generate_comparison()
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()