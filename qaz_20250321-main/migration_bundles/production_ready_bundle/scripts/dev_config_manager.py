#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 Development Configuration Manager

This script provides a development-friendly way to create and manage basic configuration sets
during the development process. It allows you to manually create configurations without touching
core flow_config files, and later can be replaced with admin interfaces.

Usage:
    python3 dev_config_manager.py

Features:
- Create and manage risk categories
- Set up SME expertise areas
- Configure test scenarios
- Manage system parameters
- Export configurations to various formats
- Import configurations from files
"""

import os
import sys
import json
import yaml
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class RiskCategory:
    """Risk category configuration."""
    name: str
    description: str
    base_risk_score: int
    keywords: List[str]
    sme_expertise_required: bool
    validation_frequency: str  # daily, weekly, monthly, quarterly
    critical_threshold: int
    warning_threshold: int

@dataclass
class SMEExpert:
    """SME expert configuration."""
    sme_id: str
    name: str
    expertise_area: str
    validation_type: str  # initial, periodic, event-driven
    risk_tier: str  # low, medium, high, critical
    availability: str  # available, busy, offline
    contact_info: str
    specialties: List[str]

@dataclass
class TestScenario:
    """Test scenario configuration."""
    scenario_id: str
    name: str
    risk_category: str
    severity: str  # low, medium, high, critical
    description: str
    expected_response: str
    test_frequency: str  # daily, weekly, monthly
    is_active: bool

@dataclass
class SystemConfig:
    """System configuration parameters."""
    debug_mode: bool
    log_level: str
    max_response_time: float
    fallback_enabled: bool
    sme_integration_enabled: bool
    dspy_integration_enabled: bool
    performance_monitoring: bool

class DevConfigManager:
    """Development configuration manager."""
    
    def __init__(self, config_dir: str = "dev_configs"):
        self.config_dir = config_dir
        self.ensure_config_dir()
        
        # Initialize default configurations
        self.risk_categories = self.load_or_create_default_categories()
        self.sme_experts = self.load_or_create_default_experts()
        self.test_scenarios = self.load_or_create_default_scenarios()
        self.system_config = self.load_or_create_default_system_config()
    
    def ensure_config_dir(self):
        """Ensure configuration directory exists."""
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            print(f"📁 Created configuration directory: {self.config_dir}")
    
    def load_or_create_default_categories(self) -> List[RiskCategory]:
        """Load existing categories or create defaults."""
        filepath = os.path.join(self.config_dir, "risk_categories.json")
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                return [RiskCategory(**cat) for cat in data]
        else:
            # Create default categories
            default_categories = [
                RiskCategory(
                    name="Model Risk",
                    description="Machine learning model performance, validation, and governance",
                    base_risk_score=6,
                    keywords=["model", "validation", "testing", "performance", "drift"],
                    sme_expertise_required=True,
                    validation_frequency="weekly",
                    critical_threshold=8,
                    warning_threshold=6
                ),
                RiskCategory(
                    name="Credit Risk",
                    description="Lending, default rates, portfolio concentration, and credit scoring",
                    base_risk_score=7,
                    keywords=["credit", "default", "lending", "borrower", "collateral"],
                    sme_expertise_required=True,
                    validation_frequency="daily",
                    critical_threshold=9,
                    warning_threshold=7
                ),
                RiskCategory(
                    name="Market Risk",
                    description="VaR models, volatility, liquidity, and market exposure",
                    base_risk_score=8,
                    keywords=["market", "volatility", "var", "liquidity", "correlation"],
                    sme_expertise_required=True,
                    validation_frequency="daily",
                    critical_threshold=9,
                    warning_threshold=7
                ),
                RiskCategory(
                    name="Operational Risk",
                    description="System failures, processes, human error, and business continuity",
                    base_risk_score=5,
                    keywords=["operational", "system", "process", "human", "technology"],
                    sme_expertise_required=True,
                    validation_frequency="weekly",
                    critical_threshold=8,
                    warning_threshold=6
                )
            ]
            
            self.save_categories(default_categories)
            return default_categories
    
    def load_or_create_default_experts(self) -> List[SMEExpert]:
        """Load existing experts or create defaults."""
        filepath = os.path.join(self.config_dir, "sme_experts.json")
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                return [SMEExpert(**expert) for expert in data]
        else:
            # Create default experts
            default_experts = [
                SMEExpert(
                    sme_id="SME_001",
                    name="Dr. Sarah Johnson",
                    expertise_area="Model Risk",
                    validation_type="initial",
                    risk_tier="high",
                    availability="available",
                    contact_info="user@example.com",
                    specialties=["ML Validation", "Model Governance", "Performance Monitoring"]
                ),
                SMEExpert(
                    sme_id="SME_002",
                    name="Michael Chen",
                    expertise_area="Credit Risk",
                    validation_type="event-driven",
                    risk_tier="critical",
                    availability="available",
                    contact_info="user@example.com",
                    specialties=["Credit Scoring", "Portfolio Analysis", "Default Modeling"]
                ),
                SMEExpert(
                    sme_id="SME_003",
                    name="Dr. Emily Rodriguez",
                    expertise_area="Market Risk",
                    validation_type="periodic",
                    risk_tier="critical",
                    availability="available",
                    contact_info="user@example.com",
                    specialties=["VaR Modeling", "Stress Testing", "Market Analysis"]
                ),
                SMEExpert(
                    sme_id="SME_004",
                    name="James Wilson",
                    expertise_area="Operational Risk",
                    validation_type="initial",
                    risk_tier="medium",
                    availability="available",
                    contact_info="user@example.com",
                    specialties=["System Reliability", "Process Optimization", "Business Continuity"]
                )
            ]
            
            self.save_experts(default_experts)
            return default_experts
    
    def load_or_create_default_scenarios(self) -> List[TestScenario]:
        """Load existing scenarios or create defaults."""
        filepath = os.path.join(self.config_dir, "test_scenarios.json")
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                return [TestScenario(**scenario) for scenario in data]
        else:
            # Create default scenarios
            default_scenarios = [
                TestScenario(
                    scenario_id="1",
                    name="Model Performance Degradation",
                    risk_category="Model Risk",
                    severity="high",
                    description="Credit scoring model accuracy dropped from 95% to 78%",
                    expected_response="Should identify as high-risk, recommend immediate review",
                    test_frequency="weekly",
                    is_active=True
                ),
                TestScenario(
                    scenario_id="2",
                    name="VaR Model Breach",
                    risk_category="Market Risk",
                    severity="critical",
                    description="Value at Risk exceeded limits during market volatility",
                    expected_response="Should trigger immediate action, recommend model suspension",
                    test_frequency="daily",
                    is_active=True
                ),
                TestScenario(
                    scenario_id="3",
                    name="Data Quality Issues",
                    risk_category="Operational Risk",
                    severity="medium",
                    description="Training data completeness dropped below 90% threshold",
                    expected_response="Should flag for investigation, recommend data review",
                    test_frequency="weekly",
                    is_active=True
                ),
                TestScenario(
                    scenario_id="4",
                    name="Credit Portfolio Concentration",
                    risk_category="Credit Risk",
                    severity="high",
                    description="Portfolio concentration in high-risk sectors exceeded limits",
                    expected_response="Should recommend portfolio rebalancing, risk assessment",
                    test_frequency="daily",
                    is_active=True
                ),
                TestScenario(
                    scenario_id="5",
                    name="Real-time Fraud Detection Failure",
                    risk_category="Operational Risk",
                    severity="critical",
                    description="Fraud detection model stopped processing transactions",
                    expected_response="Should trigger emergency response, manual review process",
                    test_frequency="daily",
                    is_active=True
                )
            ]
            
            self.save_scenarios(default_scenarios)
            return default_scenarios
    
    def load_or_create_default_system_config(self) -> SystemConfig:
        """Load existing system config or create defaults."""
        filepath = os.path.join(self.config_dir, "system_config.json")
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                return SystemConfig(**data)
        else:
            # Create default system config
            default_config = SystemConfig(
                debug_mode=True,
                log_level="INFO",
                max_response_time=5.0,
                fallback_enabled=True,
                sme_integration_enabled=True,
                dspy_integration_enabled=True,
                performance_monitoring=True
            )
            
            self.save_system_config(default_config)
            return default_config
    
    def save_categories(self, categories: List[RiskCategory]):
        """Save risk categories to file."""
        filepath = os.path.join(self.config_dir, "risk_categories.json")
        with open(filepath, 'w') as f:
            json.dump([asdict(cat) for cat in categories], f, indent=2)
    
    def save_experts(self, experts: List[SMEExpert]):
        """Save SME experts to file."""
        filepath = os.path.join(self.config_dir, "sme_experts.json")
        with open(filepath, 'w') as f:
            json.dump([asdict(expert) for expert in experts], f, indent=2)
    
    def save_scenarios(self, scenarios: List[TestScenario]):
        """Save test scenarios to file."""
        filepath = os.path.join(self.config_dir, "test_scenarios.json")
        with open(filepath, 'w') as f:
            json.dump([asdict(scenario) for scenario in scenarios], f, indent=2)
    
    def save_system_config(self, config: SystemConfig):
        """Save system configuration to file."""
        filepath = os.path.join(self.config_dir, "system_config.json")
        with open(filepath, 'w') as f:
            json.dump(asdict(config), f, indent=2)
    
    def add_risk_category(self, category: RiskCategory):
        """Add a new risk category."""
        self.risk_categories.append(category)
        self.save_categories(self.risk_categories)
        print(f"✅ Added risk category: {category.name}")
    
    def add_sme_expert(self, expert: SMEExpert):
        """Add a new SME expert."""
        self.sme_experts.append(expert)
        self.save_experts(self.sme_experts)
        print(f"✅ Added SME expert: {expert.name}")
    
    def add_test_scenario(self, scenario: TestScenario):
        """Add a new test scenario."""
        self.test_scenarios.append(scenario)
        self.save_scenarios(self.test_scenarios)
        print(f"✅ Added test scenario: {scenario.name}")
    
    def update_system_config(self, config: SystemConfig):
        """Update system configuration."""
        self.system_config = config
        self.save_system_config(config)
        print("✅ Updated system configuration")
    
    def export_to_sql(self, output_file: str = "dev_configs.sql"):
        """Export configurations to SQL format."""
        sql_content = []
        sql_content.append("-- Development Configuration Export")
        sql_content.append(f"-- Generated: {datetime.now().isoformat()}")
        sql_content.append("")
        
        # Export risk categories
        sql_content.append("-- Risk Categories")
        for cat in self.risk_categories:
            sql_content.append(f"INSERT INTO risk_categories (name, description, base_risk_score, keywords, sme_expertise_required, validation_frequency, critical_threshold, warning_threshold) VALUES")
            sql_content.append(f"('{cat.name}', '{cat.description}', {cat.base_risk_score}, '{json.dumps(cat.keywords)}', {str(cat.sme_expertise_required).lower()}, '{cat.validation_frequency}', {cat.critical_threshold}, {cat.warning_threshold});")
            sql_content.append("")
        
        # Export SME experts
        sql_content.append("-- SME Experts")
        for expert in self.sme_experts:
            sql_content.append(f"INSERT INTO sme_experts (sme_id, name, expertise_area, validation_type, risk_tier, availability, contact_info, specialties) VALUES")
            sql_content.append(f"('{expert.sme_id}', '{expert.name}', '{expert.expertise_area}', '{expert.validation_type}', '{expert.risk_tier}', '{expert.availability}', '{expert.contact_info}', '{json.dumps(expert.specialties)}');")
            sql_content.append("")
        
        # Export test scenarios
        sql_content.append("-- Test Scenarios")
        for scenario in self.test_scenarios:
            sql_content.append(f"INSERT INTO test_scenarios (scenario_id, name, risk_category, severity, description, expected_response, test_frequency, is_active) VALUES")
            sql_content.append(f"('{scenario.scenario_id}', '{scenario.name}', '{scenario.risk_category}', '{scenario.severity}', '{scenario.description}', '{scenario.expected_response}', '{scenario.test_frequency}', {str(scenario.is_active).lower()});")
            sql_content.append("")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(sql_content))
        
        print(f"✅ Exported configurations to SQL: {output_file}")
    
    def export_to_yaml(self, output_file: str = "dev_configs.yaml"):
        """Export configurations to YAML format."""
        config_data = {
            "risk_categories": [asdict(cat) for cat in self.risk_categories],
            "sme_experts": [asdict(expert) for expert in self.sme_experts],
            "test_scenarios": [asdict(scenario) for scenario in self.test_scenarios],
            "system_config": asdict(self.system_config),
            "export_info": {
                "generated": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
        
        with open(output_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        
        print(f"✅ Exported configurations to YAML: {output_file}")
    
    def import_from_yaml(self, input_file: str):
        """Import configurations from YAML file."""
        try:
            with open(input_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Import risk categories
            if "risk_categories" in config_data:
                self.risk_categories = [RiskCategory(**cat) for cat in config_data["risk_categories"]]
                self.save_categories(self.risk_categories)
            
            # Import SME experts
            if "sme_experts" in config_data:
                self.sme_experts = [SMEExpert(**expert) for expert in config_data["sme_experts"]]
                self.save_experts(self.sme_experts)
            
            # Import test scenarios
            if "test_scenarios" in config_data:
                self.test_scenarios = [TestScenario(**scenario) for scenario in config_data["test_scenarios"]]
                self.save_scenarios(self.test_scenarios)
            
            # Import system config
            if "system_config" in config_data:
                self.system_config = SystemConfig(**config_data["system_config"])
                self.save_system_config(self.system_config)
            
            print(f"✅ Imported configurations from: {input_file}")
            
        except Exception as e:
            print(f"❌ Error importing configurations: {e}")
    
    def show_summary(self):
        """Show configuration summary."""
        print("\n" + "="*80)
        print("🔧 DEVELOPMENT CONFIGURATION SUMMARY")
        print("="*80)
        
        print(f"\n📊 Risk Categories: {len(self.risk_categories)}")
        for cat in self.risk_categories:
            print(f"   • {cat.name} (Base Score: {cat.base_risk_score}, Validation: {cat.validation_frequency})")
        
        print(f"\n👥 SME Experts: {len(self.sme_experts)}")
        for expert in self.sme_experts:
            print(f"   • {expert.name} ({expert.expertise_area}, {expert.availability})")
        
        print(f"\n🧪 Test Scenarios: {len(self.test_scenarios)}")
        active_scenarios = [s for s in self.test_scenarios if s.is_active]
        print(f"   • Active: {len(active_scenarios)}")
        for scenario in active_scenarios:
            print(f"     - {scenario.name} ({scenario.risk_category}, {scenario.severity})")
        
        print(f"\n⚙️ System Configuration:")
        print(f"   • Debug Mode: {self.system_config.debug_mode}")
        print(f"   • Log Level: {self.system_config.log_level}")
        print(f"   • Max Response Time: {self.system_config.max_response_time}s")
        print(f"   • Fallback Enabled: {self.system_config.fallback_enabled}")
        print(f"   • SME Integration: {self.system_config.sme_integration_enabled}")
        print(f"   • DSPy Integration: {self.system_config.dspy_integration_enabled}")
        print(f"   • Performance Monitoring: {self.system_config.performance_monitoring}")
        
        print(f"\n📁 Configuration Directory: {self.config_dir}")
        print("="*80)

def main():
    """Main function for development configuration management."""
    print("🔧 Development Configuration Manager")
    print("="*60)
    
    manager = DevConfigManager()
    
    while True:
        print("\nOptions:")
        print("1. Show configuration summary")
        print("2. Add new risk category")
        print("3. Add new SME expert")
        print("4. Add new test scenario")
        print("5. Update system configuration")
        print("6. Export to SQL")
        print("7. Export to YAML")
        print("8. Import from YAML")
        print("9. Exit")
        
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == "1":
            manager.show_summary()
        
        elif choice == "2":
            print("\n📋 Adding New Risk Category")
            name = input("Category name: ").strip()
            description = input("Description: ").strip()
            base_score = int(input("Base risk score (1-10): ").strip())
            keywords = input("Keywords (comma-separated): ").strip().split(",")
            validation_freq = input("Validation frequency (daily/weekly/monthly/quarterly): ").strip()
            critical_thresh = int(input("Critical threshold (1-10): ").strip())
            warning_thresh = int(input("Warning threshold (1-10): ").strip())
            
            category = RiskCategory(
                name=name,
                description=description,
                base_risk_score=base_score,
                keywords=keywords,
                sme_expertise_required=True,
                validation_frequency=validation_freq,
                critical_threshold=critical_thresh,
                warning_threshold=warning_thresh
            )
            manager.add_risk_category(category)
        
        elif choice == "3":
            print("\n👥 Adding New SME Expert")
            sme_id = input("SME ID: ").strip()
            name = input("Name: ").strip()
            expertise = input("Expertise area: ").strip()
            validation_type = input("Validation type (initial/periodic/event-driven): ").strip()
            risk_tier = input("Risk tier (low/medium/high/critical): ").strip()
            availability = input("Availability (available/busy/offline): ").strip()
            contact = input("Contact info: ").strip()
            specialties = input("Specialties (comma-separated): ").strip().split(",")
            
            expert = SMEExpert(
                sme_id=sme_id,
                name=name,
                expertise_area=expertise,
                validation_type=validation_type,
                risk_tier=risk_tier,
                availability=availability,
                contact_info=contact,
                specialties=specialties
            )
            manager.add_sme_expert(expert)
        
        elif choice == "4":
            print("\n🧪 Adding New Test Scenario")
            scenario_id = input("Scenario ID: ").strip()
            name = input("Name: ").strip()
            category = input("Risk category: ").strip()
            severity = input("Severity (low/medium/high/critical): ").strip()
            description = input("Description: ").strip()
            expected = input("Expected response: ").strip()
            frequency = input("Test frequency (daily/weekly/monthly): ").strip()
            active = input("Active (y/n): ").strip().lower() == 'y'
            
            scenario = TestScenario(
                scenario_id=scenario_id,
                name=name,
                risk_category=category,
                severity=severity,
                description=description,
                expected_response=expected,
                test_frequency=frequency,
                is_active=active
            )
            manager.add_test_scenario(scenario)
        
        elif choice == "5":
            print("\n⚙️ Updating System Configuration")
            debug = input("Debug mode (y/n): ").strip().lower() == 'y'
            log_level = input("Log level (DEBUG/INFO/WARNING/ERROR): ").strip()
            max_time = float(input("Max response time (seconds): ").strip())
            fallback = input("Fallback enabled (y/n): ").strip().lower() == 'y'
            sme = input("SME integration enabled (y/n): ").strip().lower() == 'y'
            dspy = input("DSPy integration enabled (y/n): ").strip().lower() == 'y'
            perf = input("Performance monitoring (y/n): ").strip().lower() == 'y'
            
            config = SystemConfig(
                debug_mode=debug,
                log_level=log_level,
                max_response_time=max_time,
                fallback_enabled=fallback,
                sme_integration_enabled=sme,
                dspy_integration_enabled=dspy,
                performance_monitoring=perf
            )
            manager.update_system_config(config)
        
        elif choice == "6":
            output_file = input("Output SQL file (default: dev_configs.sql): ").strip() or "dev_configs.sql"
            manager.export_to_sql(output_file)
        
        elif choice == "7":
            output_file = input("Output YAML file (default: dev_configs.yaml): ").strip() or "dev_configs.yaml"
            manager.export_to_yaml(output_file)
        
        elif choice == "8":
            input_file = input("Input YAML file: ").strip()
            manager.import_from_yaml(input_file)
        
        elif choice == "9":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-9.")

if __name__ == "__main__":
    main()
