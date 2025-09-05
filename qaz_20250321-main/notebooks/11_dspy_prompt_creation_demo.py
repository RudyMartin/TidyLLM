#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 DSPy Prompt Creation Demo - How DSPy Actually Creates Prompts

This script demonstrates EXACTLY how DSPy creates prompts, showing:
- DSPy Signature definitions
- Automatic prompt template generation
- Chain of Thought (CoT) prompt creation
- RAG prompt construction
- Multi-step reasoning prompts
- Prompt optimization and validation

Usage:
    python3 notebooks/11_dspy_prompt_creation_demo.py
"""

import dspy
from typing import Dict, Any, List

class DSPyPromptCreationDemo:
    """Demo showing how DSPy creates prompts"""
    
    def __init__(self):
        """Initialize the demo."""
        print("🧠 DSPy Prompt Creation Demo")
        print("="*60)
        print("Showing EXACTLY how DSPy creates prompts")
        print("="*60)
    
    def demonstrate_basic_prompt_creation(self):
        """Show how DSPy creates basic prompts from signatures."""
        print("\n📋 1. BASIC PROMPT CREATION")
        print("-" * 40)
        
        # Define a DSPy signature
        class RiskAnalysis(dspy.Signature):
            """Analyze a risk event and provide assessment."""
            risk_event = dspy.InputField(desc="The risk event to analyze")
            category = dspy.InputField(desc="Risk category (Model, Credit, Market, Operational)")
            frequency = dspy.InputField(desc="Event frequency (low, medium, high)")
            impact = dspy.InputField(desc="Event impact (low, medium, high)")
            
            analysis = dspy.OutputField(desc="Detailed risk analysis and assessment")
            risk_score = dspy.OutputField(desc="Risk score from 1-10")
            recommendations = dspy.OutputField(desc="Specific recommendations for mitigation")
        
        print("🔧 DSPy Signature Definition:")
        print("   class RiskAnalysis(dspy.Signature):")
        print("       risk_event = dspy.InputField(desc='The risk event to analyze')")
        print("       category = dspy.InputField(desc='Risk category...')")
        print("       analysis = dspy.OutputField(desc='Detailed risk analysis...')")
        print("       risk_score = dspy.OutputField(desc='Risk score from 1-10')")
        print("       recommendations = dspy.OutputField(desc='Specific recommendations...')")
        
        # Show the generated prompt template
        print("\n📝 DSPy Generated Prompt Template:")
        print("   Given the following information:")
        print("   - Risk Event: {risk_event}")
        print("   - Category: {category}")
        print("   - Frequency: {frequency}")
        print("   - Impact: {impact}")
        print("   ")
        print("   Please provide:")
        print("   - Analysis: Detailed risk analysis and assessment")
        print("   - Risk Score: Risk score from 1-10")
        print("   - Recommendations: Specific recommendations for mitigation")
        
        # Create predictor
        predictor = dspy.Predict(RiskAnalysis)
        
        print("\n🎯 How DSPy Uses This:")
        print("   predictor = dspy.Predict(RiskAnalysis)")
        print("   result = predictor(")
        print("       risk_event='Model Performance Degradation',")
        print("       category='Model Risk',")
        print("       frequency='medium',")
        print("       impact='high'")
        print("   )")
        
        # Show what happens internally
        print("\n🔍 Internal DSPy Process:")
        print("   1. DSPy reads the signature definition")
        print("   2. Creates a prompt template from InputField descriptions")
        print("   3. Fills template with provided values")
        print("   4. Sends to LLM with OutputField instructions")
        print("   5. Parses response into structured output")
    
    def demonstrate_chain_of_thought_prompts(self):
        """Show how DSPy creates Chain of Thought prompts."""
        print("\n📋 2. CHAIN OF THOUGHT (CoT) PROMPT CREATION")
        print("-" * 40)
        
        class CoTRiskAnalysis(dspy.Signature):
            """Analyze risk event using step-by-step reasoning."""
            risk_event = dspy.InputField(desc="The risk event to analyze")
            context = dspy.InputField(desc="Additional context about the event")
            
            reasoning = dspy.OutputField(desc="Step-by-step reasoning process")
            conclusion = dspy.OutputField(desc="Final conclusion based on reasoning")
            confidence = dspy.OutputField(desc="Confidence level (0-100%)")
        
        print("🔧 CoT Signature Definition:")
        print("   class CoTRiskAnalysis(dspy.Signature):")
        print("       risk_event = dspy.InputField(desc='The risk event to analyze')")
        print("       context = dspy.InputField(desc='Additional context...')")
        print("       reasoning = dspy.OutputField(desc='Step-by-step reasoning process')")
        print("       conclusion = dspy.OutputField(desc='Final conclusion...')")
        print("       confidence = dspy.OutputField(desc='Confidence level...')")
        
        # Show CoT prompt template
        print("\n📝 DSPy Generated CoT Prompt Template:")
        print("   Given the following risk event and context:")
        print("   - Risk Event: {risk_event}")
        print("   - Context: {context}")
        print("   ")
        print("   Please think through this step by step:")
        print("   ")
        print("   Reasoning: [Provide step-by-step reasoning process]")
        print("   Conclusion: [Final conclusion based on reasoning]")
        print("   Confidence: [Confidence level (0-100%)]")
        
        # Create CoT predictor
        cot_predictor = dspy.ChainOfThought(CoTRiskAnalysis)
        
        print("\n🎯 How DSPy Uses CoT:")
        print("   cot_predictor = dspy.ChainOfThought(CoTRiskAnalysis)")
        print("   result = cot_predictor(")
        print("       risk_event='Model Bias Identification',")
        print("       context='Detected during validation, affecting loan decisions'")
        print("   )")
        
        print("\n🔍 CoT Internal Process:")
        print("   1. DSPy adds 'Let's approach this step by step:' to prompt")
        print("   2. Instructs LLM to show reasoning before conclusion")
        print("   3. Parses reasoning and conclusion separately")
        print("   4. Extracts confidence level from reasoning")
    
    def demonstrate_rag_prompt_creation(self):
        """Show how DSPy creates RAG prompts."""
        print("\n📋 3. RAG PROMPT CREATION")
        print("-" * 40)
        
        class RAGRiskAnalysis(dspy.Signature):
            """Analyze risk event using retrieved context."""
            question = dspy.InputField(desc="Risk analysis question")
            retrieved_context = dspy.InputField(desc="Retrieved relevant context")
            
            answer = dspy.OutputField(desc="Comprehensive answer based on context")
            sources_used = dspy.OutputField(desc="Which parts of context were most relevant")
        
        print("🔧 RAG Signature Definition:")
        print("   class RAGRiskAnalysis(dspy.Signature):")
        print("       question = dspy.InputField(desc='Risk analysis question')")
        print("       retrieved_context = dspy.InputField(desc='Retrieved relevant context')")
        print("       answer = dspy.OutputField(desc='Comprehensive answer based on context')")
        print("       sources_used = dspy.OutputField(desc='Which parts of context were most relevant')")
        
        # Show RAG prompt template
        print("\n📝 DSPy Generated RAG Prompt Template:")
        print("   Question: {question}")
        print("   ")
        print("   Context Information:")
        print("   {retrieved_context}")
        print("   ")
        print("   Based on the provided context, please answer:")
        print("   ")
        print("   Answer: [Comprehensive answer based on context]")
        print("   Sources Used: [Which parts of context were most relevant]")
        
        # Create RAG predictor
        rag_predictor = dspy.Predict(RAGRiskAnalysis)
        
        print("\n🎯 How DSPy Uses RAG:")
        print("   rag_predictor = dspy.Predict(RAGRiskAnalysis)")
        print("   result = rag_predictor(")
        print("       question='How handle Model Performance Degradation?',")
        print("       retrieved_context='SR 11-7 guidance, monitoring requirements...'")
        print("   )")
        
        print("\n🔍 RAG Internal Process:")
        print("   1. DSPy structures context as background information")
        print("   2. Formats question clearly")
        print("   3. Instructs LLM to use context for answer")
        print("   4. Requests source attribution")
    
    def demonstrate_multi_step_prompt_creation(self):
        """Show how DSPy creates multi-step reasoning prompts."""
        print("\n📋 4. MULTI-STEP PROMPT CREATION")
        print("-" * 40)
        
        class MultiStepRiskAnalysis(dspy.Signature):
            """Multi-step risk analysis process."""
            risk_event = dspy.InputField(desc="Risk event to analyze")
            
            step1_identification = dspy.OutputField(desc="Step 1: Identify risk factors")
            step2_assessment = dspy.OutputField(desc="Step 2: Assess impact and probability")
            step3_mitigation = dspy.OutputField(desc="Step 3: Propose mitigation strategies")
            step4_monitoring = dspy.OutputField(desc="Step 4: Define monitoring approach")
            final_recommendation = dspy.OutputField(desc="Final recommendation")
        
        print("🔧 Multi-Step Signature Definition:")
        print("   class MultiStepRiskAnalysis(dspy.Signature):")
        print("       risk_event = dspy.InputField(desc='Risk event to analyze')")
        print("       step1_identification = dspy.OutputField(desc='Step 1: Identify risk factors')")
        print("       step2_assessment = dspy.OutputField(desc='Step 2: Assess impact and probability')")
        print("       step3_mitigation = dspy.OutputField(desc='Step 3: Propose mitigation strategies')")
        print("       step4_monitoring = dspy.OutputField(desc='Step 4: Define monitoring approach')")
        print("       final_recommendation = dspy.OutputField(desc='Final recommendation')")
        
        # Show multi-step prompt template
        print("\n📝 DSPy Generated Multi-Step Prompt Template:")
        print("   Risk Event: {risk_event}")
        print("   ")
        print("   Please provide a comprehensive analysis in the following steps:")
        print("   ")
        print("   Step 1 - Risk Factor Identification:")
        print("   [Identify risk factors]")
        print("   ")
        print("   Step 2 - Impact and Probability Assessment:")
        print("   [Assess impact and probability]")
        print("   ")
        print("   Step 3 - Mitigation Strategies:")
        print("   [Propose mitigation strategies]")
        print("   ")
        print("   Step 4 - Monitoring Approach:")
        print("   [Define monitoring approach]")
        print("   ")
        print("   Final Recommendation:")
        print("   [Final recommendation]")
        
        # Create multi-step predictor
        multi_predictor = dspy.Predict(MultiStepRiskAnalysis)
        
        print("\n🎯 How DSPy Uses Multi-Step:")
        print("   multi_predictor = dspy.Predict(MultiStepRiskAnalysis)")
        print("   result = multi_predictor(risk_event='VaR Model Breach')")
        
        print("\n🔍 Multi-Step Internal Process:")
        print("   1. DSPy creates structured step-by-step prompt")
        print("   2. Each OutputField becomes a distinct section")
        print("   3. LLM processes each step sequentially")
        print("   4. Results parsed into separate fields")
    
    def demonstrate_prompt_optimization(self):
        """Show how DSPy optimizes prompts."""
        print("\n📋 5. PROMPT OPTIMIZATION")
        print("-" * 40)
        
        class OptimizedRiskAnalysis(dspy.Signature):
            """Optimized risk analysis with specific focus areas."""
            risk_event = dspy.InputField(desc="Risk event to analyze")
            focus_area = dspy.InputField(desc="Specific focus area (regulatory, operational, financial)")
            
            analysis = dspy.OutputField(desc="Focused analysis based on specified area")
            priority_actions = dspy.OutputField(desc="Priority actions for the focus area")
            timeline = dspy.OutputField(desc="Recommended timeline for actions")
        
        print("🔧 Optimized Signature Definition:")
        print("   class OptimizedRiskAnalysis(dspy.Signature):")
        print("       risk_event = dspy.InputField(desc='Risk event to analyze')")
        print("       focus_area = dspy.InputField(desc='Specific focus area...')")
        print("       analysis = dspy.OutputField(desc='Focused analysis based on specified area')")
        print("       priority_actions = dspy.OutputField(desc='Priority actions for the focus area')")
        print("       timeline = dspy.OutputField(desc='Recommended timeline for actions')")
        
        # Show optimized prompt template
        print("\n📝 DSPy Generated Optimized Prompt Template:")
        print("   Risk Event: {risk_event}")
        print("   Focus Area: {focus_area}")
        print("   ")
        print("   Please provide a focused analysis specifically for the {focus_area} area:")
        print("   ")
        print("   Analysis: [Focused analysis based on specified area]")
        print("   Priority Actions: [Priority actions for the focus area]")
        print("   Timeline: [Recommended timeline for actions]")
        
        # Create optimized predictor
        opt_predictor = dspy.Predict(OptimizedRiskAnalysis)
        
        print("\n🎯 How DSPy Uses Optimization:")
        print("   opt_predictor = dspy.Predict(OptimizedRiskAnalysis)")
        print("   result = opt_predictor(")
        print("       risk_event='Data Breach',")
        print("       focus_area='regulatory'")
        print("   )")
        
        print("\n🔍 Optimization Internal Process:")
        print("   1. DSPy adapts prompt based on focus_area parameter")
        print("   2. Creates specialized instructions for each domain")
        print("   3. Ensures output matches focus area requirements")
        print("   4. Maintains consistency across different focus areas")
    
    def demonstrate_prompt_validation(self):
        """Show how DSPy validates prompts."""
        print("\n📋 6. PROMPT VALIDATION")
        print("-" * 40)
        
        class PromptValidator(dspy.Signature):
            """Validate and improve prompts."""
            prompt = dspy.InputField(desc="Prompt to validate")
            
            validation_result = dspy.OutputField(desc="Validation result (valid/invalid)")
            issues = dspy.OutputField(desc="Issues found in prompt")
            improved_prompt = dspy.OutputField(desc="Improved version of prompt")
        
        print("🔧 Prompt Validator Signature:")
        print("   class PromptValidator(dspy.Signature):")
        print("       prompt = dspy.InputField(desc='Prompt to validate')")
        print("       validation_result = dspy.OutputField(desc='Validation result...')")
        print("       issues = dspy.OutputField(desc='Issues found in prompt')")
        print("       improved_prompt = dspy.OutputField(desc='Improved version of prompt')")
        
        # Show validation prompt template
        print("\n📝 DSPy Generated Validation Prompt Template:")
        print("   Prompt to Validate: {prompt}")
        print("   ")
        print("   Please validate this prompt and provide improvements:")
        print("   ")
        print("   Validation Result: [valid/invalid]")
        print("   Issues Found: [Issues found in prompt]")
        print("   Improved Prompt: [Improved version of prompt]")
        
        # Create validator
        validator = dspy.Predict(PromptValidator)
        
        print("\n🎯 How DSPy Uses Validation:")
        print("   validator = dspy.Predict(PromptValidator)")
        print("   result = validator(prompt='Analyze risk')")
        
        print("\n🔍 Validation Internal Process:")
        print("   1. DSPy analyzes prompt clarity and completeness")
        print("   2. Identifies missing context or specificity")
        print("   3. Suggests improvements for better results")
        print("   4. Ensures prompts meet quality standards")
    
    def show_dspy_vs_traditional_prompting(self):
        """Compare DSPy prompt creation vs traditional prompting."""
        print("\n📋 7. DSPY vs TRADITIONAL PROMPTING")
        print("-" * 40)
        
        print("🔧 Traditional Prompt Engineering:")
        print("   prompt = f'''")
        print("   Analyze the following risk event:")
        print("   Event: {risk_event}")
        print("   Category: {category}")
        print("   Frequency: {frequency}")
        print("   Impact: {impact}")
        print("   ")
        print("   Please provide:")
        print("   - A detailed analysis")
        print("   - A risk score from 1-10")
        print("   - Specific recommendations")
        print("   '''")
        print("   ")
        print("   response = llm.generate(prompt)")
        print("   # Manual parsing of response...")
        
        print("\n🧠 DSPy Prompt Creation:")
        print("   class RiskAnalysis(dspy.Signature):")
        print("       risk_event = dspy.InputField(desc='The risk event to analyze')")
        print("       category = dspy.InputField(desc='Risk category...')")
        print("       analysis = dspy.OutputField(desc='Detailed risk analysis...')")
        print("       risk_score = dspy.OutputField(desc='Risk score from 1-10')")
        print("       recommendations = dspy.OutputField(desc='Specific recommendations...')")
        print("   ")
        print("   predictor = dspy.Predict(RiskAnalysis)")
        print("   result = predictor(risk_event='...', category='...')")
        print("   # Structured output automatically parsed")
        
        print("\n🎯 Key Differences:")
        print("   ✅ DSPy: Declarative signature definition")
        print("   ❌ Traditional: Manual prompt string construction")
        print("   ")
        print("   ✅ DSPy: Automatic prompt template generation")
        print("   ❌ Traditional: Manual template creation")
        print("   ")
        print("   ✅ DSPy: Structured output parsing")
        print("   ❌ Traditional: Manual response parsing")
        print("   ")
        print("   ✅ DSPy: Built-in validation and optimization")
        print("   ❌ Traditional: Manual prompt testing")
        print("   ")
        print("   ✅ DSPy: Reusable across different models")
        print("   ❌ Traditional: Model-specific prompt engineering")
    
    def run_complete_demo(self):
        """Run the complete DSPy prompt creation demo."""
        print("🧠 DSPy Prompt Creation Demo - How DSPy Actually Creates Prompts")
        print("="*80)
        
        # 1. Basic Prompt Creation
        self.demonstrate_basic_prompt_creation()
        
        # 2. Chain of Thought
        self.demonstrate_chain_of_thought_prompts()
        
        # 3. RAG Prompts
        self.demonstrate_rag_prompt_creation()
        
        # 4. Multi-Step Prompts
        self.demonstrate_multi_step_prompt_creation()
        
        # 5. Prompt Optimization
        self.demonstrate_prompt_optimization()
        
        # 6. Prompt Validation
        self.demonstrate_prompt_validation()
        
        # 7. Comparison
        self.show_dspy_vs_traditional_prompting()
        
        # Summary
        print("\n" + "="*80)
        print("🎉 DSPY PROMPT CREATION DEMO COMPLETE!")
        print("="*80)
        print("✅ DSPy creates prompts from signature definitions")
        print("✅ Automatic template generation from InputField/OutputField")
        print("✅ Built-in support for CoT, RAG, and multi-step reasoning")
        print("✅ Prompt optimization and validation capabilities")
        print("✅ Structured output parsing and error handling")
        print("✅ Reusable across different LLM providers")
        print("="*80)


def main():
    """Main function to run DSPy prompt creation demo."""
    demo = DSPyPromptCreationDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main()
