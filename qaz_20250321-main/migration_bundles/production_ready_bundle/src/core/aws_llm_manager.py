"""
AWS-Only LLM Manager for Security-Sensitive Environments

This module provides LLM functionality restricted to AWS services only,
eliminating external API dependencies for enhanced security.

TODO - Add comprehensive error handling for Bedrock API failures
TODO - Add retry logic with exponential backoff
TODO - Add model availability checking
TODO - Add cost tracking and monitoring
TODO - Add performance metrics collection
"""

import boto3
import json
import logging
from typing import Dict, Any, Optional, List
from .config import CONFIG, bedrock_client

logger = logging.getLogger(__name__)

class SecurityError(Exception):
    """Security violation exception."""
    pass

class AWSSecurityLLMManager:
    """AWS-only LLM manager with security restrictions."""
    
    def __init__(self):
        self.client = bedrock_client
        self.security_config = CONFIG.get("security", {})
        self.aws_only = self.security_config.get("aws_only", True)
        self.audit_logging = self.security_config.get("audit_logging", True)
        
        if not self.aws_only:
            raise SecurityError("AWS-only mode is required for this environment")
        
        logger.info("Initialized AWS-Only LLM Manager")
    
    def validate_aws_only(self):
        """Validate that only AWS services are being used."""
        if not self.aws_only:
            raise SecurityError("External API calls are not allowed in AWS-only mode")
    
    def log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log audit events for security compliance."""
        if self.audit_logging:
            logger.info(f"AUDIT: {event_type} - {json.dumps(details)}")
    
    def invoke_bedrock_model(self, model_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Invoke AWS Bedrock model with security validation."""
        
        # Security validation
        self.validate_aws_only()
        
        # Audit logging
        self.log_audit_event("bedrock_api_call", {
            "model_id": model_id,
            "prompt_length": len(prompt),
            "max_tokens": kwargs.get("max_tokens", 4096)
        })
        
        try:
            # Prepare payload based on model type
            if "claude" in model_id.lower():
                payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": kwargs.get("max_tokens", 4096),
                    "temperature": kwargs.get("temperature", 0.1),
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
            elif "titan" in model_id.lower():
                payload = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": kwargs.get("max_tokens", 4096),
                        "temperature": kwargs.get("temperature", 0.1)
                    }
                }
            else:
                # Default payload format
                payload = {
                    "prompt": prompt,
                    "max_tokens": kwargs.get("max_tokens", 4096),
                    "temperature": kwargs.get("temperature", 0.1)
                }
            
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(payload)
            )
            
            # Parse response based on model type
            response_body = json.loads(response["body"].read())
            
            if "claude" in model_id.lower():
                return {
                    "completion": response_body.get("content", [{}])[0].get("text", ""),
                    "model": model_id,
                    "usage": response_body.get("usage", {})
                }
            elif "titan" in model_id.lower():
                return {
                    "completion": response_body.get("results", [{}])[0].get("outputText", ""),
                    "model": model_id,
                    "usage": response_body.get("usage", {})
                }
            else:
                return {
                    "completion": response_body.get("completion", ""),
                    "model": model_id,
                    "usage": response_body.get("usage", {})
                }
            
        except Exception as e:
            logger.error(f"Bedrock API error for {model_id}: {e}")
            self.log_audit_event("bedrock_api_error", {
                "model_id": model_id,
                "error": str(e)
            })
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embeddings using AWS Titan only."""
        self.validate_aws_only()
        
        model_id = CONFIG.get("bedrock_models", {}).get("embedding", {}).get("primary", "amazon.titan-embed-text-v2:0")
        
        self.log_audit_event("embedding_generation", {
            "model_id": model_id,
            "text_length": len(text)
        })
        
        try:
            payload = {"inputText": text}
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(payload)
            )
            
            result = json.loads(response["body"].read())
            embedding = result.get("embedding", [])
            
            self.log_audit_event("embedding_success", {
                "model_id": model_id,
                "embedding_dimensions": len(embedding)
            })
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            self.log_audit_event("embedding_error", {
                "model_id": model_id,
                "error": str(e)
            })
            raise
    
    def analyze_document(self, content: str) -> Dict[str, Any]:
        """Analyze document using AWS Claude only."""
        self.validate_aws_only()
        
        model_id = CONFIG.get("bedrock_models", {}).get("document_analysis", {}).get("primary", "anthropic.claude-3-sonnet-20240229-v1:0")
        
        prompt = f"""
        Analyze the following document and extract key information:
        
        {content}
        
        Return the analysis in JSON format with the following fields:
        - review_id: The review identifier (format: REVXXXXX)
        - model_type: Type of model being reviewed
        - risk_tier: Risk assessment tier (Low/Medium/High)
        - model_id: Model identifier
        - model_name: Name of the model
        - version: Model version
        - authors: Model authors
        - date: Review date (MM-DD-YYYY format)
        - validation_type: Type of validation
        
        Ensure the response is valid JSON format.
        """
        
        self.log_audit_event("document_analysis_start", {
            "model_id": model_id,
            "content_length": len(content)
        })
        
        try:
            response = self.invoke_bedrock_model(
                model_id, 
                prompt, 
                max_tokens=4096, 
                temperature=0.1
            )
            
            # Try to parse JSON from response
            completion = response.get("completion", "")
            
            # Extract JSON from the response (handle cases where model adds extra text)
            try:
                # Find JSON in the response
                start_idx = completion.find("{")
                end_idx = completion.rfind("}") + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_str = completion[start_idx:end_idx]
                    analysis = json.loads(json_str)
                else:
                    # If no JSON found, create a structured response
                    analysis = {
                        "review_id": "REV00000",
                        "model_type": "Unknown",
                        "risk_tier": "Medium",
                        "model_id": "unknown",
                        "model_name": "Unknown Model",
                        "version": "1.0",
                        "authors": "Unknown",
                        "date": "01-01-2024",
                        "validation_type": "Standard",
                        "raw_response": completion
                    }
                
            except json.JSONDecodeError:
                # If JSON parsing fails, create structured response
                analysis = {
                    "review_id": "REV00000",
                    "model_type": "Unknown",
                    "risk_tier": "Medium",
                    "model_id": "unknown",
                    "model_name": "Unknown Model",
                    "version": "1.0",
                    "authors": "Unknown",
                    "date": "01-01-2024",
                    "validation_type": "Standard",
                    "raw_response": completion,
                    "parse_error": True
                }
            
            self.log_audit_event("document_analysis_success", {
                "model_id": model_id,
                "extracted_fields": list(analysis.keys())
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Document analysis error: {e}")
            self.log_audit_event("document_analysis_error", {
                "model_id": model_id,
                "error": str(e)
            })
            raise
    
    def generate_report(self, analysis_data: Dict[str, Any]) -> str:
        """Generate QA report using AWS Claude only."""
        self.validate_aws_only()
        
        model_id = CONFIG.get("bedrock_models", {}).get("report_generation", {}).get("primary", "anthropic.claude-3-opus-20240229-v1:0")
        
        prompt = f"""
        Generate a comprehensive QA HealthCheck report based on the following analysis:
        
        {json.dumps(analysis_data, indent=2)}
        
        Include the following sections:
        1. Executive Summary
        2. Document Analysis Results
        3. Compliance Checklist
        4. Risk Assessment
        5. Recommendations
        6. Next Steps
        
        Format as a professional report with clear headings and bullet points.
        Make it suitable for business stakeholders.
        """
        
        self.log_audit_event("report_generation_start", {
            "model_id": model_id,
            "analysis_data_keys": list(analysis_data.keys())
        })
        
        try:
            response = self.invoke_bedrock_model(
                model_id, 
                prompt, 
                max_tokens=4096, 
                temperature=0.2
            )
            
            report = response.get("completion", "")
            
            self.log_audit_event("report_generation_success", {
                "model_id": model_id,
                "report_length": len(report)
            })
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            self.log_audit_event("report_generation_error", {
                "model_id": model_id,
                "error": str(e)
            })
            raise
    
    def quick_validation(self, content: str, validation_type: str) -> Dict[str, Any]:
        """Perform quick validation using AWS Claude Haiku."""
        self.validate_aws_only()
        
        model_id = CONFIG.get("bedrock_models", {}).get("quick_processing", {}).get("primary", "anthropic.claude-3-haiku-20240307-v1:0")
        
        prompt = f"""
        Perform a quick validation on the following content for {validation_type}:
        
        {content}
        
        Return a JSON response with:
        - valid: boolean (true/false)
        - issues: list of any issues found
        - confidence: confidence level (0-100)
        - recommendations: list of recommendations
        """
        
        self.log_audit_event("quick_validation_start", {
            "model_id": model_id,
            "validation_type": validation_type,
            "content_length": len(content)
        })
        
        try:
            response = self.invoke_bedrock_model(
                model_id, 
                prompt, 
                max_tokens=2048, 
                temperature=0.1
            )
            
            completion = response.get("completion", "")
            
            # Try to parse JSON response
            try:
                start_idx = completion.find("{")
                end_idx = completion.rfind("}") + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_str = completion[start_idx:end_idx]
                    validation_result = json.loads(json_str)
                else:
                    validation_result = {
                        "valid": False,
                        "issues": ["Could not parse validation response"],
                        "confidence": 0,
                        "recommendations": ["Manual review required"],
                        "raw_response": completion
                    }
                    
            except json.JSONDecodeError:
                validation_result = {
                    "valid": False,
                    "issues": ["JSON parsing error"],
                    "confidence": 0,
                    "recommendations": ["Manual review required"],
                    "raw_response": completion
                }
            
            self.log_audit_event("quick_validation_success", {
                "model_id": model_id,
                "validation_type": validation_type,
                "is_valid": validation_result.get("valid", False)
            })
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Quick validation error: {e}")
            self.log_audit_event("quick_validation_error", {
                "model_id": model_id,
                "validation_type": validation_type,
                "error": str(e)
            })
            raise
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all configured AWS models."""
        self.validate_aws_only()
        
        status = {
            "aws_only_mode": self.aws_only,
            "audit_logging": self.audit_logging,
            "models": {}
        }
        
        bedrock_models = CONFIG.get("bedrock_models", {})
        
        for task_type, model_config in bedrock_models.items():
            primary_model = model_config.get("primary", "unknown")
            fallback_model = model_config.get("fallback", "none")
            
            status["models"][task_type] = {
                "primary": primary_model,
                "fallback": fallback_model,
                "available": True  # Assume available, could add actual checking
            }
        
        return status
