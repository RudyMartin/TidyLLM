"""
DSPy Service
============

Standalone service for DSPy prompt engineering and signature optimization.

Features:
- Direct DSPy usage for prompt optimization
- MLflow integration for experiment tracking
- Signature engineering and modules
- RAG enhancement through DSPy patterns
- Custom question-based RAG design

Usage:
    from tidyllm.services.dspy_service import DSPyService

    dspy_service = DSPyService()
    result = dspy_service.optimize_prompt("original prompt")
    enhanced_query = dspy_service.enhance_rag_query("user query")
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# DSPy Framework
try:
    import dspy
    DSPY_FRAMEWORK_AVAILABLE = True
except ImportError:
    DSPY_FRAMEWORK_AVAILABLE = False
    print("⚠️ DSPy framework not available")

# MLflow Integration
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class CorporateDSPyLM(dspy.BaseLM if DSPY_FRAMEWORK_AVAILABLE else object):
    """DSPy Language Model wrapper for CorporateLLMAdapter - NO OpenAI calls."""

    def __init__(self, model_name: str = "claude-3-sonnet", temperature: float = 0.7):
        if DSPY_FRAMEWORK_AVAILABLE:
            super().__init__(model=model_name)
        self.model_name = model_name
        self.temperature = temperature
        self.adapter = None
        self._initialize_adapter()

    def _initialize_adapter(self):
        """Initialize CorporateLLMGateway."""
        try:
            from tidyllm.gateways import CorporateLLMGateway
            self.adapter = CorporateLLMGateway()
            logger.info("CorporateLLMGateway initialized for DSPy")
        except ImportError as e:
            logger.error(f"Failed to import CorporateLLMGateway: {e}")
            raise
        except Exception as e:
            # Log warning but continue - USM might not be fully configured
            logger.warning(f"CorporateLLMGateway initialization had issues: {e}")
            # Still try to create the gateway - it might work for some operations
            try:
                from tidyllm.gateways import CorporateLLMGateway
                self.adapter = CorporateLLMGateway()
            except Exception as e2:
                logger.error(f"Failed to create CorporateLLMGateway: {e2}")
                raise

    def __call__(self, prompt=None, messages=None, **kwargs):
        """Process prompt through CorporateLLMGateway (DSPy interface)."""
        if not self.adapter:
            raise RuntimeError("CorporateLLMGateway not available")

        try:
            from tidyllm.gateways import LLMRequest

            # Debug: Log what DSPy is sending us
            logger.info(f"DSPy call - prompt: {prompt}")
            logger.info(f"DSPy call - messages: {messages}")
            logger.info(f"DSPy call - kwargs: {kwargs}")

            # Handle DSPy message format properly
            if messages and isinstance(messages, list):
                # DSPy sends structured messages - combine system + user
                prompt_parts = []
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get('role', '')
                        content = msg.get('content', '')
                        if role == 'system':
                            prompt_parts.append(f"System: {content}")
                        elif role == 'user':
                            prompt_parts.append(f"User: {content}")
                        else:
                            prompt_parts.append(content)
                    else:
                        prompt_parts.append(str(msg))
                prompt_text = "\n\n".join(prompt_parts)
            else:
                prompt_text = str(prompt) if prompt else ""

            # Create LLM request
            request = LLMRequest(
                prompt=prompt_text,
                model_id=self.model_name,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', 4000),
                user_id=kwargs.get('user_id', 'dspy_user'),
                audit_reason='dspy_processing'
            )

            # Process through corporate adapter
            response = self.adapter.process_request(request)

            if response.success:
                return [response.content]  # DSPy expects list format
            else:
                # No mock mode - raise proper error
                raise RuntimeError(f"CorporateLLMGateway failed: {response.error}")

        except Exception as e:
            logger.error(f"CorporateDSPyLM processing failed: {e}")
            raise

    def generate(self, prompt, **kwargs):
        """Generate response (alternative DSPy interface)."""
        results = self(prompt, **kwargs)
        return results[0] if results else ""


    def __repr__(self):
        return f"CorporateDSPyLM(model={self.model_name}, temp={self.temperature})"


class DSPyService:
    """
    Standalone DSPy service for prompt engineering and signature optimization.

    Features:
    - Direct DSPy usage for prompt optimization
    - MLflow integration for experiment tracking
    - Signature engineering and modules
    - RAG enhancement through DSPy patterns
    - Custom question-based RAG design
    """

    def __init__(self, auto_configure: bool = True):
        """Initialize DSPy Service."""
        if not DSPY_FRAMEWORK_AVAILABLE:
            raise ImportError("DSPy framework required for DSPy Service")

        # Configuration
        self.current_lm = None
        self.signatures_cache = {}
        self.optimizers_cache = {}
        self.mlflow_enabled = MLFLOW_AVAILABLE

        # Initialize signatures
        self._initialize_rag_signatures()

        # Auto-configure if requested
        if auto_configure:
            self.configure_lm()

        logger.info("DSPy Service initialized")

    def _initialize_rag_signatures(self):
        """Initialize common DSPy signatures using standard patterns."""

        # Proper DSPy signatures based on documentation
        # Note: ChainOfThought automatically adds reasoning field

        # Define signature classes for better control
        class BasicQA(dspy.Signature):
            """Answer questions with step-by-step reasoning."""
            question = dspy.InputField()
            answer = dspy.OutputField(desc="concise answer")

        class ChatResponse(dspy.Signature):
            """Generate helpful chat responses."""
            message = dspy.InputField()
            response = dspy.OutputField(desc="helpful response")

        # Store both string and class signatures
        self.signatures = {
            # Use signature classes for better control
            'cot': BasicQA,
            'chat': ChatResponse,

            # Simple string signatures for others
            'query_enhancement': "query -> enhanced_query",
            'context_qa': "context, question -> answer",
            'summarize': "document -> summary",
            'optimize_prompt': "prompt -> optimized_prompt"
        }

        # Create DSPy modules with ChainOfThought for reasoning
        self.modules = {
            'cot': dspy.ChainOfThought(self.signatures['cot']),
            'chat': dspy.ChainOfThought(self.signatures['chat']),
            'query_enhancement': dspy.ChainOfThought(self.signatures['query_enhancement']),
            'context_qa': dspy.ChainOfThought(self.signatures['context_qa']),
            'summarize': dspy.ChainOfThought(self.signatures['summarize']),
            'optimize_prompt': dspy.ChainOfThought(self.signatures['optimize_prompt'])
        }

    # ==================== CORE DSPy OPERATIONS ====================

    def optimize_prompt(self, original_prompt: str, domain: str = "general",
                       optimization_goals: str = "clarity,specificity,performance",
                       use_mlflow: bool = False) -> Dict[str, Any]:
        """
        Optimize a prompt using DSPy Chain of Thought.

        Args:
            original_prompt: The prompt to optimize
            domain: Domain context
            optimization_goals: Comma-separated optimization goals
            use_mlflow: Whether to log to MLflow

        Returns:
            Dict with optimized prompt and rationale
        """
        if not self.current_lm:
            return {
                "success": False,
                "error": "No LM configured for DSPy. Call configure_lm() first."
            }

        try:
            # Start MLflow run if enabled
            if use_mlflow and self.mlflow_enabled:
                mlflow.start_run()
                mlflow.log_param("original_prompt", original_prompt)

            # Use standard DSPy ChainOfThought module
            optimizer = self.modules['optimize_prompt']

            # Run optimization using DSPy
            result = optimizer(prompt=original_prompt)

            response = {
                "success": True,
                "original_prompt": original_prompt,
                "optimized_prompt": result.optimized_prompt,
                "rationale": getattr(result, 'rationale', f"DSPy Chain of Thought optimization for {domain}"),
                "domain": domain,
                "goals": optimization_goals,
                "timestamp": datetime.now().isoformat()
            }

            # Log to MLflow if enabled
            if use_mlflow and self.mlflow_enabled:
                mlflow.log_param("optimized_prompt", result.optimized_prompt)
                mlflow.log_metric("optimization_success", 1.0)
                mlflow.end_run()

            return response

        except Exception as e:
            logger.error(f"Prompt optimization failed: {e}")

            if use_mlflow and self.mlflow_enabled:
                mlflow.log_metric("optimization_success", 0.0)
                mlflow.log_param("error", str(e))
                mlflow.end_run()

            return {
                "success": False,
                "error": str(e),
                "original_prompt": original_prompt
            }

    def chat_with_cot(self, message: str, use_mlflow: bool = False) -> Dict[str, Any]:
        """
        Chat with Chain of Thought reasoning using DSPy.

        Args:
            message: User message
            use_mlflow: Whether to log to MLflow

        Returns:
            Dict with response and reasoning
        """
        if not self.current_lm:
            return {
                "success": False,
                "error": "No LM configured for DSPy. Call configure_lm() first."
            }

        try:
            # Use standard DSPy ChainOfThought for chat
            chat_module = self.modules['cot']

            # Run chat with reasoning
            result = chat_module(question=message)

            # Debug: Check what DSPy actually returned
            logger.info(f"DSPy result type: {type(result)}")
            logger.info(f"DSPy result attributes: {dir(result)}")
            logger.info(f"DSPy result string: {str(result)}")

            # Extract response content from DSPy prediction objects
            if hasattr(result, 'answer'):
                response_content = result.answer
            elif hasattr(result, 'response'):
                response_content = result.response
            else:
                response_content = str(result)

            # Extract reasoning if available
            if hasattr(result, 'rationale'):
                reasoning_content = result.rationale
            elif hasattr(result, 'reasoning'):
                reasoning_content = result.reasoning
            else:
                reasoning_content = "DSPy Chain of Thought reasoning"

            response = {
                "success": True,
                "message": message,
                "response": response_content,
                "reasoning": reasoning_content,
                "timestamp": datetime.now().isoformat()
            }

            if use_mlflow and self.mlflow_enabled:
                mlflow.start_run()
                mlflow.log_param("message", message)
                mlflow.log_param("response", result.answer)
                mlflow.log_metric("chat_success", 1.0)
                mlflow.end_run()

            return response

        except Exception as e:
            logger.error(f"DSPy chat failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": message
            }

    def enhance_rag_query(self, user_query: str, domain: str = "general",
                         use_mlflow: bool = False) -> Dict[str, Any]:
        """
        Enhance a user query for better RAG retrieval.

        Args:
            user_query: Original user query
            domain: Domain context
            use_mlflow: Whether to log to MLflow

        Returns:
            Dict with enhanced query and search strategy
        """
        if not self.current_lm:
            return {
                "success": False,
                "error": "No LM configured for DSPy"
            }

        try:
            if use_mlflow and self.mlflow_enabled:
                mlflow.start_run()
                mlflow.log_param("user_query", user_query)
                mlflow.log_param("domain", domain)

            enhancer = dspy.ChainOfThought(self.signatures['query_enhancement'])

            result = enhancer(
                user_query=user_query,
                domain=domain
            )

            response = {
                "success": True,
                "original_query": user_query,
                "enhanced_query": result.enhanced_query,
                "search_strategy": result.search_strategy,
                "domain": domain,
                "timestamp": datetime.now().isoformat()
            }

            if use_mlflow and self.mlflow_enabled:
                mlflow.log_param("enhanced_query", result.enhanced_query)
                mlflow.log_param("search_strategy", result.search_strategy)
                mlflow.log_metric("enhancement_success", 1.0)
                mlflow.end_run()

            return response

        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")

            if use_mlflow and self.mlflow_enabled:
                mlflow.log_metric("enhancement_success", 0.0)
                mlflow.log_param("error", str(e))
                mlflow.end_run()

            return {
                "success": False,
                "error": str(e),
                "original_query": user_query
            }

    def design_rag_system(self, requirements: str, domain: str = "general",
                         use_mlflow: bool = False) -> Dict[str, Any]:
        """
        Design a RAG system based on requirements using DSPy.

        Args:
            requirements: User requirements and constraints
            domain: Domain context
            use_mlflow: Whether to log to MLflow

        Returns:
            Dict with recommended configuration and rationale
        """
        if not self.current_lm:
            return {
                "success": False,
                "error": "No LM configured for DSPy"
            }

        try:
            if use_mlflow and self.mlflow_enabled:
                mlflow.start_run()
                mlflow.log_param("requirements", requirements)
                mlflow.log_param("domain", domain)

            designer = dspy.ChainOfThought(self.signatures['rag_design'])

            result = designer(
                requirements=requirements,
                domain=domain
            )

            response = {
                "success": True,
                "requirements": requirements,
                "recommended_config": result.recommended_config,
                "rationale": result.rationale,
                "domain": domain,
                "timestamp": datetime.now().isoformat()
            }

            if use_mlflow and self.mlflow_enabled:
                mlflow.log_param("recommended_config", result.recommended_config)
                mlflow.log_param("rationale", result.rationale)
                mlflow.log_metric("design_success", 1.0)
                mlflow.end_run()

            return response

        except Exception as e:
            logger.error(f"RAG design failed: {e}")

            if use_mlflow and self.mlflow_enabled:
                mlflow.log_metric("design_success", 0.0)
                mlflow.log_param("error", str(e))
                mlflow.end_run()

            return {
                "success": False,
                "error": str(e),
                "requirements": requirements
            }

    def engineer_signature(self, task_description: str, input_fields: str,
                          output_fields: str, use_mlflow: bool = False) -> Dict[str, Any]:
        """
        Engineer a new DSPy signature for a specific task.

        Args:
            task_description: Description of the task
            input_fields: Required input fields
            output_fields: Desired output fields
            use_mlflow: Whether to log to MLflow

        Returns:
            Dict with generated signature code and usage example
        """
        if not self.current_lm:
            return {
                "success": False,
                "error": "No LM configured for DSPy"
            }

        try:
            if use_mlflow and self.mlflow_enabled:
                mlflow.start_run()
                mlflow.log_param("task_description", task_description)
                mlflow.log_param("input_fields", input_fields)
                mlflow.log_param("output_fields", output_fields)

            engineer = dspy.ChainOfThought(self.signatures['signature_engineering'])

            result = engineer(
                task_description=task_description,
                input_fields=input_fields,
                output_fields=output_fields
            )

            response = {
                "success": True,
                "task_description": task_description,
                "signature_code": result.signature_code,
                "usage_example": result.usage_example,
                "timestamp": datetime.now().isoformat()
            }

            if use_mlflow and self.mlflow_enabled:
                mlflow.log_param("signature_code", result.signature_code)
                mlflow.log_param("usage_example", result.usage_example)
                mlflow.log_metric("engineering_success", 1.0)
                mlflow.end_run()

            return response

        except Exception as e:
            logger.error(f"Signature engineering failed: {e}")

            if use_mlflow and self.mlflow_enabled:
                mlflow.log_metric("engineering_success", 0.0)
                mlflow.log_param("error", str(e))
                mlflow.end_run()

            return {
                "success": False,
                "error": str(e),
                "task_description": task_description
            }

    # ==================== CONFIGURATION ====================

    def configure_lm(self, model_name: str = "claude-3-sonnet"):
        """Configure the language model for DSPy using our CorporateLLMAdapter - NO OpenAI."""
        try:
            # ONLY use our CorporateLLMAdapter - NO external API calls
            corporate_lm = CorporateDSPyLM(model_name=model_name)
            dspy.configure(lm=corporate_lm)
            self.current_lm = f"corporate_adapter:{model_name}"
            logger.info(f"DSPy configured with CorporateLLMAdapter: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to configure DSPy LM: {e}")
            return False

    def enable_mlflow_logging(self, enabled: bool = True):
        """Enable or disable MLflow logging."""
        if enabled and not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available, cannot enable logging")
            return False

        self.mlflow_enabled = enabled
        logger.info(f"MLflow logging {'enabled' if enabled else 'disabled'}")
        return True

    # ==================== STATUS & HEALTH ====================

    def generate_action_signature(self, action_def: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate DSPy signature from action definition.

        Args:
            action_def: Action definition from actions_spec.json

        Returns:
            Dict with generated signature and module code
        """
        if not DSPY_FRAMEWORK_AVAILABLE:
            return {
                "success": False,
                "error": "DSPy framework not available"
            }

        try:
            action_type = action_def.get('type', 'unknown')
            title = action_def.get('title', 'Unknown Action')
            description = action_def.get('description', '')
            requires = action_def.get('requires', [])
            produces = action_def.get('produces', [])
            params = action_def.get('params', {})

            # Generate signature class dynamically
            signature_name = f"{action_type.title().replace('_', '')}Signature"

            # Build input fields from requires and params
            input_fields = []
            for req in requires:
                input_fields.append(f"{req}: dspy.InputField(desc='Required input: {req}')")

            for param_name, param_spec in params.items():
                if param_spec.get('required', False):
                    input_fields.append(f"{param_name}: dspy.InputField(desc='{param_spec.get('description', param_name)}')")

            # Build output fields from produces
            output_fields = []
            for prod in produces:
                output_fields.append(f"{prod}: dspy.OutputField(desc='Generated output: {prod}')")

            # Create signature class code
            signature_code = f"""
class {signature_name}(dspy.Signature):
    \"\"\"{title}: {description}\"\"\"

    # Input fields
    {chr(10).join(input_fields) if input_fields else 'pass'}

    # Output fields
    {chr(10).join(output_fields) if output_fields else 'pass'}
"""

            # Create module code for execution
            module_code = f"""
class {action_type.title().replace('_', '')}Module(dspy.Module):
    def __init__(self):
        super().__init__()
        self.processor = dspy.ChainOfThought({signature_name})

    def forward(self, **inputs):
        result = self.processor(**inputs)
        return result
"""

            # Store signature in cache
            cache_key = f"action_{action_type}"
            self.signatures_cache[cache_key] = {
                "signature_code": signature_code,
                "module_code": module_code,
                "action_def": action_def
            }

            return {
                "success": True,
                "action_type": action_type,
                "signature_name": signature_name,
                "signature_code": signature_code,
                "module_code": module_code,
                "cached_as": cache_key,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Action signature generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "action_type": action_def.get('type', 'unknown')
            }

    def generate_action_chain_module(self, action_types: List[str]) -> Dict[str, Any]:
        """
        Generate a complete DSPy module for an action chain.

        Args:
            action_types: List of action types to chain

        Returns:
            Dict with generated module code and execution plan
        """
        if not DSPY_FRAMEWORK_AVAILABLE:
            return {
                "success": False,
                "error": "DSPy framework not available"
            }

        try:
            # Load actions loader to get definitions
            from domain.services.actions_loader_service import get_actions_loader
            loader = get_actions_loader()

            # Validate action sequence
            is_valid, errors = loader.validate_action_sequence(action_types)
            if not is_valid:
                return {
                    "success": False,
                    "errors": errors,
                    "action_types": action_types
                }

            # Generate signatures for each action
            signatures = []
            for action_type in action_types:
                action = loader.get_action(action_type)
                if action:
                    # Convert dataclass to dict
                    action_dict = {
                        'type': action.action_type,
                        'title': action.title,
                        'description': action.description,
                        'requires': action.requires,
                        'produces': action.produces,
                        'params': action.params,
                        'inputs': action.inputs,
                        'output_schema': action.output_schema
                    }
                    sig_result = self.generate_action_signature(action_dict)
                    if sig_result.get("success"):
                        signatures.append(sig_result)

            # Create chained module
            chain_module_code = f"""
class ActionChainModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # Initialize action processors
        {chr(10).join([f"        self.{sig['action_type']}_processor = dspy.ChainOfThought({sig['signature_name']})" for sig in signatures])}

    def forward(self, **initial_inputs):
        artifacts = {{}}
        artifacts.update(initial_inputs)

        # Execute action chain
        {chr(10).join([f"        # Step {i+1}: {sig['action_type']}" + chr(10) + f"        result_{sig['action_type']} = self.{sig['action_type']}_processor(**artifacts)" + chr(10) + f"        artifacts.update(result_{sig['action_type']}.to_dict())" for i, sig in enumerate(signatures)])}

        return artifacts
"""

            return {
                "success": True,
                "action_chain": action_types,
                "module_code": chain_module_code,
                "signatures": [s["signature_name"] for s in signatures],
                "execution_plan": {
                    "steps": len(action_types),
                    "artifact_flow": loader.export_as_dspy_config(action_types)
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Action chain module generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "action_types": action_types
            }

    def get_status(self) -> Dict[str, Any]:
        """Get DSPy Service status."""
        return {
            "service_name": "DSPy Service",
            "framework_available": DSPY_FRAMEWORK_AVAILABLE,
            "mlflow_available": MLFLOW_AVAILABLE,
            "lm_configured": self.current_lm is not None,
            "current_lm": self.current_lm,
            "mlflow_enabled": self.mlflow_enabled,
            "available_signatures": list(self.signatures.keys()),
            "cache_sizes": {
                "signatures": len(self.signatures_cache),
                "optimizers": len(self.optimizers_cache)
            },
            "timestamp": datetime.now().isoformat()
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            health_status = {
                "healthy": True,
                "framework_available": DSPY_FRAMEWORK_AVAILABLE,
                "lm_configured": self.current_lm is not None,
                "signatures_loaded": len(self.signatures) > 0,
                "mlflow_integration": self.mlflow_enabled and MLFLOW_AVAILABLE
            }

            # Test basic functionality
            if self.current_lm:
                test_result = self.optimize_prompt(
                    "test prompt",
                    domain="test",
                    use_mlflow=False
                )
                health_status["basic_functionality"] = test_result.get("success", False)
            else:
                health_status["basic_functionality"] = False

            health_status["overall_healthy"] = all([
                health_status["framework_available"],
                health_status["lm_configured"],
                health_status["signatures_loaded"]
            ])

            return {
                "success": True,
                "health": health_status,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "healthy": False,
                "timestamp": datetime.now().isoformat()
            }


# ==================== CONVENIENCE FUNCTIONS ====================

# Global service instance for singleton pattern
_global_dspy_service = None

def get_dspy_service(auto_configure: bool = True) -> DSPyService:
    """Get DSPy service instance - singleton pattern for consistent state."""
    global _global_dspy_service
    if _global_dspy_service is None:
        _global_dspy_service = DSPyService(auto_configure=auto_configure)
    return _global_dspy_service


# ==================== MAIN ====================

if __name__ == "__main__":
    print("🧠 TidyLLM DSPy Service")
    print("=" * 30)

    # Test DSPy service functionality
    try:
        dspy_service = DSPyService()

        # Check status
        print("\n📊 Service Status:")
        status = dspy_service.get_status()
        for key, value in status.items():
            if key not in ["cache_sizes", "timestamp"]:
                print(f"✅ {key}: {value}")

        # Health check
        print("\n💓 Health Check:")
        health = dspy_service.health_check()
        if health["success"]:
            health_info = health["health"]
            overall = "✅ Healthy" if health_info["overall_healthy"] else "❌ Unhealthy"
            print(f"Overall: {overall}")

        print("\n✅ DSPy Service test completed!")

    except Exception as e:
        print(f"❌ DSPy Service test failed: {e}")