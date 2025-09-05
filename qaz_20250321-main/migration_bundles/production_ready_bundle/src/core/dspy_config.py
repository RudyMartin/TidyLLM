"""
DSPy Configuration for VectorQA Sage
Handles DSPy parameter configuration including GPT-5 reasoning and verbosity controls
Updated for DSPy 3.0.1 with new LM architecture
"""

import dspy
from typing import Dict, Any, Optional, List
from .config import CONFIG

class DSPyConfig:
    """Manages DSPy configuration and parameter settings."""
    
    def __init__(self, provider: str = None):
        """
        Initialize DSPy configuration.
        
        Args:
            provider: LLM provider to use (openai, anthropic, cohere, etc.)
        """
        self.provider = provider or "openai"
        self.config = CONFIG["llm_models"].get(self.provider, {})
        self._setup_dspy()
    
    def _setup_dspy(self):
        """Setup DSPy with the configured provider using new DSPy 3.0.1 LM class."""
        try:
            # Get model name from config
            model_name = self.config.get("model", "gpt-3.5-turbo")
            max_tokens = self.config.get("max_tokens", 1000)
            temperature = self.config.get("temperature", 0.1)
            
            # Create model string in format "provider/model_name"
            if self.provider == "openai":
                model_string = f"openai/{model_name}"
            elif self.provider == "anthropic":
                model_string = f"anthropic/{model_name}"
            elif self.provider == "cohere":
                model_string = f"cohere/{model_name}"
            elif self.provider == "google":
                model_string = f"google/{model_name}"
            else:
                # Default to OpenAI
                model_string = f"openai/gpt-3.5-turbo"
            
            # Configure DSPy with new LM class
            dspy.configure(lm=dspy.LM(
                model=model_string,
                model_type="chat",
                temperature=temperature,
                max_tokens=max_tokens
            ))
                
        except Exception as e:
            print(f"Warning: Could not configure DSPy with {self.provider}: {e}")
            # Fallback to OpenAI
            dspy.configure(lm=dspy.LM(
                model="openai/gpt-3.5-turbo",
                model_type="chat",
                temperature=0.1,
                max_tokens=1000
            ))
    
    def configure_gpt5_parameters(self, 
                                 reasoning_effort: str = None,
                                 text_verbosity: str = None,
                                 temperature: float = None,
                                 max_tokens: int = None):
        """
        Configure GPT-5 specific parameters using DSPy 3.0.1 LM class.
        
        Args:
            reasoning_effort: "low", "medium", "high" - controls reasoning depth
            text_verbosity: "low", "medium", "high" - controls response length
            temperature: Creativity level (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
        """
        if self.provider == "openai":
            # Prepare parameters for GPT-5
            model_string = "openai/gpt-5"
            max_tokens = max_tokens or self.config.get("max_tokens", 1000)
            temperature = temperature or self.config.get("temperature", 0.1)
            
            # Create kwargs for GPT-5 specific parameters
            kwargs = {
                "model": model_string,
                "model_type": "chat",
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add GPT-5 specific parameters if provided
            if reasoning_effort:
                kwargs["reasoning"] = {"effort": reasoning_effort}
            if text_verbosity:
                kwargs["text"] = {"verbosity": text_verbosity}
            
            dspy.configure(lm=dspy.LM(**kwargs))
    
    def configure_parameters(self, 
                           temperature: float = None,
                           max_tokens: int = None,
                           model: str = None):
        """
        Configure general DSPy parameters using DSPy 3.0.1 LM class.
        
        Args:
            temperature: Creativity level (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            model: Specific model to use
        """
        # Get model name
        model_name = model or self.config.get("model", "gpt-3.5-turbo")
        max_tokens = max_tokens or self.config.get("max_tokens", 1000)
        temperature = temperature or self.config.get("temperature", 0.1)
        
        # Create model string
        model_string = f"{self.provider}/{model_name}"
        
        dspy.configure(lm=dspy.LM(
            model=model_string,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens
        ))
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current DSPy configuration."""
        return {
            "provider": self.provider,
            "model": self.config.get("model"),
            "max_tokens": self.config.get("max_tokens"),
            "temperature": self.config.get("temperature")
        }
    
    def test_configuration(self) -> bool:
        """Test if the current DSPy configuration is working."""
        try:
            # Create a simple test signature
            class TestSignature(dspy.Signature):
                input = dspy.InputField()
                output = dspy.OutputField()
            
            # Test the configuration
            predictor = dspy.Predict(TestSignature)
            result = predictor(input="Say 'Hello' in one word.")
            
            return True
        except Exception as e:
            print(f"DSPy configuration test failed: {e}")
            return False

# Global DSPy configuration instance
dspy_config = DSPyConfig()

def configure_dspy_for_provider(provider: str, 
                               reasoning_effort: str = None,
                               text_verbosity: str = None,
                               temperature: float = None,
                               max_tokens: int = None):
    """
    Configure DSPy for a specific provider with optional GPT-5 parameters.
    
    Args:
        provider: LLM provider to use (openai, anthropic, cohere, google)
        reasoning_effort: For GPT-5, controls reasoning depth
        text_verbosity: For GPT-5, controls response length
        temperature: Creativity level
        max_tokens: Maximum tokens
    """
    global dspy_config
    dspy_config = DSPyConfig(provider)
    
    if provider == "openai" and (reasoning_effort or text_verbosity):
        dspy_config.configure_gpt5_parameters(
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        dspy_config.configure_parameters(
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    return dspy_config.test_configuration()

def create_dspy_signature_with_tools(signature_class, tools: List[str] = None):
    """
    Create a DSPy signature that can use MCP tools.
    
    Args:
        signature_class: The DSPy signature class to enhance
        tools: List of available tool names
    """
    if tools:
        # Add tool usage capability to the signature
        class ToolEnhancedSignature(signature_class):
            tools_used = dspy.OutputField(desc=f"List of tools used: {', '.join(tools)}")
            tool_results = dspy.OutputField(desc="Results from tool calls")
        
        return ToolEnhancedSignature
    else:
        return signature_class
