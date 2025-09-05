"""
LLM Manager for VectorQA Sage
Handles LLM interactions using LiteLLM with OpenAI as default
"""

import os
import logging
from typing import Dict, Any, Optional, List
import litellm
from config.credential_manager import credential_manager

# Import custom Gemini client

# Robust import setup
import sys
from pathlib import Path
_src_dir = Path(__file__).parent
while _src_dir.name != "src" and _src_dir.parent != _src_dir:
    _src_dir = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

try:
    from core.gemini_client import GeminiClient
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
from .config import CONFIG

logger = logging.getLogger(__name__)

class LLMManager:
    """Manages LLM interactions for VectorQA Sage."""
    
    def __init__(self, provider: str = None):
        """
        Initialize LLM manager.
        
        Args:
            provider: LLM provider to use (default: from CONFIG)
        """
        self.provider = provider or CONFIG.get("default_llm", "openai")
        self.config = CONFIG["llm_models"].get(self.provider, {})
        
        # Initialize custom Gemini client if needed
        if self.provider == "google" and GEMINI_AVAILABLE:
            google_config = credential_manager.get_google_config()
            if google_config["api_key"]:
                self.gemini_client = GeminiClient(google_config["api_key"])
                logger.info("Custom Gemini client initialized")
            else:
                self.gemini_client = None
                logger.warning("No Google API key found for Gemini")
        else:
            self.gemini_client = None
        
        self._setup_litellm()
        
    def _setup_litellm(self):
        """Setup LiteLLM with credentials."""
        try:
            # Set API keys from credential manager
            openai_config = credential_manager.get_openai_config()
            if openai_config["api_key"]:
                os.environ["OPENAI_API_KEY"] = openai_config["api_key"]
                logger.info("OpenAI API key configured")
            
            anthropic_config = credential_manager.get_anthropic_config()
            if anthropic_config["api_key"]:
                os.environ["ANTHROPIC_API_KEY"] = anthropic_config["api_key"]
                logger.info("Anthropic API key configured")
            
            cohere_config = credential_manager.get_cohere_config()
            if cohere_config["api_key"]:
                os.environ["COHERE_API_KEY"] = cohere_config["api_key"]
                logger.info("Cohere API key configured")
            
            google_config = credential_manager.get_google_config()
            if google_config["api_key"]:
                os.environ["GOOGLE_API_KEY"] = google_config["api_key"]
                logger.info("Google Gemini API key configured")
            
            # Configure Hugging Face with PRO token (preferred fallback)
            huggingface_config = credential_manager.get_huggingface_config()
            if huggingface_config["api_key"]:
                os.environ["HUGGINGFACE_API_KEY"] = huggingface_config["api_key"]
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_config["api_key"]
                logger.info("Hugging Face PRO API key configured (preferred fallback)")
            
            # Configure LiteLLM
            litellm.set_verbose = False
            logger.info(f"LLM Manager initialized with provider: {self.provider}")
            
        except Exception as e:
            logger.error(f"Error setting up LiteLLM: {e}")
    
    def generate_response(self, 
                         prompt: str, 
                         system_message: str = None,
                         model: str = None,
                         max_tokens: int = None, 
                         temperature: float = None,
                         reasoning_effort: str = None,
                         text_verbosity: str = None) -> str:
        """
        Generate a response using the configured LLM provider.
        
        Args:
            prompt: The user prompt
            system_message: Optional system message
            model: Specific model to use
            max_tokens: Maximum tokens to generate
            temperature: Creativity level (0.0 to 1.0)
            reasoning_effort: For GPT-5, controls reasoning depth ("low", "medium", "high")
            text_verbosity: For GPT-5, controls response length ("low", "medium", "high")
            
        Returns:
            Generated response text
        """
        try:
            # Use custom Gemini client if available
            if self.provider == "google" and self.gemini_client:
                return self.gemini_client.generate_content(
                    prompt=prompt,
                    system_message=system_message,
                    max_tokens=max_tokens or self.config.get("max_tokens", 1000),
                    temperature=temperature or self.config.get("temperature", 0.1)
                )
            
            # Use provided parameters or defaults
            model = model or self.config.get("model", "gpt-4")
            max_tokens = max_tokens or self.config.get("max_tokens", 2000)
            temperature = temperature or self.config.get("temperature", 0.1)
            
            # Prepare LiteLLM parameters
            litellm_params = {
                "model": model,
                "messages": self._prepare_messages(prompt, system_message),
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Add GPT-5 specific parameters if available
            if model.startswith("gpt-5") and reasoning_effort:
                litellm_params["reasoning"] = {"effort": reasoning_effort}
            if model.startswith("gpt-5") and text_verbosity:
                litellm_params["text"] = {"verbosity": text_verbosity}
            
            response = litellm.completion(**litellm_params)
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Handle specific error types
            if "AuthenticationError" in error_type:
                if "invalid api token" in error_msg.lower() or "incorrect api key" in error_msg.lower():
                    logger.error(f"Invalid API key for {self.provider}")
                    return f"Authentication failed: Invalid API key for {self.provider}"
                elif "api_key client option must be set" in error_msg:
                    logger.error(f"No API key configured for {self.provider}")
                    return f"Authentication failed: No API key configured for {self.provider}"
                else:
                    logger.error(f"Authentication error for {self.provider}: {error_msg}")
                    return f"Authentication error: {error_msg}"
            
            elif "RateLimitError" in error_type:
                logger.error(f"Rate limit exceeded for {self.provider}")
                return f"Rate limit exceeded for {self.provider}. Please try again later."
            
            elif "BadRequestError" in error_type:
                if "provider not provided" in error_msg.lower():
                    logger.error(f"Provider configuration error for {self.provider}")
                    return f"Configuration error: Invalid provider setup for {self.provider}"
                else:
                    logger.error(f"Bad request for {self.provider}: {error_msg}")
                    return f"Request error: {error_msg}"
            
            else:
                logger.error(f"Error generating response: {e}")
                # Try fallback model if available
                if self.config.get("fallback_model") and model != self.config["fallback_model"]:
                    logger.info(f"Trying fallback model: {self.config['fallback_model']}")
                    return self.generate_response(
                        prompt, system_message, 
                        self.config["fallback_model"], 
                        max_tokens, temperature
                    )
                
                # Try fallback providers if available
                fallback_result = self._try_fallback_providers(prompt, system_message, max_tokens, temperature)
                if fallback_result:
                    return fallback_result
                
                return f"Error generating response: {str(e)}"
    
    def _try_fallback_providers(self, prompt: str, system_message: str = None, 
                               max_tokens: int = None, temperature: float = None) -> str:
        """
        Try fallback providers when primary provider fails.
        
        Args:
            prompt: User prompt
            system_message: System message
            max_tokens: Maximum tokens
            temperature: Temperature
            
        Returns:
            Response from fallback provider or None if all fail
        """
        fallback_order = CONFIG.get("fallback_order", ["cohere", "google", "huggingface", "openai", "anthropic"])
        current_provider = self.provider
        
        for fallback_provider in fallback_order:
            if fallback_provider == current_provider:
                continue  # Skip current provider
                
            try:
                logger.info(f"Trying fallback provider: {fallback_provider}")
                fallback_llm = LLMManager(provider=fallback_provider)
                response = fallback_llm.generate_response(
                    prompt, system_message, max_tokens=max_tokens, temperature=temperature
                )
                
                # Check if response is not an error
                if not response.startswith("Error"):
                    logger.info(f"Successfully used fallback provider: {fallback_provider}")
                    return f"[{fallback_provider.upper()}] {response}"
                    
            except Exception as e:
                logger.warning(f"Fallback provider {fallback_provider} failed: {e}")
                continue
        
        return None  # All fallbacks failed
    
    def validate_qa_response(self,
                           topic: str, 
                           report_chunk: str, 
                           retrieved_context: str) -> str:
        """
        Validate if a report chunk is consistent with retrieved context.
        
        Args:
            topic: The topic being evaluated
            report_chunk: The chunk of text from the report
            retrieved_context: Additional reference context
            
        Returns:
            Validation result: "Correct", "Missing Info", or "Inconsistent"
        """
        system_message = """You are a validation expert. Your task is to determine if a report chunk is consistent with the provided context and topic.

Evaluation criteria:
- "Correct": The report chunk is accurate, complete, and consistent with the context
- "Missing Info": The report chunk is accurate but incomplete or missing important details
- "Inconsistent": The report chunk contains errors or contradicts the context

Respond with ONLY one of: Correct, Missing Info, Inconsistent"""

        prompt = f"""Topic: {topic}

Report Chunk:
{report_chunk}

Retrieved Context:
{retrieved_context}

Based on the above, is the report chunk Correct, Missing Info, or Inconsistent?"""

        response = self.generate_response(prompt, system_message, temperature=0.0)
        
        # Clean up response
        response = response.strip().lower()
        if "correct" in response:
            return "Correct"
        elif "missing" in response or "incomplete" in response:
            return "Missing Info"
        else:
            return "Inconsistent"
    
    def answer_question(self, 
                       question: str, 
                       context: str,
                       additional_info: str = None) -> str:
        """
        Answer a question based on provided context.
        
        Args:
            question: The question to answer
            context: Relevant context from documents
            additional_info: Any additional information
            
        Returns:
            Answer to the question
        """
        system_message = """You are a helpful AI assistant that answers questions based on provided context. 
- Only use information from the provided context
- If the context doesn't contain enough information, say so
- Be accurate and concise
- Cite specific parts of the context when relevant"""

        prompt = f"""Context:
{context}

{f"Additional Information: {additional_info}" if additional_info else ""}

Question: {question}

Answer:"""

        return self.generate_response(prompt, system_message)
    
    def summarize_document(self, 
                          document_text: str, 
                          focus_areas: List[str] = None) -> str:
        """
        Summarize a document with optional focus areas.
        
        Args:
            document_text: The document to summarize
            focus_areas: Specific areas to focus on in summary
            
        Returns:
            Document summary
        """
        system_message = """You are an expert document summarizer. Create a clear, concise summary that captures the key points and main ideas."""

        focus_prompt = ""
        if focus_areas:
            focus_prompt = f"\n\nPlease focus on these areas: {', '.join(focus_areas)}"

        prompt = f"""Please summarize the following document:{focus_prompt}

Document:
{document_text}

Summary:"""

        return self.generate_response(prompt, system_message)
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get available models and their configurations."""
        return {
            "current_provider": self.provider,
            "current_model": self.config.get("model"),
            "fallback_model": self.config.get("fallback_model"),
            "available_providers": list(CONFIG["llm_models"].keys()),
            "provider_configs": CONFIG["llm_models"]
        }

# Global LLM manager instance
llm_manager = LLMManager()
