"""
Shared gateway utilities for TidyLLM demos
"""
import streamlit as st
from typing import Dict, Any, Optional
from .utils import load_settings

class GatewayManager:
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = settings or load_settings()
        self.mlflow_client = None
        self.tidyllm_gateway = None
        self.mlflow_connected = False
        self.tidyllm_connected = False
    
    def setup_mlflow_gateway(self) -> bool:
        """Setup MLflow Gateway connection"""
        try:
            import mlflow.gateway as gateway
            
            gateway_uri = self.settings.get('mlflow', {}).get('gateway_uri', 'http://localhost:5000')
            gateway.set_gateway_uri(gateway_uri)
            self.mlflow_client = gateway
            
            # Test connection
            test_uri = gateway.get_gateway_uri()
            self.mlflow_connected = True
            return True
            
        except ImportError:
            st.error("MLflow Gateway not available. Install with: pip install mlflow")
            return False
        except Exception as e:
            st.error(f"MLflow Gateway connection failed: {e}")
            return False
    
    def setup_tidyllm_gateway(self) -> bool:
        """Setup TidyLLM Gateway connection"""
        try:
            from tidyllm_gateway.gateways.llm_gateway import LLMGateway, LLMGatewayConfig
            
            gateway_config = self.settings.get('tidyllm_gateway', {})
            config = LLMGatewayConfig(
                mlflow_gateway_uri=gateway_config.get('mlflow_gateway_uri', 'http://localhost:5000'),
                default_provider=gateway_config.get('default_provider', 'claude'),
                default_model=gateway_config.get('default_model', 'claude-3-5-sonnet')
            )
            
            self.tidyllm_gateway = LLMGateway(config)
            self.tidyllm_connected = True
            return True
            
        except ImportError:
            st.error("TidyLLM Gateway not available. Install with: pip install -e ../tidyllm-gateway")
            return False
        except Exception as e:
            st.error(f"TidyLLM Gateway connection failed: {e}")
            return False
    
    def make_llm_request(self, prompt: str, model: str = None, provider: str = None) -> Optional[Dict[str, Any]]:
        """Make an LLM request through available gateways"""
        try:
            # Try TidyLLM Gateway first
            if self.tidyllm_connected and self.tidyllm_gateway:
                response = self.tidyllm_gateway.chat(
                    messages=[{"role": "user", "content": prompt}],
                    user_id="tidyllm_demos",
                    audit_reason="Demo request",
                    provider=provider or "claude",
                    model=model or "claude-3-5-sonnet",
                    temperature=0.1,
                    max_tokens=200
                )
                
                return {
                    "content": response.content if hasattr(response, 'content') else str(response),
                    "cost": response.cost_usd if hasattr(response, 'cost_usd') else 0.0,
                    "tokens": response.usage.get("total_tokens", 0) if hasattr(response, 'usage') else 0,
                    "gateway": "tidyllm",
                    "success": True
                }
            
            # Fallback to MLflow Gateway
            elif self.mlflow_connected and self.mlflow_client:
                # Note: Full MLflow Gateway API requires additional setup
                st.info("MLflow Gateway connected but full API not implemented yet")
                return None
            
            else:
                st.error("No gateway connections available")
                return None
                
        except Exception as e:
            st.error(f"LLM request failed: {e}")
            return None
    
    def get_connection_status(self) -> Dict[str, bool]:
        """Get connection status for all gateways"""
        return {
            "mlflow_gateway": self.mlflow_connected,
            "tidyllm_gateway": self.tidyllm_connected
        }
    
    def disconnect(self):
        """Disconnect from all gateways"""
        self.mlflow_client = None
        self.tidyllm_gateway = None
        self.mlflow_connected = False
        self.tidyllm_connected = False

def get_gateway_manager() -> GatewayManager:
    """Get a gateway manager instance"""
    return GatewayManager()


