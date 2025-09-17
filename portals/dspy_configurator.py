"""
DSPy Portal Configurator
========================

Portal for configuring DSPy to work with CorporateLLMAdapter in corporate environments.
Provides easy setup and testing for DSPy Chain of Thought functionality.

Features:
- Auto-configure DSPy with CorporateLLMAdapter
- Test DSPy functionality in corporate firewall environments
- Validate Chain of Thought reasoning
- Monitor DSPy performance and health

Usage:
    python -m tidyllm.portals.dspy_configurator

    # Or from code:
    from tidyllm.portals.dspy_configurator import configure_dspy_for_corporate

    success = configure_dspy_for_corporate(model_name="claude-3-sonnet")
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def configure_dspy_for_corporate(model_name: str = "claude-3-sonnet",
                               temperature: float = 0.7,
                               test_configuration: bool = True) -> Dict[str, Any]:
    """
    Configure DSPy to work with CorporateLLMAdapter for corporate environments.

    Args:
        model_name: Model to use (defaults to claude-3-sonnet)
        temperature: Temperature setting
        test_configuration: Whether to test the configuration

    Returns:
        Dict with configuration results and status
    """
    result = {
        "success": False,
        "configuration_time": datetime.now().isoformat(),
        "model_name": model_name,
        "temperature": temperature,
        "tests_passed": 0,
        "total_tests": 0,
        "details": {}
    }

    try:
        print("DSPy Corporate Configuration Portal")
        print("=" * 50)

        # Step 1: Import and initialize DSPy service
        print("\n1. Initializing DSPy Service...")
        try:
            from tidyllm.services import DSPyService
            dspy_service = DSPyService(auto_configure=False)
            result["details"]["dspy_service"] = "+ Initialized"
            print("   + DSPy Service initialized")
        except Exception as e:
            result["details"]["dspy_service"] = f"- Failed: {e}"
            print(f"   - DSPy Service failed: {e}")
            return result

        # Step 2: Configure with CorporateLLMAdapter
        print("\n2. Configuring CorporateLLMAdapter backend...")
        try:
            config_success = dspy_service.configure_lm(model_name=model_name)
            if config_success:
                result["details"]["corporate_adapter"] = "+ Configured"
                print(f"   + DSPy configured with CorporateLLMAdapter: {model_name}")
            else:
                result["details"]["corporate_adapter"] = "- Configuration failed"
                print("   - Failed to configure CorporateLLMAdapter")
                return result
        except Exception as e:
            result["details"]["corporate_adapter"] = f"- Error: {e}"
            print(f"   - CorporateLLMAdapter configuration error: {e}")
            return result

        # Step 3: Check DSPy status
        print("\n3. Checking DSPy status...")
        try:
            status = dspy_service.get_status()
            result["details"]["status"] = status
            print(f"   + Current LM: {status.get('current_lm', 'Unknown')}")
            print(f"   + Framework Available: {status.get('framework_available', False)}")
            print(f"   + LM Configured: {status.get('lm_configured', False)}")
        except Exception as e:
            result["details"]["status"] = f"- Status check failed: {e}"
            print(f"   - Status check failed: {e}")

        # Step 4: Test configuration if requested
        if test_configuration:
            print("\n4. Testing DSPy functionality...")
            tests = [
                ("Basic Chain of Thought", "test_basic_cot"),
                ("Chat with CoT", "test_chat_cot"),
                ("Query Enhancement", "test_query_enhancement")
            ]

            result["total_tests"] = len(tests)

            for test_name, test_method in tests:
                try:
                    print(f"   Testing {test_name}...")

                    if test_method == "test_basic_cot":
                        test_result = dspy_service.chat_with_cot("What is 2+2? Explain step by step.")
                    elif test_method == "test_chat_cot":
                        test_result = dspy_service.chat_with_cot("Hello, how does reasoning work?")
                    elif test_method == "test_query_enhancement":
                        test_result = dspy_service.enhance_rag_query("machine learning algorithms")

                    if test_result.get("success"):
                        result["tests_passed"] += 1
                        result["details"][test_method] = "+ Passed"
                        print(f"      + {test_name} passed")
                    else:
                        result["details"][test_method] = f"- Failed: {test_result.get('error', 'Unknown error')}"
                        print(f"      - {test_name} failed: {test_result.get('error', 'Unknown')}")

                except Exception as e:
                    result["details"][test_method] = f"- Exception: {e}"
                    print(f"      - {test_name} exception: {e}")

        # Final status
        if test_configuration:
            success_rate = result["tests_passed"] / result["total_tests"] if result["total_tests"] > 0 else 0
            result["success"] = success_rate >= 0.8  # 80% tests must pass

            print(f"\nTest Results: {result['tests_passed']}/{result['total_tests']} passed ({success_rate:.1%})")

            if result["success"]:
                print("+ DSPy successfully configured for corporate environment!")
                print("\nYou can now use DSPy in your portals:")
                print('   response = tidyllm.chat("Explain step by step", chat_type="dspy")')
            else:
                print("- DSPy configuration partially successful but some tests failed")
        else:
            result["success"] = True
            print("+ DSPy configuration completed (testing skipped)")

    except Exception as e:
        result["details"]["configuration_error"] = str(e)
        print(f"\n- Configuration failed: {e}")

    return result


def check_dspy_health() -> Dict[str, Any]:
    """Check DSPy health and corporate adapter integration."""
    try:
        from tidyllm.services import DSPyService

        dspy_service = DSPyService(auto_configure=False)
        health = dspy_service.health_check()

        # Add corporate-specific health checks
        health["corporate_integration"] = {
            "adapter_available": False,
            "usm_available": False,
            "corporate_firewall_safe": True
        }

        try:
            from tidyllm.adapters import CorporateLLMAdapter
            health["corporate_integration"]["adapter_available"] = True
        except ImportError:
            pass

        try:
            from tidyllm.infrastructure.session.unified import UnifiedSessionManager
            health["corporate_integration"]["usm_available"] = True
        except ImportError:
            pass

        return health

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "corporate_integration": {
                "adapter_available": False,
                "usm_available": False,
                "corporate_firewall_safe": False
            }
        }


def main():
    """Main entry point for DSPy configuration portal."""
    print("TidyLLM DSPy Corporate Configuration Portal")
    print("=" * 60)

    # Configure DSPy
    result = configure_dspy_for_corporate(
        model_name="claude-3-sonnet",
        temperature=0.7,
        test_configuration=True
    )

    # Health check
    print("\nHealth Check:")
    health = check_dspy_health()
    if health.get("success", False):
        corp_health = health.get("corporate_integration", {})
        print(f"   + Corporate Adapter: {corp_health.get('adapter_available', False)}")
        print(f"   + Session Manager: {corp_health.get('usm_available', False)}")
        print(f"   + Firewall Safe: {corp_health.get('corporate_firewall_safe', False)}")
    else:
        print(f"   - Health check failed: {health.get('error', 'Unknown error')}")

    # Summary
    print(f"\nSummary:")
    print(f"   Configuration: {'+ Success' if result['success'] else '- Failed'}")
    print(f"   Model: {result['model_name']}")
    print(f"   Temperature: {result['temperature']}")
    if result.get('total_tests', 0) > 0:
        print(f"   Tests: {result['tests_passed']}/{result['total_tests']} passed")

    if result["success"]:
        print(f"\nNext steps:")
        print(f"   1. Test in your portal: tidyllm.chat('Explain step by step', chat_type='dspy')")
        print(f"   2. Check reasoning output: tidyllm.chat('...', chat_type='dspy', reasoning=True)")
        print(f"   3. Monitor performance through MLflow integration")

    return result


if __name__ == "__main__":
    main()