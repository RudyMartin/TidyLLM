 Basic API Examples

  import tidyllm

  # Simple chat
  response = tidyllm.chat("Hello, how are you?")
  print(response)

  # Query with context
  answer = tidyllm.query("What is this about?", context="Document content...")
  print(answer)

  # Process documents
  result = tidyllm.process_document("document.pdf", "Summarize this")
  print(result)

  # List available models
  models = tidyllm.list_models()
  print(models)

  # Set default model
  tidyllm.set_model("anthropic/claude-3-sonnet-20240229")

  # Check API status
  status = tidyllm.status()
  print(status)

  Advanced Gateway Examples

  from tidyllm.gateways import init_gateways, get_global_registry
  from tidyllm.gateways.corporate_llm_gateway import LLMRequest

  # Initialize gateway registry
  registry = init_gateways({
      "corporate_llm": {
          "budget_limit_daily_usd": 10.0,
          "compliance_mode": True,
          "audit_enabled": True
      }
  })

  # Use corporate gateway
  corporate_gateway = registry.get("corporate_llm")
  llm_request = LLMRequest(
      prompt="Hello, world!",
      model="claude-3-sonnet",
      audit_reason="api_test"
  )
  response = corporate_gateway.process_llm_request(llm_request)

  Configuration

  The API automatically routes through audit-compliant gateways and falls back to simulation mode when credentials
  aren't available.