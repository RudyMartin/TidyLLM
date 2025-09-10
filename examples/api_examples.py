

 # Basic API Examples


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
