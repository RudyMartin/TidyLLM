---
name: Bug report
about: Create a report to help us improve TidyLLM
title: '[BUG] '
labels: ['bug']
assignees: ''
---

## 🐛 Bug Description

A clear and concise description of what the bug is.

## 🔄 Steps to Reproduce

1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## ✅ Expected Behavior

A clear and concise description of what you expected to happen.

## ❌ Actual Behavior

A clear and concise description of what actually happened.

## 📋 Environment

**OS:** [e.g. macOS 12.0, Ubuntu 20.04, Windows 11]
**Python Version:** [e.g. 3.9.7]
**TidyLLM Version:** [e.g. 0.1.0]
**Package Manager:** [e.g. pip, conda]

## 📦 Dependencies

```bash
# Run this and paste the output
pip freeze | grep -E "(tidyllm|dspy|datatable|mlflow)"
```

## 💻 Code Example

```python
from tidyllm import llm_message, chat, claude

# Minimal code that reproduces the issue
response = llm_message("Hello") | chat(claude())
print(response)
```

## 📝 Additional Context

Add any other context about the problem here, such as:
- Error messages or stack traces
- Screenshots if applicable
- Related issues or discussions

## 🔍 Debugging Information

If you have debugging information, please include:
- Log files
- Debug output
- Performance metrics

## ✅ Checklist

- [ ] I have searched existing issues to avoid duplicates
- [ ] I have provided a minimal reproduction example
- [ ] I have included all relevant environment information
- [ ] I have tested with the latest version of TidyLLM


