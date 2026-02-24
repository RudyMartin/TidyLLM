
Here is the 2025 version of the OWASP Top 10 for LLMs in a structured two-column table format: Title/Detail and Example/Remediation. Let me know if you’d like a downloadable version or if you'd prefer this exported to Markdown or CSV.

| Title / Detail | Example / Remediation |
|----------------|------------------------|
| **LLM01: Prompt Injection**<br>Malicious prompts alter system behavior or bypass controls. | **Example**: 'Ignore previous instructions...'<br>**Remediation**: Separate user/system prompts; apply strict validation. |
| **LLM02: Sensitive Information Disclosure**<br>LLMs expose secrets or PII from training or context. | **Example**: API keys revealed from prompt history.<br>**Remediation**: Filter outputs; strip sensitive data from training logs. |
| **LLM03: Supply Chain Vulnerabilities**<br>Insecure model components or dependencies. | **Example**: Compromised plugin leads to RCE.<br>**Remediation**: Verify sources, use signed packages, monitor packages. |
| **LLM04: Data and Model Poisoning**<br>Malicious inputs poison behavior or logic. | **Example**: Poisoned fine-tuning data creates toxic outputs.<br>**Remediation**: Vet datasets; log and validate sources. |
| **LLM05: Improper Output Handling**<br>Generated text triggers unwanted actions or security issues. | **Example**: HTML injection via generated code.<br>**Remediation**: Sanitize output; isolate downstream systems. |
| **LLM06: Excessive Agency**<br>LLMs given unsafe execution power without checks. | **Example**: Agent books expensive resources.<br>**Remediation**: Add human-in-the-loop; enforce permission boundaries. |
| **LLM07: System Prompt Leakage**<br>Exposure of hidden system instructions. | **Example**: User tricks model into revealing sys prompt.<br>**Remediation**: Mask internal prompts; test for jailbreaks. |
| **LLM08: Vector and Embedding Weaknesses**<br>Embedding misuse or vector injection. | **Example**: Malicious vector contaminates nearest-neighbor retrieval.<br>**Remediation**: Validate inputs to embedding layer; monitor vector DB. |
| **LLM09: Misinformation**<br>LLMs generate plausible but false outputs. | **Example**: Legal doc includes hallucinated regulation.<br>**Remediation**: Use retrieval-augmented generation (RAG); provide trusted context. |
| **LLM10: Unbounded Consumption**<br>Uncontrolled compute, memory, or cost via input abuse. | **Example**: Large input leads to OOM crash or high costs.<br>**Remediation**: Limit input size; rate-limit usage; detect prompt loops. |
