from dspy import Example

def render_dspy_prompt(example: Example, strategy="naive", retrieved_texts=None):
    if strategy == "cot":
        return f'''Given the topic "{example.topic}", think step by step about whether this report chunk:
---
{example.report_chunk}
---
is consistent with the following supporting info:
---
{retrieved_texts}
---
Answer with one of: Correct, Missing Info, Inconsistent.'''

    elif strategy == "retrieval":
        return f'''Topic: {example.topic}
Chunk: {example.report_chunk}
Context: {retrieved_texts}
How do they compare?'''

    return f'''Is the following chunk relevant and complete based on the topic "{example.topic}"?

{example.report_chunk}

Context: {retrieved_texts}'''
