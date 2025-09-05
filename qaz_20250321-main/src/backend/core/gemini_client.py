"""
Custom Gemini Client for VectorQA Sage
Works without google-auth dependency
"""

import requests
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class GeminiClient:
    """Custom Gemini client using direct API calls."""
    
    def __init__(self, api_key: str):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google API key for Gemini
        """
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': api_key
        }
    
    def generate_content(self, 
                        prompt: str, 
                        system_message: Optional[str] = None,
                        max_tokens: int = 1000,
                        temperature: float = 0.1) -> str:
        """
        Generate content using Gemini API.
        
        Args:
            prompt: The user prompt
            system_message: Optional system message
            max_tokens: Maximum tokens to generate
            temperature: Creativity level (0.0 to 1.0)
            
        Returns:
            Generated text response
        """
        try:
            # Prepare the request
            url = f"{self.base_url}/models/gemini-2.0-flash:generateContent"
            
            # Build the content
            content_parts = []
            
            # Add system message if provided
            if system_message:
                content_parts.append({
                    "role": "user",
                    "parts": [{"text": f"System: {system_message}\n\nUser: {prompt}"}]
                })
            else:
                content_parts.append({
                    "role": "user", 
                    "parts": [{"text": prompt}]
                })
            
            data = {
                "contents": content_parts,
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature
                }
            }
            
            # Make the request
            response = requests.post(url, headers=self.headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    return result['candidates'][0]['content']['parts'][0]['text'].strip()
                else:
                    logger.error(f"Unexpected response format: {result}")
                    return "Error: Unexpected response format"
            else:
                logger.error(f"API Error {response.status_code}: {response.text}")
                return f"Error: API returned {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return f"Error: Request failed - {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"Error: {str(e)}"
    
    def answer_question(self, question: str, context: str) -> str:
        """
        Answer a question based on provided context.
        
        Args:
            question: The question to answer
            context: Relevant context
            
        Returns:
            Answer to the question
        """
        prompt = f"""Context:
{context}

Question: {question}

Answer based on the context above:"""
        
        return self.generate_content(prompt, temperature=0.0)
    
    def validate_qa(self, topic: str, report_chunk: str, retrieved_context: str) -> str:
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

        response = self.generate_content(prompt, system_message, temperature=0.0)
        
        # Clean up response
        response = response.strip().lower()
        if "correct" in response:
            return "Correct"
        elif "missing" in response or "incomplete" in response:
            return "Missing Info"
        else:
            return "Inconsistent"
    
    def summarize_document(self, document_text: str) -> str:
        """
        Summarize a document.
        
        Args:
            document_text: The document to summarize
            
        Returns:
            Summary of the document
        """
        prompt = f"""Please provide a concise summary of the following document:

{document_text}

Summary:"""
        
        return self.generate_content(prompt, temperature=0.3)
