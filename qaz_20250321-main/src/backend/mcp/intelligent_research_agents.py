#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent Research Agents - MCP Architecture

This module implements intelligent research agents using:
- HuggingFace: For model inference, text analysis, and embeddings
- SerAPI: For academic research and paper validation
- OpenAI: For advanced reasoning and analysis
- Cross-agent collaboration for comprehensive research validation
"""

import os
import sys
import re
import json
import hashlib
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv
import fitz  # pymupdf

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Load environment variables
load_dotenv('src/backend/config/credentials.env')

class AgentType(Enum):
    """Types of intelligent research agents"""
    HUGGINGFACE = "huggingface"
    SERAPI = "serapi"
    OPENAI = "openai"
    CROSS_AGENT = "cross_agent"

@dataclass
class ResearchRequest:
    """Request for research agent analysis"""
    agent_type: AgentType
    query: str
    context: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1

@dataclass
class ResearchResponse:
    """Response from research agent"""
    success: bool
    data: Dict[str, Any]
    agent_type: AgentType
    execution_time: float = 0.0
    confidence_score: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class HuggingFaceAgent:
    """Intelligent agent using HuggingFace models for research and analysis"""
    
    def __init__(self):
        self.api_token = os.getenv('HUGGINGFACE_API_TOKEN')
        self.base_url = "https://api-inference.huggingface.co/models"
        self.available_models = {
            "text_classification": "distilbert-base-uncased-finetuned-sst-2-english",
            "question_answering": "deepset/roberta-base-squad2",
            "text_generation": "gpt2",
            "summarization": "facebook/bart-large-cnn",
            "sentiment_analysis": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "zero_shot_classification": "facebook/bart-large-mnli"
        }
    
    def analyze_text_sentiment(self, text: str) -> ResearchResponse:
        """Analyze text sentiment using HuggingFace"""
        start_time = datetime.now()
        
        try:
            if not self.api_token:
                return ResearchResponse(
                    success=False,
                    data={},
                    agent_type=AgentType.HUGGINGFACE,
                    errors=["HuggingFace API token not configured"]
                )
            
            headers = {"Authorization": f"Bearer {self.api_token}"}
            model = self.available_models["sentiment_analysis"]
            
            response = requests.post(
                f"{self.base_url}/{model}",
                headers=headers,
                json={"inputs": text}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Process sentiment results
                sentiment_data = {
                    "text": text,
                    "sentiment_scores": result,
                    "primary_sentiment": max(result, key=lambda x: x['score'])['label'],
                    "confidence": max(result, key=lambda x: x['score'])['score'],
                    "analysis_timestamp": datetime.now().isoformat()
                }
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return ResearchResponse(
                    success=True,
                    data=sentiment_data,
                    agent_type=AgentType.HUGGINGFACE,
                    execution_time=execution_time,
                    confidence_score=sentiment_data["confidence"]
                )
            else:
                return ResearchResponse(
                    success=False,
                    data={},
                    agent_type=AgentType.HUGGINGFACE,
                    errors=[f"API request failed: {response.status_code}"]
                )
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ResearchResponse(
                success=False,
                data={},
                agent_type=AgentType.HUGGINGFACE,
                execution_time=execution_time,
                errors=[str(e)]
            )
    
    def classify_text(self, text: str, candidate_labels: List[str]) -> ResearchResponse:
        """Classify text using zero-shot classification"""
        start_time = datetime.now()
        
        try:
            if not self.api_token:
                return ResearchResponse(
                    success=False,
                    data={},
                    agent_type=AgentType.HUGGINGFACE,
                    errors=["HuggingFace API token not configured"]
                )
            
            headers = {"Authorization": f"Bearer {self.api_token}"}
            model = self.available_models["zero_shot_classification"]
            
            response = requests.post(
                f"{self.base_url}/{model}",
                headers=headers,
                json={
                    "inputs": text,
                    "parameters": {"candidate_labels": candidate_labels}
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                
                classification_data = {
                    "text": text,
                    "candidate_labels": candidate_labels,
                    "classification": result,
                    "best_label": result['labels'][0],
                    "confidence": result['scores'][0],
                    "analysis_timestamp": datetime.now().isoformat()
                }
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return ResearchResponse(
                    success=True,
                    data=classification_data,
                    agent_type=AgentType.HUGGINGFACE,
                    execution_time=execution_time,
                    confidence_score=classification_data["confidence"]
                )
            else:
                return ResearchResponse(
                    success=False,
                    data={},
                    agent_type=AgentType.HUGGINGFACE,
                    errors=[f"API request failed: {response.status_code}"]
                )
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ResearchResponse(
                success=False,
                data={},
                agent_type=AgentType.HUGGINGFACE,
                execution_time=execution_time,
                errors=[str(e)]
            )
    
    def answer_question(self, question: str, context: str) -> ResearchResponse:
        """Answer questions using question-answering model"""
        start_time = datetime.now()
        
        try:
            if not self.api_token:
                return ResearchResponse(
                    success=False,
                    data={},
                    agent_type=AgentType.HUGGINGFACE,
                    errors=["HuggingFace API token not configured"]
                )
            
            headers = {"Authorization": f"Bearer {self.api_token}"}
            model = self.available_models["question_answering"]
            
            response = requests.post(
                f"{self.base_url}/{model}",
                headers=headers,
                json={
                    "inputs": {
                        "question": question,
                        "context": context
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                
                qa_data = {
                    "question": question,
                    "context": context,
                    "answer": result['answer'],
                    "confidence": result['score'],
                    "start": result['start'],
                    "end": result['end'],
                    "analysis_timestamp": datetime.now().isoformat()
                }
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return ResearchResponse(
                    success=True,
                    data=qa_data,
                    agent_type=AgentType.HUGGINGFACE,
                    execution_time=execution_time,
                    confidence_score=qa_data["confidence"]
                )
            else:
                return ResearchResponse(
                    success=False,
                    data={},
                    agent_type=AgentType.HUGGINGFACE,
                    errors=[f"API request failed: {response.status_code}"]
                )
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ResearchResponse(
                success=False,
                data={},
                agent_type=AgentType.HUGGINGFACE,
                execution_time=execution_time,
                errors=[str(e)]
            )

class SerAPIAgent:
    """Intelligent agent using SerAPI for academic research and paper validation"""
    
    def __init__(self):
        self.api_key = os.getenv('SERAPI_API_KEY')
        self.base_url = "https://serpapi.com/search"
    
    def search_academic_papers(self, query: str, max_results: int = 10) -> ResearchResponse:
        """Search for academic papers using SerAPI"""
        start_time = datetime.now()
        
        try:
            if not self.api_key:
                return ResearchResponse(
                    success=False,
                    data={},
                    agent_type=AgentType.SERAPI,
                    errors=["SerAPI key not configured"]
                )
            
            params = {
                "api_key": self.api_key,
                "engine": "google_scholar",
                "q": query,
                "num": max_results,
                "as_ylo": "2015"  # Papers from 2015 onwards
            }
            
            response = requests.get(self.base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract relevant information
                papers = []
                if "organic_results" in data:
                    for result in data["organic_results"]:
                        paper_info = {
                            "title": result.get("title", ""),
                            "authors": result.get("authors", ""),
                            "publication": result.get("publication", ""),
                            "year": result.get("year", ""),
                            "citations": result.get("inline_links", {}).get("cited_by", {}).get("total", 0),
                            "snippet": result.get("snippet", ""),
                            "link": result.get("link", "")
                        }
                        papers.append(paper_info)
                
                search_data = {
                    "query": query,
                    "total_results": len(papers),
                    "papers": papers,
                    "search_metadata": {
                        "total_results_found": data.get("search_information", {}).get("total_results", 0),
                        "time_taken": data.get("search_information", {}).get("time_taken_displayed", 0)
                    },
                    "analysis_timestamp": datetime.now().isoformat()
                }
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return ResearchResponse(
                    success=True,
                    data=search_data,
                    agent_type=AgentType.SERAPI,
                    execution_time=execution_time,
                    confidence_score=0.8 if papers else 0.0
                )
            else:
                return ResearchResponse(
                    success=False,
                    data={},
                    agent_type=AgentType.SERAPI,
                    errors=[f"API request failed: {response.status_code}"]
                )
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ResearchResponse(
                success=False,
                data={},
                agent_type=AgentType.SERAPI,
                execution_time=execution_time,
                errors=[str(e)]
            )
    
    def validate_reference(self, reference_text: str) -> ResearchResponse:
        """Validate academic references by searching for them"""
        start_time = datetime.now()
        
        try:
            if not self.api_key:
                return ResearchResponse(
                    success=False,
                    data={},
                    agent_type=AgentType.SERAPI,
                    errors=["SerAPI key not configured"]
                )
            
            # Extract key information from reference
            author_match = re.search(r'([A-Z][a-z]+,\s*[A-Z]\.)', reference_text)
            year_match = re.search(r'(\d{4})', reference_text)
            
            if author_match and year_match:
                author = author_match.group(1)
                year = year_match.group(1)
                search_query = f'"{author}" "{year}"'
                
                params = {
                    "api_key": self.api_key,
                    "engine": "google_scholar",
                    "q": search_query,
                    "num": 5
                }
                
                response = requests.get(self.base_url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    validation_data = {
                        "reference_text": reference_text,
                        "search_query": search_query,
                        "found_papers": [],
                        "validation_score": 0.0,
                        "is_valid": False
                    }
                    
                    if "organic_results" in data and data["organic_results"]:
                        validation_data["found_papers"] = data["organic_results"][:3]
                        validation_data["validation_score"] = 0.8
                        validation_data["is_valid"] = True
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    return ResearchResponse(
                        success=True,
                        data=validation_data,
                        agent_type=AgentType.SERAPI,
                        execution_time=execution_time,
                        confidence_score=validation_data["validation_score"]
                    )
                else:
                    return ResearchResponse(
                        success=False,
                        data={},
                        agent_type=AgentType.SERAPI,
                        errors=[f"API request failed: {response.status_code}"]
                    )
            else:
                return ResearchResponse(
                    success=False,
                    data={},
                    agent_type=AgentType.SERAPI,
                    errors=["Could not extract author and year from reference"]
                )
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ResearchResponse(
                success=False,
                data={},
                agent_type=AgentType.SERAPI,
                execution_time=execution_time,
                errors=[str(e)]
            )

class OpenAIAgent:
    """Intelligent agent using OpenAI for advanced reasoning and analysis"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4"
    
    def analyze_argument_strength(self, text: str) -> ResearchResponse:
        """Analyze argument strength and reasoning quality"""
        start_time = datetime.now()
        
        try:
            if not self.api_key:
                return ResearchResponse(
                    success=False,
                    data={},
                    agent_type=AgentType.OPENAI,
                    errors=["OpenAI API key not configured"]
                )
            
            prompt = f"""
            Analyze the following text for argument strength and reasoning quality. 
            Provide a detailed analysis including:
            1. Argument strength (weak/moderate/strong)
            2. Evidence quality
            3. Logical consistency
            4. Potential biases
            5. Recommendations for improvement
            
            Text: {text}
            
            Provide your analysis in JSON format with the following structure:
            {{
                "argument_strength": "weak/moderate/strong",
                "evidence_quality": "low/medium/high",
                "logical_consistency": "low/medium/high",
                "potential_biases": ["bias1", "bias2"],
                "recommendations": ["rec1", "rec2"],
                "confidence_score": 0.0-1.0,
                "detailed_analysis": "detailed explanation"
            }}
            """
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert in argument analysis and critical thinking."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = requests.post(self.base_url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse JSON response
                try:
                    analysis_data = json.loads(content)
                    analysis_data["original_text"] = text
                    analysis_data["analysis_timestamp"] = datetime.now().isoformat()
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    return ResearchResponse(
                        success=True,
                        data=analysis_data,
                        agent_type=AgentType.OPENAI,
                        execution_time=execution_time,
                        confidence_score=analysis_data.get("confidence_score", 0.0)
                    )
                except json.JSONDecodeError:
                    return ResearchResponse(
                        success=False,
                        data={},
                        agent_type=AgentType.OPENAI,
                        errors=["Failed to parse OpenAI response as JSON"]
                    )
            else:
                return ResearchResponse(
                    success=False,
                    data={},
                    agent_type=AgentType.OPENAI,
                    errors=[f"API request failed: {response.status_code}"]
                )
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ResearchResponse(
                success=False,
                data={},
                agent_type=AgentType.OPENAI,
                execution_time=execution_time,
                errors=[str(e)]
            )
    
    def detect_contradictions(self, text: str) -> ResearchResponse:
        """Detect contradictions and inconsistencies using OpenAI"""
        start_time = datetime.now()
        
        try:
            if not self.api_key:
                return ResearchResponse(
                    success=False,
                    data={},
                    agent_type=AgentType.OPENAI,
                    errors=["OpenAI API key not configured"]
                )
            
            prompt = f"""
            Analyze the following text for contradictions, inconsistencies, and logical conflicts.
            Identify any statements that contradict each other or are inconsistent.
            
            Text: {text}
            
            Provide your analysis in JSON format:
            {{
                "contradictions_found": true/false,
                "contradiction_count": 0,
                "contradictions": [
                    {{
                        "statement1": "first statement",
                        "statement2": "contradicting statement",
                        "explanation": "why they contradict",
                        "severity": "low/medium/high"
                    }}
                ],
                "overall_consistency": "high/medium/low",
                "confidence_score": 0.0-1.0
            }}
            """
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert in logical analysis and contradiction detection."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 800
            }
            
            response = requests.post(self.base_url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                try:
                    contradiction_data = json.loads(content)
                    contradiction_data["original_text"] = text
                    contradiction_data["analysis_timestamp"] = datetime.now().isoformat()
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    return ResearchResponse(
                        success=True,
                        data=contradiction_data,
                        agent_type=AgentType.OPENAI,
                        execution_time=execution_time,
                        confidence_score=contradiction_data.get("confidence_score", 0.0)
                    )
                except json.JSONDecodeError:
                    return ResearchResponse(
                        success=False,
                        data={},
                        agent_type=AgentType.OPENAI,
                        errors=["Failed to parse OpenAI response as JSON"]
                    )
            else:
                return ResearchResponse(
                    success=False,
                    data={},
                    agent_type=AgentType.OPENAI,
                    errors=[f"API request failed: {response.status_code}"]
                )
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ResearchResponse(
                success=False,
                data={},
                agent_type=AgentType.OPENAI,
                execution_time=execution_time,
                errors=[str(e)]
            )
    
    def validate_claims(self, claims: List[str]) -> ResearchResponse:
        """Validate claims against current knowledge"""
        start_time = datetime.now()
        
        try:
            if not self.api_key:
                return ResearchResponse(
                    success=False,
                    data={},
                    agent_type=AgentType.OPENAI,
                    errors=["OpenAI API key not configured"]
                )
            
            claims_text = "\n".join([f"{i+1}. {claim}" for i, claim in enumerate(claims)])
            
            prompt = f"""
            Evaluate the following claims for accuracy and validity based on current knowledge.
            For each claim, provide:
            1. Validity assessment (valid/invalid/unclear)
            2. Confidence level (high/medium/low)
            3. Supporting evidence or counter-evidence
            4. Recommendations for verification
            
            Claims:
            {claims_text}
            
            Provide your analysis in JSON format:
            {{
                "claims_analysis": [
                    {{
                        "claim": "original claim",
                        "validity": "valid/invalid/unclear",
                        "confidence": "high/medium/low",
                        "evidence": "supporting or counter evidence",
                        "recommendations": ["rec1", "rec2"]
                    }}
                ],
                "overall_assessment": "summary of findings",
                "confidence_score": 0.0-1.0
            }}
            """
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert fact-checker and claim validator."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 1200
            }
            
            response = requests.post(self.base_url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                try:
                    validation_data = json.loads(content)
                    validation_data["original_claims"] = claims
                    validation_data["analysis_timestamp"] = datetime.now().isoformat()
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    return ResearchResponse(
                        success=True,
                        data=validation_data,
                        agent_type=AgentType.OPENAI,
                        execution_time=execution_time,
                        confidence_score=validation_data.get("confidence_score", 0.0)
                    )
                except json.JSONDecodeError:
                    return ResearchResponse(
                        success=False,
                        data={},
                        agent_type=AgentType.OPENAI,
                        errors=["Failed to parse OpenAI response as JSON"]
                    )
            else:
                return ResearchResponse(
                    success=False,
                    data={},
                    agent_type=AgentType.OPENAI,
                    errors=[f"API request failed: {response.status_code}"]
                )
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ResearchResponse(
                success=False,
                data={},
                agent_type=AgentType.OPENAI,
                execution_time=execution_time,
                errors=[str(e)]
            )

class CrossAgentCoordinator:
    """Coordinates multiple AI agents for comprehensive research analysis"""
    
    def __init__(self):
        self.hf_agent = HuggingFaceAgent()
        self.serapi_agent = SerAPIAgent()
        self.openai_agent = OpenAIAgent()
        self.analysis_cache = {}
    
    def comprehensive_document_analysis(self, document_path: str) -> Dict[str, Any]:
        """Perform comprehensive document analysis using all agents"""
        print(f"🤖 Multi-Agent Analysis: {os.path.basename(document_path)}")
        
        # Extract text from document
        doc = fitz.open(document_path)
        all_text = ""
        for page in doc:
            all_text += page.get_text()
        doc.close()
        
        # Split text into manageable chunks
        chunks = self._split_text_into_chunks(all_text, max_chunk_size=1000)
        
        analysis_results = {
            "document_name": os.path.splitext(os.path.basename(document_path))[0],
            "analysis_timestamp": datetime.now().isoformat(),
            "agents_used": ["HuggingFace", "SerAPI", "OpenAI"],
            "chunk_analyses": [],
            "overall_assessment": {},
            "cross_agent_insights": {}
        }
        
        # Analyze each chunk with multiple agents
        for i, chunk in enumerate(chunks):
            print(f"  📝 Analyzing chunk {i+1}/{len(chunks)}...")
            
            chunk_analysis = {
                "chunk_id": f"chunk_{i+1:03d}",
                "text": chunk,
                "agent_results": {}
            }
            
            # HuggingFace analysis
            sentiment_response = self.hf_agent.analyze_text_sentiment(chunk)
            if sentiment_response.success:
                chunk_analysis["agent_results"]["huggingface_sentiment"] = sentiment_response.data
            
            # OpenAI analysis
            argument_response = self.openai_agent.analyze_argument_strength(chunk)
            if argument_response.success:
                chunk_analysis["agent_results"]["openai_argument_analysis"] = argument_response.data
            
            contradiction_response = self.openai_agent.detect_contradictions(chunk)
            if contradiction_response.success:
                chunk_analysis["agent_results"]["openai_contradiction_analysis"] = contradiction_response.data
            
            analysis_results["chunk_analyses"].append(chunk_analysis)
        
        # Extract claims for validation
        claims = self._extract_claims_from_text(all_text)
        if claims:
            print(f"  🔍 Validating {len(claims)} claims...")
            claims_response = self.openai_agent.validate_claims(claims[:5])  # Limit to 5 claims
            if claims_response.success:
                analysis_results["claims_validation"] = claims_response.data
        
        # Search for related academic papers
        search_query = self._generate_search_query(all_text)
        if search_query:
            print(f"  📚 Searching for related papers: {search_query}")
            papers_response = self.serapi_agent.search_academic_papers(search_query, max_results=5)
            if papers_response.success:
                analysis_results["related_papers"] = papers_response.data
        
        # Generate overall assessment
        analysis_results["overall_assessment"] = self._generate_overall_assessment(analysis_results)
        
        # Generate cross-agent insights
        analysis_results["cross_agent_insights"] = self._generate_cross_agent_insights(analysis_results)
        
        return analysis_results
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Split text into manageable chunks"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_size = len(sentence)
            
            if current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _extract_claims_from_text(self, text: str) -> List[str]:
        """Extract claims from text using regex patterns"""
        claims = []
        
        # Patterns for claims
        claim_patterns = [
            r'[A-Z][^.!?]*\s+(is|are|was|were|will be|should be|must be)[^.!?]*[.!?]',
            r'[A-Z][^.!?]*\s+(proves|demonstrates|shows|indicates|suggests)[^.!?]*[.!?]',
            r'[A-Z][^.!?]*\s+(therefore|thus|consequently|as a result)[^.!?]*[.!?]'
        ]
        
        for pattern in claim_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                claim = match.group(0).strip()
                if len(claim) > 20:  # Only include substantial claims
                    claims.append(claim)
        
        return claims[:10]  # Limit to 10 claims
    
    def _generate_search_query(self, text: str) -> str:
        """Generate search query from text"""
        # Extract key terms
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        word_freq = {}
        
        for word in words:
            if len(word) > 3:  # Only consider substantial words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top 3 most frequent words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if top_words:
            return " ".join([word for word, freq in top_words])
        else:
            return "model validation research"
    
    def _generate_overall_assessment(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment from all agent results"""
        assessment = {
            "document_quality": "unknown",
            "argument_strength": "unknown",
            "evidence_quality": "unknown",
            "contradiction_level": "unknown",
            "recommendations": []
        }
        
        # Aggregate sentiment scores
        sentiment_scores = []
        argument_scores = []
        contradiction_counts = []
        
        for chunk_analysis in analysis_results.get("chunk_analyses", []):
            # Collect sentiment data
            if "huggingface_sentiment" in chunk_analysis["agent_results"]:
                sentiment_data = chunk_analysis["agent_results"]["huggingface_sentiment"]
                if "confidence" in sentiment_data:
                    sentiment_scores.append(sentiment_data["confidence"])
            
            # Collect argument strength data
            if "openai_argument_analysis" in chunk_analysis["agent_results"]:
                arg_data = chunk_analysis["agent_results"]["openai_argument_analysis"]
                if "confidence_score" in arg_data:
                    argument_scores.append(arg_data["confidence_score"])
            
            # Collect contradiction data
            if "openai_contradiction_analysis" in chunk_analysis["agent_results"]:
                contra_data = chunk_analysis["agent_results"]["openai_contradiction_analysis"]
                if "contradiction_count" in contra_data:
                    contradiction_counts.append(contra_data["contradiction_count"])
        
        # Calculate overall scores
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            if avg_sentiment > 0.7:
                assessment["document_quality"] = "high"
            elif avg_sentiment > 0.4:
                assessment["document_quality"] = "medium"
            else:
                assessment["document_quality"] = "low"
        
        if argument_scores:
            avg_argument = sum(argument_scores) / len(argument_scores)
            if avg_argument > 0.7:
                assessment["argument_strength"] = "strong"
            elif avg_argument > 0.4:
                assessment["argument_strength"] = "moderate"
            else:
                assessment["argument_strength"] = "weak"
        
        if contradiction_counts:
            total_contradictions = sum(contradiction_counts)
            if total_contradictions == 0:
                assessment["contradiction_level"] = "none"
            elif total_contradictions < 3:
                assessment["contradiction_level"] = "low"
            elif total_contradictions < 10:
                assessment["contradiction_level"] = "medium"
            else:
                assessment["contradiction_level"] = "high"
        
        # Generate recommendations
        if assessment["argument_strength"] == "weak":
            assessment["recommendations"].append("Document has weak arguments - needs more evidence")
        
        if assessment["contradiction_level"] in ["medium", "high"]:
            assessment["recommendations"].append("High contradiction level - requires careful review")
        
        if assessment["document_quality"] == "low":
            assessment["recommendations"].append("Document quality is low - consider revision")
        
        return assessment
    
    def _generate_cross_agent_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from cross-agent collaboration"""
        insights = {
            "consensus_areas": [],
            "conflicting_assessments": [],
            "research_gaps": [],
            "validation_opportunities": []
        }
        
        # Analyze consensus between agents
        for chunk_analysis in analysis_results.get("chunk_analyses", []):
            agent_results = chunk_analysis["agent_results"]
            
            # Check for consensus in sentiment and argument analysis
            if "huggingface_sentiment" in agent_results and "openai_argument_analysis" in agent_results:
                hf_sentiment = agent_results["huggingface_sentiment"]
                oai_argument = agent_results["openai_argument_analysis"]
                
                # Check if both agents agree on quality
                if (hf_sentiment.get("confidence", 0) > 0.7 and 
                    oai_argument.get("confidence_score", 0) > 0.7):
                    insights["consensus_areas"].append(f"High confidence in chunk {chunk_analysis['chunk_id']}")
                elif (hf_sentiment.get("confidence", 0) < 0.3 and 
                      oai_argument.get("confidence_score", 0) < 0.3):
                    insights["consensus_areas"].append(f"Low confidence in chunk {chunk_analysis['chunk_id']}")
        
        # Identify research gaps
        if "related_papers" in analysis_results:
            papers = analysis_results["related_papers"].get("papers", [])
            if len(papers) < 3:
                insights["research_gaps"].append("Limited related research found")
        
        # Identify validation opportunities
        if "claims_validation" in analysis_results:
            claims_data = analysis_results["claims_validation"]
            if claims_data.get("claims_analysis"):
                unclear_claims = [
                    claim for claim in claims_data["claims_analysis"]
                    if claim.get("validity") == "unclear"
                ]
                if unclear_claims:
                    insights["validation_opportunities"].append(f"{len(unclear_claims)} claims need further validation")
        
        return insights

def main():
    """Main function to demonstrate intelligent research agents"""
    # Initialize the cross-agent coordinator
    coordinator = CrossAgentCoordinator()
    
    # Get PDF files
    reviews_dir = Path("data/input/reviews")
    pdf_files = list(reviews_dir.glob("*.pdf"))
    
    # Skip already processed files
    pdf_files = [f for f in pdf_files if f.name not in [
        "Whitepaper-Model-Validation-Best-Practices-1.pdf", 
        "investment-model-validation.pdf"
    ]]
    
    print(f"🤖 Intelligent Research Agents - Multi-Agent Analysis")
    print(f"📊 Processing {len(pdf_files)} documents")
    print("=" * 70)
    
    all_results = {}
    
    # Process each document
    for pdf_file in pdf_files:
        try:
            print(f"\n🔍 Processing: {pdf_file.name}")
            results = coordinator.comprehensive_document_analysis(str(pdf_file))
            all_results[pdf_file.name] = results
            print(f"✅ Completed: {pdf_file.name}")
        except Exception as e:
            print(f"❌ Failed to process {pdf_file.name}: {e}")
    
    # Generate comprehensive report
    report = {
        "intelligent_agents_summary": {
            "total_documents": len(all_results),
            "analysis_timestamp": datetime.now().isoformat(),
            "agents_used": ["HuggingFace", "SerAPI", "OpenAI"],
            "system_architecture": "Multi-Agent MCP System"
        },
        "document_analyses": all_results,
        "cross_document_insights": {
            "total_chunks_analyzed": sum(
                len(r.get("chunk_analyses", [])) for r in all_results.values()
            ),
            "total_claims_validated": sum(
                len(r.get("claims_validation", {}).get("claims_analysis", [])) 
                for r in all_results.values()
            ),
            "total_papers_found": sum(
                len(r.get("related_papers", {}).get("papers", [])) 
                for r in all_results.values()
            )
        }
    }
    
    # Save report
    with open("intelligent_agents_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n🎉 Intelligent Research Agents Analysis completed!")
    print(f"📊 Processed {len(all_results)} documents")
    print(f"📄 Report saved to: intelligent_agents_analysis_report.json")
    
    # Print summary statistics
    total_chunks = report["cross_document_insights"]["total_chunks_analyzed"]
    total_claims = report["cross_document_insights"]["total_claims_validated"]
    total_papers = report["cross_document_insights"]["total_papers_found"]
    
    print(f"📝 Total chunks analyzed: {total_chunks}")
    print(f"🔍 Total claims validated: {total_claims}")
    print(f"📚 Total related papers found: {total_papers}")

if __name__ == "__main__":
    main()
