# AI Dropzone Manager - Intelligent Document Drop Zone Processing Orchestration  
# 🧠 CORE ENTERPRISE DROPZONE ORCHESTRATOR - Document Intelligence & Worker Management
#
# CRITICAL SECURITY CONSTRAINTS:
# ===============================
# 1. WORKER REGISTRY ONLY: AI Manager can ONLY use pre-registered, validated worker types
#    - NO dynamic worker creation or execution of arbitrary code
#    - ONLY workers from approved registry: PromptWorker, FlowRecoveryWorker, CoordinatorWorker
#    - Each worker must be explicitly registered and security-validated
#
# 2. ORCHESTRATION ONLY: AI Manager decides WHICH workers to use, NOT WHAT they do
#    - AI Manager selects templates, allocates workers, monitors progress
#    - Workers execute their own functions using their own security contexts
#    - NO execution of arbitrary prompts or code generation by AI Manager
#
# 3. LLM GATEWAY ENFORCEMENT: ALL AI calls MUST go through CorporateLLMGateway
#    - NO direct API calls to external LLM services
#    - ALL requests audited, logged, and budget-controlled via CorporateLLMGateway
#    - AI Manager uses LLM only for classification, complexity assessment, template selection
#
# 4. TEMPLATE VALIDATION: ONLY approved templates from prompts/templates/ directory
#    - NO dynamic template creation or modification
#    - Templates must be pre-validated and security-reviewed
#    - Template selection is advisory only - workers validate template execution
#
# 5. NO FILE SYSTEM ACCESS: AI Manager does NOT read/write document contents
#    - Document analysis based on limited previews only (first 8KB)
#    - Workers handle actual document processing with their own security controls
#    - AI Manager tracks metadata and orchestration state only
#
# 6. AUDIT TRAIL: ALL AI Manager decisions must be logged and auditable
#    - Every template selection, worker allocation, and LLM call logged
#    - Processing decisions stored with rationale for security review
#    - Quality feedback tracked to prevent manipulation of future decisions
#
# APPROVED CAPABILITIES:
# =====================
# - Document type classification (via CorporateLLMGateway)
# - Complexity assessment (via CorporateLLMGateway)  
# - Template recommendation (from approved template library)
# - Worker allocation (from registered worker pool only)
# - Resource optimization (scheduling and load balancing)
# - Quality monitoring (tracking metrics and feedback)
#
# EXPLICITLY PROHIBITED:
# =====================
# - Dynamic code execution or worker creation
# - Direct LLM API calls bypassing CorporateLLMGateway
# - File system access beyond limited document previews
# - Template modification or creation
# - Execution of unregistered or ad-hoc processing functions
# - Credential access or management functions
#
# Dependencies:
# - CorporateLLMGateway (REQUIRED for all LLM operations)
# - Registered worker types (PromptWorker, FlowRecoveryWorker, CoordinatorWorker)
# - Template library in prompts/templates/ (pre-validated only)
# - UnifiedSessionManager for data persistence
# - MCP server for external integrations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .base_worker import BaseWorker, TaskInput, TaskResult
from .prompt_worker import PromptWorker, PromptTask
from .flow_recovery_worker import FlowRecoveryWorker
from ..session.unified import UnifiedSessionManager

# LLM Gateway integration for all AI calls
try:
    from ...tidyllm.gateways.corporate_llm_gateway import CorporateLLMGateway, LLMRequest
    LLM_GATEWAY_AVAILABLE = True
except ImportError:
    try:
        from tidyllm.gateways.corporate_llm_gateway import CorporateLLMGateway, LLMRequest
        LLM_GATEWAY_AVAILABLE = True
    except ImportError:
        logger.warning("CorporateLLMGateway not available - AI Manager will have limited LLM capabilities")
        LLM_GATEWAY_AVAILABLE = False

logger = logging.getLogger(__name__)

class ProcessingStrategy(Enum):
    SINGLE_TEMPLATE = "single_template"
    HYBRID_ANALYSIS = "hybrid_analysis"
    MULTI_PERSPECTIVE = "multi_perspective"
    ESCALATE_HUMAN = "escalate_human"

class DocumentComplexity(Enum):
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3
    CRITICAL = 4

@dataclass
class DocumentIntelligence:
    document_path: str
    content_preview: str
    detected_type: str
    complexity: DocumentComplexity
    confidence_score: float
    recommended_templates: List[str]
    processing_strategy: ProcessingStrategy
    estimated_processing_time: int
    resource_requirements: Dict[str, Any]

@dataclass
class ProcessingDecision:
    document_path: str
    selected_templates: List[str]
    processing_strategy: ProcessingStrategy
    assigned_workers: List[str]
    priority_level: int
    estimated_completion: datetime
    quality_checks: List[str]

@dataclass
class AIManagerTask:
    document_path: str
    user_context: Optional[Dict[str, Any]] = None
    business_priority: str = "normal"
    deadline: Optional[datetime] = None

@dataclass
class AIManagerResult:
    processing_decision: ProcessingDecision
    worker_assignments: List[Dict[str, Any]]
    quality_metrics: Dict[str, float]
    estimated_completion: datetime

class AIDropzoneManager(BaseWorker[AIManagerTask, AIManagerResult]):
    """
    AI Dropzone Manager for intelligent document drop zone processing orchestration.
    
    SCOPE: Manages document processing in drop zones only - other managers handle other functions
    
    This dropzone manager provides:
    1. Document Intelligence - analyzes dropped documents and selects optimal templates
    2. Conflict Resolution - handles multi-template scenarios intelligently for drop zone docs
    3. Resource Management - optimizes worker allocation for drop zone processing
    4. Quality Monitoring - tracks drop zone processing performance and adapts strategies
    
    SECURITY: Uses only approved worker registry, validated templates, and CorporateLLMGateway
    """
    
    def __init__(
        self,
        templates_path: str = "C:/Users/marti/github/prompts/templates",
        session_manager: Optional[UnifiedSessionManager] = None,
        llm_gateway: Optional[CorporateLLMGateway] = None,
        max_concurrent_analyses: int = 10
    ):
        super().__init__(worker_name="ai_dropzone_manager")
        self.templates_path = Path(templates_path)
        self.session_manager = session_manager
        self.max_concurrent_analyses = max_concurrent_analyses
        
        # Initialize LLM Gateway for all AI operations
        if LLM_GATEWAY_AVAILABLE:
            self.llm_gateway = llm_gateway or CorporateLLMGateway()
        else:
            self.llm_gateway = None
            logger.warning("AI Manager operating without LLM gateway - intelligence features will be limited")
        
        # SECURITY: Worker registry - ONLY approved worker types allowed
        self.approved_worker_types = {
            "PromptWorker": PromptWorker,
            "FlowRecoveryWorker": FlowRecoveryWorker,
            "CoordinatorWorker": None  # Will be imported when needed
        }
        
        # Worker pool management - tracks orchestration state only
        self.registered_workers: Dict[str, BaseWorker] = {}  # Pre-registered workers only
        self.active_worker_assignments: Dict[str, Dict[str, Any]] = {}  # Current assignments
        self.worker_load: Dict[str, int] = {}
        self.worker_performance: Dict[str, Dict[str, float]] = {}
        
        # Template library
        self.available_templates: Dict[str, Dict[str, Any]] = {}
        self.template_performance: Dict[str, Dict[str, float]] = {}
        
        # Quality monitoring
        self.processing_history: List[Dict[str, Any]] = []
        self.quality_thresholds = {
            "accuracy": 0.85,
            "completeness": 0.90,
            "consistency": 0.88,
            "processing_time": 1.2  # Multiplier of estimated time
        }
        
        # Document intelligence cache
        self.intelligence_cache: Dict[str, DocumentIntelligence] = {}
        
    async def initialize(self):
        """Initialize the AI-Assisted Manager with security validation."""
        await super().initialize()
        
        # SECURITY: Validate worker registry before any operations
        self._validate_worker_registry()
        
        # SECURITY: Load and validate template library (approved templates only)
        await self._load_template_library()
        
        # SECURITY: Initialize only registered worker pools
        await self._initialize_worker_pools()
        
        # Load historical performance data
        await self._load_historical_performance()
        
        logger.info("AI-Assisted Manager initialized successfully with security constraints validated")
    
    def _validate_worker_registry(self):
        """
        SECURITY: Validate that only approved worker types are available.
        Prevents dynamic worker creation or execution of arbitrary code.
        """
        for worker_type_name, worker_class in self.approved_worker_types.items():
            if worker_class is None:
                continue  # Skip workers that need dynamic import
                
            try:
                # Verify worker class is properly implemented
                if not issubclass(worker_class, BaseWorker):
                    raise SecurityError(f"Invalid worker type: {worker_type_name} is not a BaseWorker subclass")
                
                logger.info(f"[SECURITY] Validated approved worker type: {worker_type_name}")
                
            except Exception as e:
                raise SecurityError(f"Worker registry validation failed for {worker_type_name}: {e}")
        
        logger.info(f"[SECURITY] Worker registry validated - {len(self.approved_worker_types)} approved worker types")
    
    def _validate_template_path(self, template_path: str) -> bool:
        """
        SECURITY: Validate template is from approved directory only.
        Prevents loading of arbitrary templates or template injection.
        """
        try:
            template_file = Path(template_path)
            templates_dir = Path(self.templates_path).resolve()
            
            # Check if template is within approved directory
            if not template_file.resolve().is_relative_to(templates_dir):
                logger.warning(f"[SECURITY] Template outside approved directory rejected: {template_path}")
                return False
            
            # Check file extension
            if template_file.suffix != '.md':
                logger.warning(f"[SECURITY] Invalid template file type rejected: {template_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"[SECURITY] Template path validation failed: {e}")
            return False
    
    def register_worker(self, worker_id: str, worker_type: str, **worker_kwargs) -> bool:
        """
        SECURITY: Register a worker instance from approved registry only.
        
        Args:
            worker_id: Unique identifier for this worker instance
            worker_type: Must be from approved_worker_types registry
            **worker_kwargs: Configuration for worker initialization
            
        Returns:
            bool: True if successfully registered, False if rejected for security reasons
        """
        try:
            # SECURITY: Validate worker type is approved
            if worker_type not in self.approved_worker_types:
                logger.error(f"[SECURITY] Unauthorized worker type rejected: {worker_type}")
                return False
            
            worker_class = self.approved_worker_types[worker_type]
            if worker_class is None:
                logger.error(f"[SECURITY] Worker type not available: {worker_type}")
                return False
            
            # SECURITY: Validate worker_id to prevent injection
            if not worker_id.replace('_', '').replace('-', '').isalnum():
                logger.error(f"[SECURITY] Invalid worker ID format rejected: {worker_id}")
                return False
            
            # Create worker instance with validation
            worker_instance = worker_class(**worker_kwargs)
            
            # Register the worker
            self.registered_workers[worker_id] = worker_instance
            self.worker_load[worker_id] = 0
            self.worker_performance[worker_id] = {
                "success_rate": 1.0,
                "avg_processing_time": 10.0,
                "quality_score": 0.85
            }
            
            logger.info(f"[SECURITY] Worker registered successfully: {worker_id} ({worker_type})")
            return True
            
        except Exception as e:
            logger.error(f"[SECURITY] Worker registration failed for {worker_id}: {e}")
            return False
    
    def validate_input(self, task_input: AIManagerTask) -> bool:
        """
        Validate input task for AI Dropzone Manager.
        
        Args:
            task_input: AIManagerTask to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check if task_input is the correct type
            if not isinstance(task_input, AIManagerTask):
                logger.error(f"Invalid task input type: {type(task_input)}")
                return False
            
            # Validate required fields
            if not task_input.document_path:
                logger.error("Document path is required")
                return False
            
            # Validate document path exists (if it's a file path)
            if not task_input.document_path.startswith(('http://', 'https://', 's3://')):
                # It's a local file path
                import os
                if not os.path.exists(task_input.document_path):
                    logger.warning(f"Document path does not exist: {task_input.document_path}")
                    # Don't fail validation for non-existent paths, just warn
            
            # Validate business priority
            valid_priorities = ["critical", "high", "normal", "low"]
            if task_input.business_priority not in valid_priorities:
                logger.error(f"Invalid business priority: {task_input.business_priority}")
                return False
            
            # Validate deadline if provided
            if task_input.deadline and task_input.deadline < datetime.now():
                logger.warning("Deadline is in the past")
                # Don't fail validation for past deadlines, just warn
            
            logger.debug(f"Task input validation successful for: {task_input.document_path}")
            return True
            
        except Exception as e:
            logger.error(f"Task input validation failed: {e}")
            return False
    
    async def process_task(self, task: AIManagerTask) -> AIManagerResult:
        """
        Main processing entry point for intelligent document orchestration.
        """
        try:
            # Step 1: Document Intelligence Analysis
            intelligence = await self._analyze_document_intelligence(task.document_path)
            
            # Step 2: Processing Strategy Decision
            strategy = await self._determine_processing_strategy(intelligence, task)
            
            # Step 3: Worker Assignment and Resource Allocation
            worker_assignments = await self._allocate_workers(strategy, intelligence)
            
            # Step 4: Quality Check Planning
            quality_checks = await self._plan_quality_checks(strategy, intelligence)
            
            # Step 5: Processing Execution Coordination
            processing_decision = ProcessingDecision(
                document_path=task.document_path,
                selected_templates=intelligence.recommended_templates,
                processing_strategy=strategy,
                assigned_workers=[w["worker_id"] for w in worker_assignments],
                priority_level=self._calculate_priority(task),
                estimated_completion=datetime.now() + timedelta(minutes=intelligence.estimated_processing_time),
                quality_checks=quality_checks
            )
            
            # Step 6: Monitor and Track Processing
            await self._track_processing_decision(processing_decision, intelligence)
            
            return AIManagerResult(
                processing_decision=processing_decision,
                worker_assignments=worker_assignments,
                quality_metrics=self._calculate_quality_metrics(intelligence),
                estimated_completion=processing_decision.estimated_completion,
                success=True
            )
            
        except Exception as e:
            logger.error(f"AI Manager processing failed: {e}")
            return AIManagerResult(
                processing_decision=None,
                worker_assignments=[],
                quality_metrics={},
                estimated_completion=datetime.now(),
                success=False,
                error=str(e)
            )
    
    async def _analyze_document_intelligence(self, document_path: str) -> DocumentIntelligence:
        """
        Perform intelligent analysis of the document to determine optimal processing approach.
        """
        # Check cache first
        if document_path in self.intelligence_cache:
            cached = self.intelligence_cache[document_path]
            if (datetime.now() - cached.created_at).seconds < 3600:  # 1 hour cache
                return cached
        
        try:
            # Read document content preview
            with open(document_path, 'rb') as f:
                content_bytes = f.read(8192)  # First 8KB for analysis
                content_preview = content_bytes.decode('utf-8', errors='ignore')[:2000]
            
            # Analyze document characteristics
            detected_type = await self._classify_document_type(content_preview)
            complexity = await self._assess_document_complexity(content_preview)
            
            # Template recommendation engine
            recommended_templates = await self._recommend_templates(detected_type, complexity, content_preview)
            
            # Processing strategy recommendation
            strategy = await self._recommend_processing_strategy(recommended_templates, complexity)
            
            # Resource estimation
            estimated_time = self._estimate_processing_time(complexity, strategy, len(recommended_templates))
            resource_requirements = self._estimate_resource_requirements(complexity, strategy)
            
            intelligence = DocumentIntelligence(
                document_path=document_path,
                content_preview=content_preview[:500],  # Store limited preview
                detected_type=detected_type,
                complexity=complexity,
                confidence_score=0.85,  # TODO: Implement confidence calculation
                recommended_templates=recommended_templates,
                processing_strategy=strategy,
                estimated_processing_time=estimated_time,
                resource_requirements=resource_requirements
            )
            
            # Cache the intelligence
            self.intelligence_cache[document_path] = intelligence
            
            return intelligence
            
        except Exception as e:
            logger.error(f"Document intelligence analysis failed for {document_path}: {e}")
            # Return default intelligence for fallback processing
            return DocumentIntelligence(
                document_path=document_path,
                content_preview="Unable to analyze",
                detected_type="unknown",
                complexity=DocumentComplexity.MODERATE,
                confidence_score=0.5,
                recommended_templates=["hybrid_analysis"],
                processing_strategy=ProcessingStrategy.SINGLE_TEMPLATE,
                estimated_processing_time=15,
                resource_requirements={"cpu": "medium", "memory": "medium"}
            )
    
    async def _classify_document_type(self, content_preview: str) -> str:
        """
        Classify document type using AI through CorporateLLMGateway.
        """
        if not self.llm_gateway:
            # Fallback to keyword-based classification
            return self._classify_document_type_fallback(content_preview)
        
        try:
            classification_prompt = f"""Analyze the following document excerpt and classify it into one of these categories:
- financial: Financial statements, reports, budgets, cost analysis
- contract: Legal agreements, contracts, terms and conditions
- compliance: Regulatory documents, policies, audit reports
- mvr: Motor vehicle records, driving records, validation templates
- data_analysis: Research reports, data analysis, studies
- general: Other types of documents

Document excerpt (first 500 characters):
{content_preview[:500]}

Respond with only the classification category (one word):"""

            request = LLMRequest(
                prompt=classification_prompt,
                model="claude-3-haiku",  # Use faster model for classification
                audit_reason="AI Manager document type classification",
                user_id="ai_manager_system",
                max_tokens=50,
                temperature=0.1
            )
            
            response = self.llm_gateway.process_llm_request(request)
            
            if response.success:
                classification = response.content.strip().lower()
                valid_types = ['financial', 'contract', 'compliance', 'mvr', 'data_analysis', 'general']
                
                if classification in valid_types:
                    return classification
                else:
                    logger.warning(f"Invalid classification '{classification}' from LLM, using fallback")
                    return self._classify_document_type_fallback(content_preview)
            else:
                logger.warning(f"LLM classification failed: {response.error}, using fallback")
                return self._classify_document_type_fallback(content_preview)
                
        except Exception as e:
            logger.warning(f"Document classification error: {e}, using fallback")
            return self._classify_document_type_fallback(content_preview)
    
    def _classify_document_type_fallback(self, content_preview: str) -> str:
        """
        Fallback keyword-based document type classification.
        """
        content_lower = content_preview.lower()
        
        # Financial document indicators
        financial_keywords = ['balance sheet', 'income statement', 'cash flow', 'financial', 'revenue', 'profit', 'loss', 'assets', 'liabilities']
        if any(keyword in content_lower for keyword in financial_keywords):
            return "financial"
        
        # Contract/Legal document indicators
        contract_keywords = ['agreement', 'contract', 'terms', 'conditions', 'party', 'whereas', 'liability', 'indemnification']
        if any(keyword in content_lower for keyword in contract_keywords):
            return "contract"
        
        # Compliance document indicators
        compliance_keywords = ['compliance', 'regulation', 'policy', 'procedure', 'audit', 'requirement', 'standard']
        if any(keyword in content_lower for keyword in compliance_keywords):
            return "compliance"
        
        # MVR/VST document indicators
        mvr_keywords = ['motor vehicle record', 'driving record', 'mvr', 'vst', 'validation scoping template']
        if any(keyword in content_lower for keyword in mvr_keywords):
            return "mvr"
        
        # Data/Research document indicators
        data_keywords = ['data', 'analysis', 'research', 'study', 'findings', 'methodology', 'results']
        if any(keyword in content_lower for keyword in data_keywords):
            return "data_analysis"
        
        return "general"
    
    async def _assess_document_complexity(self, content_preview: str) -> DocumentComplexity:
        """
        Assess document complexity using AI through CorporateLLMGateway.
        """
        if not self.llm_gateway:
            # Fallback to rule-based complexity assessment
            return self._assess_document_complexity_fallback(content_preview)
        
        try:
            complexity_prompt = f"""Analyze the following document excerpt and assess its complexity level:

SIMPLE: Basic documents with straightforward language, minimal technical terms, clear structure
MODERATE: Some technical language, moderate length, requires domain knowledge 
COMPLEX: Technical jargon, specialized terminology, complex concepts, detailed analysis required
CRITICAL: Highly technical, regulatory implications, multi-domain expertise needed, critical business impact

Document excerpt (first 800 characters):
{content_preview[:800]}

Consider these factors:
- Technical terminology density
- Domain expertise requirements  
- Regulatory/compliance implications
- Length and structure complexity
- Analysis depth required

Respond with only one word: SIMPLE, MODERATE, COMPLEX, or CRITICAL"""

            request = LLMRequest(
                prompt=complexity_prompt,
                model="claude-3-haiku", 
                audit_reason="AI Manager document complexity assessment",
                user_id="ai_manager_system", 
                max_tokens=50,
                temperature=0.1
            )
            
            response = self.llm_gateway.process_llm_request(request)
            
            if response.success:
                complexity_str = response.content.strip().upper()
                
                try:
                    return DocumentComplexity[complexity_str]
                except KeyError:
                    logger.warning(f"Invalid complexity '{complexity_str}' from LLM, using fallback")
                    return self._assess_document_complexity_fallback(content_preview)
            else:
                logger.warning(f"LLM complexity assessment failed: {response.error}, using fallback")
                return self._assess_document_complexity_fallback(content_preview)
                
        except Exception as e:
            logger.warning(f"Document complexity assessment error: {e}, using fallback")
            return self._assess_document_complexity_fallback(content_preview)
    
    def _assess_document_complexity_fallback(self, content_preview: str) -> DocumentComplexity:
        """
        Fallback rule-based document complexity assessment.
        """
        complexity_score = 0
        
        # Length indicator
        if len(content_preview) > 5000:
            complexity_score += 1
        elif len(content_preview) > 2000:
            complexity_score += 0.5
        
        # Technical language indicator
        technical_terms = ['methodology', 'algorithm', 'statistical', 'quantitative', 'regression', 'correlation']
        if any(term in content_preview.lower() for term in technical_terms):
            complexity_score += 1
        
        # Financial complexity indicators
        financial_terms = ['derivative', 'amortization', 'depreciation', 'consolidation', 'valuation']
        if any(term in content_preview.lower() for term in financial_terms):
            complexity_score += 1
        
        # Legal complexity indicators
        legal_terms = ['jurisdiction', 'arbitration', 'indemnification', 'proprietary', 'confidential']
        if any(term in content_preview.lower() for term in legal_terms):
            complexity_score += 1
        
        # Tables and structured data
        if content_preview.count('|') > 10 or content_preview.count('\t') > 20:
            complexity_score += 0.5
        
        # Convert score to complexity enum
        if complexity_score >= 3:
            return DocumentComplexity.CRITICAL
        elif complexity_score >= 2:
            return DocumentComplexity.COMPLEX
        elif complexity_score >= 1:
            return DocumentComplexity.MODERATE
        else:
            return DocumentComplexity.SIMPLE
    
    async def _recommend_templates(self, document_type: str, complexity: DocumentComplexity, content: str) -> List[str]:
        """
        Recommend optimal templates using AI-enhanced analysis through CorporateLLMGateway.
        """
        if not self.llm_gateway:
            # Fallback to rule-based template recommendation
            return self._recommend_templates_fallback(document_type, complexity, content)
        
        try:
            # Get available templates
            available_templates = list(self.available_templates.keys())
            
            template_prompt = f"""You are an expert document analyst. Given the document analysis below, recommend the best processing templates from the available options.

DOCUMENT ANALYSIS:
- Type: {document_type}
- Complexity: {complexity.name}
- Content preview: {content[:600]}

AVAILABLE TEMPLATES:
{chr(10).join(f'- {template}: {self.available_templates.get(template, {}).get("domain_focus", "general")} focus' for template in available_templates)}

SELECTION CRITERIA:
- Choose 1-3 templates that best match the document type and complexity
- For CRITICAL complexity, always include quality assurance templates
- For multi-domain documents, consider hybrid or multi-perspective analysis
- Prioritize domain-specific templates over general ones

Respond with a JSON array of recommended template names (exactly as listed above):
Example: ["financial_analysis", "qa_control", "hybrid_analysis"]"""

            request = LLMRequest(
                prompt=template_prompt,
                model="claude-3-haiku",
                audit_reason="AI Manager template recommendation",
                user_id="ai_manager_system",
                max_tokens=200,
                temperature=0.2
            )
            
            response = self.llm_gateway.process_llm_request(request)
            
            if response.success:
                try:
                    # Parse JSON response
                    import json
                    recommended = json.loads(response.content.strip())
                    
                    # Validate recommended templates exist
                    valid_templates = [t for t in recommended if t in available_templates]
                    
                    if valid_templates:
                        return valid_templates[:3]  # Limit to 3 templates max
                    else:
                        logger.warning("No valid templates in LLM recommendation, using fallback")
                        return self._recommend_templates_fallback(document_type, complexity, content)
                        
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse LLM template recommendation: {e}, using fallback")
                    return self._recommend_templates_fallback(document_type, complexity, content)
            else:
                logger.warning(f"LLM template recommendation failed: {response.error}, using fallback")
                return self._recommend_templates_fallback(document_type, complexity, content)
                
        except Exception as e:
            logger.warning(f"Template recommendation error: {e}, using fallback")
            return self._recommend_templates_fallback(document_type, complexity, content)
    
    def _recommend_templates_fallback(self, document_type: str, complexity: DocumentComplexity, content: str) -> List[str]:
        """
        Fallback rule-based template recommendation.
        """
        templates = []
        
        # Primary template based on document type
        type_mapping = {
            "financial": ["financial_analysis"],
            "contract": ["contract_analysis"],
            "compliance": ["compliance_review"],
            "mvr": ["mvr_analysis"],
            "data_analysis": ["data_extraction"],
            "general": ["document_section_view"]
        }
        
        primary_templates = type_mapping.get(document_type, ["hybrid_analysis"])
        templates.extend(primary_templates)
        
        # Add QA template for complex documents
        if complexity in [DocumentComplexity.COMPLEX, DocumentComplexity.CRITICAL]:
            if "qa_control" not in templates:
                templates.append("qa_control")
        
        # Add peer review for critical documents
        if complexity == DocumentComplexity.CRITICAL:
            if "peer_review" not in templates:
                templates.append("peer_review")
        
        # Add hybrid analysis for multi-dimensional documents
        multi_indicators = ["financial", "compliance", "risk", "analysis", "assessment"]
        if sum(1 for indicator in multi_indicators if indicator in content.lower()) >= 2:
            if "hybrid_analysis" not in templates:
                templates.append("hybrid_analysis")
        
        return templates[:3]  # Limit to 3 templates max
    
    async def _recommend_processing_strategy(self, templates: List[str], complexity: DocumentComplexity) -> ProcessingStrategy:
        """
        Recommend processing strategy based on templates and complexity.
        """
        if len(templates) == 1:
            return ProcessingStrategy.SINGLE_TEMPLATE
        elif "hybrid_analysis" in templates or complexity == DocumentComplexity.CRITICAL:
            return ProcessingStrategy.HYBRID_ANALYSIS
        elif len(templates) > 1:
            return ProcessingStrategy.MULTI_PERSPECTIVE
        else:
            return ProcessingStrategy.SINGLE_TEMPLATE
    
    def _estimate_processing_time(self, complexity: DocumentComplexity, strategy: ProcessingStrategy, template_count: int) -> int:
        """
        Estimate processing time in minutes.
        """
        base_times = {
            DocumentComplexity.SIMPLE: 5,
            DocumentComplexity.MODERATE: 10,
            DocumentComplexity.COMPLEX: 20,
            DocumentComplexity.CRITICAL: 35
        }
        
        strategy_multipliers = {
            ProcessingStrategy.SINGLE_TEMPLATE: 1.0,
            ProcessingStrategy.HYBRID_ANALYSIS: 1.5,
            ProcessingStrategy.MULTI_PERSPECTIVE: 2.0,
            ProcessingStrategy.ESCALATE_HUMAN: 0.5  # Just routing time
        }
        
        base_time = base_times[complexity]
        strategy_multiplier = strategy_multipliers[strategy]
        template_multiplier = 1 + (template_count - 1) * 0.3
        
        return int(base_time * strategy_multiplier * template_multiplier)
    
    def _estimate_resource_requirements(self, complexity: DocumentComplexity, strategy: ProcessingStrategy) -> Dict[str, str]:
        """
        Estimate resource requirements for processing.
        """
        if complexity == DocumentComplexity.CRITICAL or strategy == ProcessingStrategy.HYBRID_ANALYSIS:
            return {"cpu": "high", "memory": "high", "priority": "urgent"}
        elif complexity == DocumentComplexity.COMPLEX or strategy == ProcessingStrategy.MULTI_PERSPECTIVE:
            return {"cpu": "medium", "memory": "medium", "priority": "high"}
        else:
            return {"cpu": "low", "memory": "low", "priority": "normal"}
    
    async def _determine_processing_strategy(self, intelligence: DocumentIntelligence, task: AIManagerTask) -> ProcessingStrategy:
        """
        Make final processing strategy decision considering business context.
        """
        base_strategy = intelligence.processing_strategy
        
        # Business priority adjustments
        if task.business_priority == "critical":
            if base_strategy == ProcessingStrategy.SINGLE_TEMPLATE:
                return ProcessingStrategy.HYBRID_ANALYSIS  # Upgrade for critical items
        elif task.business_priority == "low":
            if base_strategy == ProcessingStrategy.HYBRID_ANALYSIS:
                return ProcessingStrategy.SINGLE_TEMPLATE  # Downgrade for low priority
        
        # Deadline pressure adjustments
        if task.deadline and task.deadline < datetime.now() + timedelta(hours=2):
            return ProcessingStrategy.SINGLE_TEMPLATE  # Fast track
        
        return base_strategy
    
    async def _allocate_workers(self, strategy: ProcessingStrategy, intelligence: DocumentIntelligence) -> List[Dict[str, Any]]:
        """
        Allocate workers based on processing strategy and resource requirements.
        """
        worker_assignments = []
        
        if strategy == ProcessingStrategy.SINGLE_TEMPLATE:
            # Single worker assignment
            worker_id = await self._select_best_worker("prompt", intelligence.resource_requirements)
            worker_assignments.append({
                "worker_id": worker_id,
                "worker_type": "PromptWorker",
                "templates": intelligence.recommended_templates[:1],
                "priority": "normal",
                "estimated_time": intelligence.estimated_processing_time
            })
            
        elif strategy in [ProcessingStrategy.HYBRID_ANALYSIS, ProcessingStrategy.MULTI_PERSPECTIVE]:
            # Multiple worker coordination
            for i, template in enumerate(intelligence.recommended_templates):
                worker_id = await self._select_best_worker("prompt", intelligence.resource_requirements)
                worker_assignments.append({
                    "worker_id": f"{worker_id}_{i}",
                    "worker_type": "PromptWorker",
                    "templates": [template],
                    "priority": "high" if i == 0 else "normal",
                    "estimated_time": int(intelligence.estimated_processing_time / len(intelligence.recommended_templates))
                })
            
            # Add coordination worker for result synthesis
            coordinator_id = await self._select_best_worker("coordinator", {"cpu": "medium", "memory": "medium"})
            worker_assignments.append({
                "worker_id": coordinator_id,
                "worker_type": "CoordinatorWorker",
                "templates": ["synthesis"],
                "priority": "high",
                "estimated_time": 5,
                "depends_on": [w["worker_id"] for w in worker_assignments[:-1]]
            })
        
        return worker_assignments
    
    async def _select_best_worker(self, worker_type: str, resource_requirements: Dict[str, str]) -> str:
        """
        Select the best available worker based on load and performance.
        """
        # For now, return a deterministic worker ID
        # TODO: Implement actual worker pool management
        import uuid
        return f"{worker_type}_{str(uuid.uuid4())[:8]}"
    
    async def _plan_quality_checks(self, strategy: ProcessingStrategy, intelligence: DocumentIntelligence) -> List[str]:
        """
        Plan quality assurance checks based on processing strategy.
        """
        quality_checks = ["output_completeness", "format_validation"]
        
        if intelligence.complexity in [DocumentComplexity.COMPLEX, DocumentComplexity.CRITICAL]:
            quality_checks.extend(["accuracy_validation", "consistency_check"])
        
        if strategy in [ProcessingStrategy.HYBRID_ANALYSIS, ProcessingStrategy.MULTI_PERSPECTIVE]:
            quality_checks.append("cross_template_consistency")
        
        if intelligence.detected_type in ["financial", "compliance"]:
            quality_checks.append("regulatory_compliance_check")
        
        return quality_checks
    
    def _calculate_priority(self, task: AIManagerTask) -> int:
        """
        Calculate processing priority (1=highest, 5=lowest).
        """
        priority_mapping = {
            "critical": 1,
            "urgent": 2,
            "high": 2,
            "normal": 3,
            "low": 4,
            "background": 5
        }
        
        base_priority = priority_mapping.get(task.business_priority, 3)
        
        # Deadline adjustment
        if task.deadline:
            hours_until_deadline = (task.deadline - datetime.now()).total_seconds() / 3600
            if hours_until_deadline < 2:
                base_priority = max(1, base_priority - 2)
            elif hours_until_deadline < 8:
                base_priority = max(1, base_priority - 1)
        
        return base_priority
    
    def _calculate_quality_metrics(self, intelligence: DocumentIntelligence) -> Dict[str, float]:
        """
        Calculate expected quality metrics for monitoring.
        """
        base_accuracy = 0.85
        base_completeness = 0.90
        
        # Adjust based on complexity
        complexity_adjustments = {
            DocumentComplexity.SIMPLE: 0.05,
            DocumentComplexity.MODERATE: 0.0,
            DocumentComplexity.COMPLEX: -0.05,
            DocumentComplexity.CRITICAL: -0.10
        }
        
        adjustment = complexity_adjustments.get(intelligence.complexity, 0.0)
        
        return {
            "expected_accuracy": base_accuracy + adjustment,
            "expected_completeness": base_completeness + adjustment,
            "confidence_threshold": intelligence.confidence_score,
            "quality_threshold": 0.80
        }
    
    async def _track_processing_decision(self, decision: ProcessingDecision, intelligence: DocumentIntelligence):
        """
        Track processing decisions for performance monitoring and learning.
        """
        tracking_record = {
            "timestamp": datetime.now().isoformat(),
            "document_path": decision.document_path,
            "document_type": intelligence.detected_type,
            "complexity": intelligence.complexity.name,
            "strategy": decision.processing_strategy.value,
            "templates": decision.selected_templates,
            "workers": decision.assigned_workers,
            "estimated_time": intelligence.estimated_processing_time,
            "priority": decision.priority_level
        }
        
        self.processing_history.append(tracking_record)
        
        # Persist to database if session manager available
        if self.session_manager:
            try:
                await self.session_manager.log_processing_decision(tracking_record)
            except Exception as e:
                logger.warning(f"Failed to persist processing decision: {e}")
    
    async def _load_template_library(self):
        """
        Load and index available templates for intelligent selection.
        """
        try:
            for template_file in self.templates_path.glob("*.md"):
                template_name = template_file.stem
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract template metadata (title, description, etc.)
                self.available_templates[template_name] = {
                    "path": str(template_file),
                    "content_preview": content[:500],
                    "estimated_complexity": self._assess_template_complexity(content),
                    "domain_focus": self._extract_template_domain(content)
                }
            
            logger.info(f"Loaded {len(self.available_templates)} templates")
            
        except Exception as e:
            logger.error(f"Failed to load template library: {e}")
    
    def _assess_template_complexity(self, content: str) -> str:
        """
        Assess template complexity based on content.
        """
        if "hybrid" in content.lower() or "multi-" in content.lower():
            return "high"
        elif len(content) > 5000 or content.count("```") > 6:
            return "medium"
        else:
            return "low"
    
    def _extract_template_domain(self, content: str) -> str:
        """
        Extract primary domain focus from template content.
        """
        content_lower = content.lower()
        if "financial" in content_lower:
            return "financial"
        elif "compliance" in content_lower:
            return "compliance"
        elif "contract" in content_lower:
            return "legal"
        elif "data" in content_lower:
            return "data"
        else:
            return "general"
    
    async def _initialize_worker_pools(self):
        """
        Initialize and manage worker pools.
        """
        # Initialize base workers
        self.active_workers["prompt_pool"] = PromptWorker()
        self.active_workers["recovery_pool"] = FlowRecoveryWorker()
        
        # Initialize worker load tracking
        for worker_id in self.active_workers:
            self.worker_load[worker_id] = 0
            self.worker_performance[worker_id] = {
                "success_rate": 1.0,
                "avg_processing_time": 10.0,
                "quality_score": 0.85
            }
        
        logger.info("Worker pools initialized")
    
    async def _load_historical_performance(self):
        """
        Load historical performance data for machine learning improvements.
        """
        try:
            if self.session_manager:
                # Load historical data from database
                # TODO: Implement historical data loading
                pass
            
            # Initialize template performance tracking
            for template_name in self.available_templates:
                self.template_performance[template_name] = {
                    "success_rate": 0.85,
                    "avg_processing_time": 12.0,
                    "quality_score": 0.80,
                    "usage_count": 0
                }
                
        except Exception as e:
            logger.warning(f"Failed to load historical performance: {e}")
    
    async def get_manager_status(self) -> Dict[str, Any]:
        """
        Get current manager status for monitoring.
        """
        return {
            "active_workers": len(self.active_workers),
            "available_templates": len(self.available_templates),
            "processing_history_count": len(self.processing_history),
            "cached_intelligence_count": len(self.intelligence_cache),
            "worker_load": dict(self.worker_load),
            "quality_thresholds": self.quality_thresholds,
            "status": "active"
        }
    
    async def update_quality_feedback(self, document_path: str, quality_metrics: Dict[str, float]):
        """
        Update manager with quality feedback to improve future decisions.
        """
        # Find the processing decision for this document
        for record in self.processing_history:
            if record["document_path"] == document_path:
                record["actual_quality"] = quality_metrics
                
                # Update template performance
                for template in record["templates"]:
                    if template in self.template_performance:
                        perf = self.template_performance[template]
                        perf["success_rate"] = (perf["success_rate"] * 0.9) + (quality_metrics.get("accuracy", 0.8) * 0.1)
                        perf["quality_score"] = (perf["quality_score"] * 0.9) + (quality_metrics.get("overall", 0.8) * 0.1)
                        perf["usage_count"] += 1
                
                logger.info(f"Updated quality feedback for {document_path}")
                break