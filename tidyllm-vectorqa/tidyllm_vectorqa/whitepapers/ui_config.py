"""
UI Configuration and Query Management
====================================

Central place for managing all domain queries, UI settings, and improvement ideas.
This makes it easy to update queries, add new domains, and track UI enhancement ideas.
"""

# =============================================================================
# DOMAIN RESEARCH QUERIES
# =============================================================================

DOMAIN_QUERIES = {
    "Context Engineering": {
        "icon": "🔧",
        "description": "Prompt engineering, RAG, context management, and information quality control",
        "queries": [
            "prompt engineering techniques optimization",
            "retrieval augmented generation methods", 
            "context collapse mitigation strategies",
            "information quality control systems",
            "contextual relevance evaluation metrics",
            "few shot learning prompt design",
            "chain of thought reasoning techniques",
            "in context learning optimization"
        ]
    },
    "Machine Learning": {
        "icon": "🤖",
        "description": "Neural networks, deep learning, and AI algorithms",
        "queries": [
            "deep neural networks classification",
            "transformer architecture attention mechanism",
            "reinforcement learning optimization",
            "generative adversarial networks training",
            "convolutional neural networks computer vision",
            "recurrent neural networks sequence modeling",
            "self supervised learning methods"
        ]
    },
    "Signal Processing": {
        "icon": "📡",
        "description": "Digital signal processing, filtering, and frequency analysis",
        "queries": [
            "signal decomposition noise reduction",
            "frequency domain filtering techniques", 
            "wavelet transform applications",
            "digital signal processing algorithms",
            "fourier transform signal analysis",
            "adaptive filtering methods",
            "spectral analysis techniques"
        ]
    },
    "Quantum Physics": {
        "icon": "⚛️",
        "description": "Quantum mechanics, computing, and information theory",
        "queries": [
            "quantum entanglement measurement",
            "quantum mechanics superposition",
            "quantum computing algorithms",
            "quantum error correction methods",
            "quantum information theory",
            "quantum machine learning",
            "quantum cryptography protocols"
        ]
    },
    "Computer Science": {
        "icon": "💻",
        "description": "Algorithms, software engineering, and distributed systems",
        "queries": [
            "algorithm complexity analysis",
            "distributed systems architecture",
            "software engineering patterns",
            "database optimization techniques",
            "concurrent programming methods",
            "data structures performance analysis",
            "microservices architecture design"
        ]
    },
    "Mathematics": {
        "icon": "📐",
        "description": "Mathematical optimization, algebra, and probability theory",
        "queries": [
            "mathematical optimization methods",
            "linear algebra decomposition",
            "probability theory applications",
            "numerical methods convergence",
            "graph theory algorithms",
            "combinatorial optimization",
            "statistical inference methods"
        ]
    },
    "Biomedical": {
        "icon": "🧬",
        "description": "Medical imaging, biomarkers, and clinical research",
        "queries": [
            "medical image analysis",
            "biomarker discovery methods",
            "clinical trial analysis",
            "genomics data processing",
            "drug discovery computational methods",
            "bioinformatics algorithms",
            "precision medicine approaches"
        ]
    },
    "Natural Language Processing": {
        "icon": "🗣️", 
        "description": "Language models, text processing, and computational linguistics",
        "queries": [
            "large language model training",
            "natural language understanding",
            "text classification methods",
            "named entity recognition",
            "sentiment analysis techniques",
            "machine translation quality",
            "language model evaluation metrics"
        ]
    }
}

# =============================================================================
# DOMAIN CLASSIFICATION KEYWORDS
# =============================================================================

DOMAIN_KEYWORDS = {
    'Context Engineering': [
        'context', 'prompt', 'engineering', 'rag', 'retrieval', 'augmented', 'generation', 
        'llm', 'language', 'model', 'contextual', 'collapse', 'quality', 'control', 
        'information', 'curation', 'few-shot', 'in-context', 'chain-of-thought'
    ],
    'Machine Learning': [
        'neural', 'learning', 'ai', 'artificial', 'deep', 'machine', 'algorithm',
        'training', 'model', 'network', 'classification', 'regression', 'supervised',
        'unsupervised', 'reinforcement', 'generative', 'adversarial'
    ],
    'Signal Processing': [
        'signal', 'processing', 'filter', 'frequency', 'audio', 'image', 'fourier',
        'wavelet', 'transform', 'spectral', 'digital', 'analog', 'noise', 'filtering'
    ],
    'Quantum Physics': [
        'quantum', 'physics', 'particle', 'entanglement', 'mechanics', 'superposition',
        'computing', 'qubit', 'interference', 'measurement', 'uncertainty', 'wave'
    ],
    'Mathematics': [
        'mathematical', 'theorem', 'proof', 'algebra', 'geometry', 'calculus',
        'optimization', 'matrix', 'vector', 'differential', 'integral', 'topology'
    ],
    'Computer Science': [
        'computer', 'software', 'programming', 'algorithm', 'data', 'structure',
        'distributed', 'concurrent', 'parallel', 'complexity', 'database', 'system'
    ],
    'Biology/Medicine': [
        'bio', 'medical', 'health', 'disease', 'clinical', 'gene', 'protein',
        'genomics', 'bioinformatics', 'drug', 'therapy', 'diagnosis', 'treatment'
    ],
    'Statistics': [
        'statistical', 'probability', 'regression', 'analysis', 'data', 'inference',
        'hypothesis', 'distribution', 'sampling', 'correlation', 'significance'
    ],
    'Natural Language Processing': [
        'nlp', 'text', 'language', 'linguistic', 'parsing', 'tokenization', 
        'embedding', 'sentiment', 'translation', 'generation', 'understanding'
    ]
}

# =============================================================================
# UI IMPROVEMENT IDEAS AND FEATURES
# =============================================================================

UI_IMPROVEMENTS = {
    "search_interface": {
        "priority": "high",
        "ideas": [
            "Add search filters by date range, author, journal",
            "Implement search history with favorites",
            "Add advanced search with boolean operators",
            "Include search suggestions and autocomplete",
            "Add search export functionality (CSV, JSON, BibTeX)"
        ]
    },
    "paper_display": {
        "priority": "high", 
        "ideas": [
            "Add paper comparison view (side-by-side)",
            "Implement paper tagging and categorization",
            "Add paper notes and annotations",
            "Include citation network visualization",
            "Show related papers recommendations"
        ]
    },
    "analysis_dashboard": {
        "priority": "medium",
        "ideas": [
            "Add interactive Y=R+S+N score charts",
            "Implement trend analysis over time",
            "Add domain-specific quality metrics",
            "Include research impact predictions",
            "Add collaboration network analysis"
        ]
    },
    "data_management": {
        "priority": "medium",
        "ideas": [
            "Add data export/import functionality",
            "Implement search result caching",
            "Add offline mode for cached results",
            "Include backup and restore features",
            "Add data visualization tools"
        ]
    },
    "user_experience": {
        "priority": "high",
        "ideas": [
            "Add dark/light theme toggle",
            "Implement keyboard shortcuts",
            "Add mobile-responsive design",
            "Include tutorial and onboarding",
            "Add customizable dashboard layouts"
        ]
    },
    "integration": {
        "priority": "low",
        "ideas": [
            "Add integration with reference managers (Zotero, Mendeley)",
            "Implement Google Scholar API integration",
            "Add Slack/Discord notifications for new results",
            "Include email alerts for search updates",
            "Add API endpoints for external access"
        ]
    }
}

# =============================================================================
# UI CONFIGURATION SETTINGS
# =============================================================================

UI_CONFIG = {
    "app_title": "Y=R+S+N Research Quality Analytics",
    "app_icon": "🔬",
    "sidebar_width": 300,
    "max_papers_per_search": 50,
    "default_search_source": "ArXiv",
    "default_search_limit": 10,
    "theme": {
        "primary_color": "#1f77b4",
        "background_color": "#ffffff", 
        "secondary_background_color": "#f0f2f6",
        "text_color": "#262730"
    },
    "pagination": {
        "papers_per_page": 10,
        "search_history_per_page": 20
    },
    "cache_settings": {
        "ttl_seconds": 3600,  # 1 hour
        "max_entries": 1000
    }
}

# =============================================================================
# SEARCH QUALITY METRICS
# =============================================================================

QUALITY_THRESHOLDS = {
    "excellent": {"y_score": 0.8, "context_risk": 0.3},
    "good": {"y_score": 0.7, "context_risk": 0.4}, 
    "fair": {"y_score": 0.6, "context_risk": 0.5},
    "poor": {"y_score": 0.5, "context_risk": 0.6}
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_domain_info(domain_name):
    """Get complete domain information including queries and keywords"""
    return {
        "queries": DOMAIN_QUERIES.get(domain_name, {}).get("queries", []),
        "keywords": DOMAIN_KEYWORDS.get(domain_name, []),
        "icon": DOMAIN_QUERIES.get(domain_name, {}).get("icon", "📚"),
        "description": DOMAIN_QUERIES.get(domain_name, {}).get("description", "")
    }

def get_all_domains():
    """Get list of all available domains"""
    return list(DOMAIN_QUERIES.keys())

def get_improvement_ideas(category=None):
    """Get UI improvement ideas, optionally filtered by category"""
    if category:
        return UI_IMPROVEMENTS.get(category, {})
    return UI_IMPROVEMENTS

def get_quality_label(y_score, context_risk):
    """Determine quality label based on Y score and context risk"""
    for label, thresholds in QUALITY_THRESHOLDS.items():
        if y_score >= thresholds["y_score"] and context_risk <= thresholds["context_risk"]:
            return label
    return "poor"

# =============================================================================
# QUERY SUGGESTIONS
# =============================================================================

TRENDING_QUERIES = [
    "large language model alignment",
    "diffusion model image generation", 
    "graph neural network applications",
    "federated learning privacy",
    "transformer model efficiency",
    "multimodal learning techniques",
    "causal inference methods",
    "neural architecture search",
    "explainable AI techniques",
    "contrastive learning methods"
]

RESEARCH_TEMPLATES = {
    "literature_review": "systematic review {topic} methodology",
    "empirical_study": "{topic} experimental evaluation metrics",
    "theoretical_analysis": "theoretical framework {topic} complexity",
    "application_study": "{topic} real world applications case study",
    "survey_paper": "comprehensive survey {topic} recent advances"
}