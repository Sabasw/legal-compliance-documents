from pydantic_settings import BaseSettings
from typing import List, Dict, Optional, Any
import os
from datetime import timedelta
from cryptography.fernet import Fernet

class Settings(BaseSettings):
    # Environment Configuration
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    SQL_DEBUG: bool = False
    
    # Application Metadata
    APP_NAME: str = "Australian Legal Compliance Analyzer"
    VERSION: str = "4.0"
    API_V1_STR: str = "/api/v1"
    
    # Blockchain Key Encryption
    BLOCKCHAIN_KEY_ENCRYPTION_KEY: str = os.getenv(
        "BLOCKCHAIN_KEY_ENCRYPTION_KEY", 
        Fernet.generate_key().decode()  # Generate a proper Fernet key
    )
    
    # Document Processing Configuration
    MAX_TEXT_LENGTH: int = 100000
    TOP_K_RULES: int = 7
    SUMMARY_LENGTH: int = 600
    TEMPERATURE: float = 0.2
    MAX_KB_DISTANCE: float = 1.5
    PREDICTIVE_MODEL_THRESHOLD: float = 0.7
    MIN_CONFIDENCE: float = 0.6
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    
    # Document Types and Processing
    DOCUMENT_TYPES: List[str] = ["contract", "court_ruling", "regulatory_filing", "policy", "unknown"]
    SUPPORTED_IMAGE_FORMATS: List[str] = [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
    SUPPORTED_DOC_FORMATS: List[str] = [".pdf", ".docx", ".txt"]
    
    # File Storage Paths
    UPLOAD_DIR: str = "uploads"
    REPORT_DIR: str = "reports"
    KNOWLEDGE_DIR: str = "knowledge_docs"
    FAISS_INDEX_PATH: str = "compliance_index.faiss"
    FAISS_CHUNKS_PATH: str = "chunks.txt"
    
    # OCR Configuration
    OCR_LANGUAGES: List[str] = ["en"]
    OCR_GPU: bool = True
    OCR_DETAIL_LEVEL: int = 1  # 0=fast, 1=balanced, 2=best quality
    OCR_MIN_TEXT_CONFIDENCE: float = 0.6
    OCR_DETAILED_OUTPUT: bool = True
    OCR_PARAGRAPH: bool = True
    OCR_ROTATE_PAGES: bool = True
    
    # Compliance Labels
    COMPLIANCE_LABELS: Dict[str, str] = {
        "COMPLIANT": "✅ COMPLIANT",
        "NEEDS_REVIEW": "⚠️ NEEDS REVIEW", 
        "NON_COMPLIANT": "❌ NON-COMPLIANT"
    }
    
    # Risk Levels
    RISK_LEVELS: List[str] = ["Low", "Medium", "High", "Critical", "Unknown"]
    
    # Patterns and Matching
    COURT_PATTERNS: List[str] = [
        r'\[20\d{2}\] [A-Z]+ \d+',
        r'\b\d{4} [A-Z]+ \d+\b', 
        r'\b[A-Z]{2,3} \d+\b'
    ]
    
    # Database Configuration
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres123")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5433")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "legal_compliance")
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    
    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_CACHE_TTL: int = 3600  # 1 hour
    
    # Security Configuration
    SECRET_KEY: str = os.getenv("SECRET_KEY", "d62a6f2ae074f19fc910614cb2a44ad0710e0f875024a6706feb9e58ad1b902d")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 3600
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "2decbda5eec47a6c7cfa8c81bc4e8618106042c297d2e100d0cfdb8350803747")
    JWT_REFRESH_SECRET_KEY: str = os.getenv("JWT_REFRESH_SECRET_KEY", "d600bee8cac3c763b43f41c835a868caacff0311290a8997d5a2203c2793ca53")
    
    # Rate Limiting
    RATE_LIMIT_DEFAULT: int = 100  # requests per minute
    RATE_LIMIT_UPLOAD: int = 10    # uploads per minute
    RATE_LIMIT_ANALYSIS: int = 30  # analysis requests per minute
    RATE_LIMIT_KB: int = 50        # knowledge base requests per minute
    
    # Security Headers
    ALLOWED_HOSTS: List[str] = ["*"]  # In production, specify actual hosts
    CORS_ORIGINS: List[str] = ["*"]   # In production, specify actual origins
    CORS_METHODS: List[str] = ["*"]
    CORS_HEADERS: List[str] = ["*"]
    CORS_CREDENTIALS: bool = True
    CORS_MAX_AGE: int = 600
    
    # External Services
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # Stripe Configuration (for billing)
    STRIPE_SECRET_KEY: str = os.getenv("STRIPE_SECRET_KEY", "")
    STRIPE_PUBLISHABLE_KEY: str = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
    STRIPE_WEBHOOK_SECRET: str = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    
    # Elasticsearch Configuration (for semantic search)
    ELASTICSEARCH_HOST: str = os.getenv("ELASTICSEARCH_HOST", "localhost")
    ELASTICSEARCH_PORT: int = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
    ELASTICSEARCH_INDEX_NAME: str = "legal_documents"
    
    # Blockchain Configuration (Enhanced)
    BLOCKCHAIN_NETWORK: str = os.getenv("BLOCKCHAIN_NETWORK", "sepolia")  # For testing
    BLOCKCHAIN_PRIVATE_KEY: str = os.getenv("BLOCKCHAIN_PRIVATE_KEY", "")
    BLOCKCHAIN_ENABLE_AUDIT: bool = os.getenv("BLOCKCHAIN_ENABLE_AUDIT", "true").lower() == "true"
    
    # AI Model Configuration
    AI_MODEL_CACHE_DIR: str = "ai_models"
    AI_PREDICTION_THRESHOLD: float = 0.75
    AI_MAX_INPUT_LENGTH: int = 8192
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")
    
    # Anonymization Configuration
    ANONYMIZATION_CONFIDENCE_THRESHOLD: float = 0.8
    ANONYMIZATION_PRESERVE_STRUCTURE: bool = True
    ANONYMIZATION_ENTITY_TYPES: List[str] = [
        "PERSON", "ORG", "LOCATION", "PHONE_NUMBER", "EMAIL", 
        "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS", "CRYPTO"
    ]
    
    # 2FA Configuration
    TWO_FACTOR_ISSUER: str = "Legal Compliance API"
    
    # Blockchain Configuration
    BLOCKCHAIN_PROVIDER_URL: str = "https://mainnet.infura.io/v3/6e44c09524a846c1b6b933d2bd456f8a"
    BLOCKCHAIN_CONTRACT_ADDRESS: str = "0xYourContractAddress"
    BLOCKCHAIN_GAS_LIMIT: int = 500000
    BLOCKCHAIN_GAS_PRICE: int = 20  # gwei
    BLOCKCHAIN_MAX_RETRIES: int = 3
    BLOCKCHAIN_CONFIRMATION_BLOCKS: int = 2
    
    # Monetization Configuration
    MONETIZATION_PRICING_TIERS: Dict[str, Dict[str, float]] = {
        "basic": {"per_doc": 0.10, "monthly": 9.99},
        "professional": {"per_doc": 0.25, "monthly": 24.99},
        "enterprise": {"per_doc": 0.50, "monthly": 49.99}
    }
    MONETIZATION_CURRENCY: str = "USD"
    MONETIZATION_MINIMUM_CHARGE: float = 1.00
    MONETIZATION_DAILY_LIMITS: Dict[str, int] = {
        "basic": 50,
        "professional": 200,
        "enterprise": 1000
    }
    
    # Jurisdictions Configuration
    JURISDICTIONS: Dict[str, Dict[str, Any]] = {
        "AU": {
            "name": "Australia",
            "currency": "AUD",
            "legal_system": "common_law",
            "timezone": "Australia/Sydney",
            "compliance_rules": {
                "privacy": "Privacy Act 1988",
                "consumer": "Australian Consumer Law"
            }
        },
        "UK": {
            "name": "United Kingdom",
            "currency": "GBP",
            "legal_system": "common_law",
            "timezone": "Europe/London",
            "compliance_rules": {
                "privacy": "GDPR",
                "consumer": "Consumer Rights Act 2015"
            }
        },
        "US": {
            "name": "United States",
            "currency": "USD",
            "legal_system": "common_law",
            "timezone": "America/New_York",
            "compliance_rules": {
                "privacy": "CCPA",
                "consumer": "Uniform Commercial Code"
            }
        },
        "EU": {
            "name": "European Union",
            "currency": "EUR",
            "legal_system": "civil_law",
            "timezone": "Europe/Brussels",
            "compliance_rules": {
                "privacy": "GDPR",
                "consumer": "EU Consumer Rights Directive"
            }
        }
    }
    
    # Celery Configuration
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_RESULT_SERIALIZER: str = "json"
    CELERY_ACCEPT_CONTENT: List[str] = ["json"]
    CELERY_TASK_ROUTES: Dict[str, Dict[str, str]] = {
        "update_knowledge_base": {"queue": "updates"},
        "process_document": {"queue": "analysis"}
    }
    
    # Scheduled Tasks Configuration
    SCHEDULED_TASKS_RULE_UPDATES_TIME: str = "03:00"  # Daily at 3 AM
    SCHEDULED_TASKS_RULE_UPDATES_TIMEZONE: str = "UTC"
    
    # Document Type Prompts
    DOC_TYPE_PROMPTS: Dict[str, Dict[str, Any]] = {
        "contract": {
            "focus_areas": [
                "Missing essential clauses",
                "Ambiguous terms",
                "Australian Consumer Law compliance",
                "Contract law requirements",
                "Blockchain audit trails",
                "RBAC provisions",
                "Predictive dispute analysis"
            ],
            "examples": [
                "Corporations Act 2001 (Cth) s 1337H",
                "Competition and Consumer Act 2010 (Cth)",
                "Electronic Transactions Act 1999 (Cth)"
            ]
        },
        "court_ruling": {
            "focus_areas": [
                "Judicial reasoning",
                "Statutory interpretation",
                "Precedent alignment",
                "Jurisdictional issues",
                "Evidence handling",
                "Court procedures",
                "Predictive outcome analysis"
            ],
            "examples": [
                "Evidence Act 1995 (Cth) s 55",
                "Judiciary Act 1903 (Cth)",
                "Uniform Evidence Legislation"
            ]
        },
        "regulatory_filing": {
            "focus_areas": [
                "Disclosure completeness",
                "Reporting accuracy",
                "Timeliness requirements",
                "ASIC/APRA compliance",
                "Audit trails",
                "Data governance",
                "Regulatory risk assessment"
            ],
            "examples": [
                "Corporations Act 2001 (Cth) s 295A",
                "Australian Securities and Investments Commission Act 2001 (Cth)",
                "Anti-Money Laundering and Counter-Terrorism Financing Act 2006 (Cth)"
            ]
        },
        "policy": {
            "focus_areas": [
                "Policy currency",
                "Compliance risks",
                "Workplace health and safety",
                "Privacy compliance",
                "Access controls",
                "Policy enforcement",
                "Impact analysis"
            ],
            "examples": [
                "Privacy Act 1988 (Cth) s 6",
                "Fair Work Act 2009 (Cth)",
                "Security of Critical Infrastructure Act 2018 (Cth)"
            ]
        },
        "unknown": {
            "focus_areas": ["General legal compliance"],
            "examples": []
        }
    }
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"

# Initialize settings
settings = Settings()