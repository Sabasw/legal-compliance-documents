from sqlalchemy import (
    Column, String, JSON, DateTime, Integer, ForeignKey, Boolean, 
    Enum, Float, Table, Text, func
)
from sqlalchemy.orm import relationship
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
import datetime
from app.database.db.db_connection import Base
import enum
from typing import List, Dict, Optional

def generate_uuid():
    return str(uuid.uuid4())

class DocumentStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    
class UserRole(enum.Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    USER = "user"



class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    role = Column(
        String,
        nullable=False,
        default="user"
    )
    is_active = Column(Boolean, default=True)
    two_factor_enabled = Column(Boolean, default=False)
    two_factor_secret = Column(String, nullable=True)
    # stripe_customer_id = Column(String, nullable=True)  # For billing integration
    # current_jurisdiction = Column(String, default="AU")  # Current legal jurisdiction
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    documents = relationship("Document", back_populates="owner")
    audit_logs = relationship("AuditLog", back_populates="user")
    usage_records = relationship("UsageRecord", back_populates="user")

    # def set_password(self, password: str):
    #     self.hashed_password = pwd_context.hash(password)

    # def verify_password(self, password: str) -> bool:
    #     return pwd_context.verify(password, self.hashed_password)

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    document_type = Column(String, nullable=False)
    status = Column(String, nullable=False)
    
    # Document metadata and analysis fields
    document_metadata = Column(JSONB, nullable=False, server_default='{}')
    compliance_status = Column(String, nullable=True)
    risk_score = Column(Float, nullable=True)
    risk_profile = Column(JSONB, nullable=True, server_default='{}')
    statutory_references = Column(JSONB, nullable=True, server_default='{}')
    key_issues = Column(JSONB, nullable=True, server_default='[]')
    recommendations = Column(JSONB, nullable=True, server_default='[]')
    predictive_outcomes = Column(JSONB, nullable=True, server_default='{}')
    summary = Column(Text, nullable=True)
    full_analysis = Column(JSONB, nullable=True, server_default='{}')
    
    # Blockchain tracking fields
    blockchain_tx_hash = Column(String, nullable=True)
    blockchain_content_hash = Column(String, nullable=True)
    blockchain_document_hash = Column(String, nullable=True)
    blockchain_metadata = Column(JSONB, nullable=True, server_default='{}')
    department = Column(String, nullable=True)
    classification = Column(String, nullable=True, default="Confidential")
    
    # Audit trail fields
    audit_trail = Column(JSONB, nullable=True, server_default='[]')  # List of audit actions
    last_accessed = Column(DateTime(timezone=True), nullable=True)
    last_modified = Column(DateTime(timezone=True), nullable=True)
    version = Column(String, nullable=True, default="1.0")
    previous_versions = Column(JSONB, nullable=True, server_default='[]')  # Track version history
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    owner = relationship("User", back_populates="documents")
    analyses = relationship("DocumentAnalysis", back_populates="document")
    audit_logs = relationship("AuditLog", back_populates="document")
    risk_scores = relationship("RiskScore", back_populates="document")

    def add_audit_entry(self, action: str, user_id: str, metadata: dict = None):
        """Add an audit trail entry"""
        if self.audit_trail is None:
            self.audit_trail = []
            
        entry = {
            "action": action,
            "user_id": user_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        if action == "MODIFIED":
            # Track version history
            if self.previous_versions is None:
                self.previous_versions = []
            
            current_version = {
                "version": self.version,
                "content_hash": self.blockchain_content_hash,
                "document_hash": self.blockchain_document_hash,
                "timestamp": self.updated_at.isoformat() if self.updated_at else datetime.datetime.utcnow().isoformat()
            }
            self.previous_versions.append(current_version)
            
            # Increment version
            major, minor = self.version.split(".")
            self.version = f"{major}.{int(minor) + 1}"
            
            # Update last modified
            self.last_modified = datetime.datetime.utcnow()
            
        elif action == "ACCESSED":
            self.last_accessed = datetime.datetime.utcnow()
            
        self.audit_trail.append(entry)

    def get_audit_history(self) -> List[Dict]:
        """Get complete audit history with version tracking"""
        history = []
        
        # Add creation entry
        history.append({
            "action": "CREATED",
            "version": "1.0",
            "timestamp": self.created_at.isoformat(),
            "user_id": str(self.owner_id),
            "content_hash": self.blockchain_content_hash,
            "document_hash": self.blockchain_document_hash,
            "transaction_hash": self.blockchain_tx_hash,
            "metadata": self.blockchain_metadata
        })
        
        # Add version history
        if self.previous_versions:
            for version in self.previous_versions:
                history.append({
                    "action": "MODIFIED",
                    **version
                })
                
        # Add other audit trail entries
        if self.audit_trail:
            for entry in self.audit_trail:
                if entry["action"] not in ["CREATED", "MODIFIED"]:
                    history.append(entry)
                    
        return sorted(history, key=lambda x: x["timestamp"])

class ComplianceRule(Base):
    __tablename__ = "compliance_rules"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code = Column(String, unique=True, nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text)
    category = Column(String)
    jurisdiction = Column(String)
    effective_date = Column(DateTime)
    version = Column(String)
    parameters = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # DRE-specific fields
    regulation_name = Column(String, nullable=True)
    rule_description = Column(Text, nullable=True)
    rule_logic = Column(JSON, nullable=True)
    last_updated = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    risk_scores = relationship("RiskScore", back_populates="compliance_rule")

class DocumentAnalysis(Base):
    __tablename__ = "document_analyses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    compliance_status = Column(String)
    risk_score = Column(String)
    risk_profile = Column(JSON)
    statutory_references = Column(JSON)
    key_issues = Column(JSON)
    recommendations = Column(JSON)
    predictive_outcomes = Column(JSON)
    summary = Column(Text)
    full_analysis = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="analyses")

class RiskScore(Base):
    __tablename__ = "risk_scores"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    compliance_rule_id = Column(UUID(as_uuid=True), ForeignKey("compliance_rules.id"))
    score = Column(Float)
    severity = Column(String)
    impact_areas = Column(JSON)
    mitigation_suggestions = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="risk_scores")
    compliance_rule = relationship("ComplianceRule", back_populates="risk_scores")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)
    action = Column(String, nullable=False)
    entity_type = Column(String, nullable=False)  # e.g., "document", "user", "compliance_rule"
    entity_id = Column(String, nullable=False)
    changes = Column(JSON)  # Store before/after states
    ip_address = Column(String)
    user_agent = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Blockchain-related fields
    data_hash = Column(String, nullable=True)  # Hash of the data for blockchain verification
    blockchain_tx_hash = Column(String, nullable=True)  # Blockchain transaction hash
    blockchain_block_number = Column(Integer, nullable=True)  # Block number in blockchain
    additional_data = Column(JSON, nullable=True)  # Additional metadata
    verified = Column(Boolean, default=False)  # Whether blockchain verification passed
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    document = relationship("Document", back_populates="audit_logs")

class UsageRecord(Base):
    __tablename__ = "usage_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    usage_type = Column(String, nullable=False)  # document_upload, ai_analysis, prediction, storage, etc.
    quantity = Column(Integer, default=1)
    unit_cost = Column(Float, nullable=True)  # Cost per unit
    total_cost = Column(Float, nullable=True)  # Total cost for this usage
    billing_period = Column(String)  # YYYY-MM format
    usage_metadata = Column(JSON)  # Additional usage metadata
    recorded_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="usage_records")



class Invoice(Base):
    __tablename__ = "invoices"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subscription_id = Column(UUID(as_uuid=True), nullable=True)  # Removed subscription reference
    stripe_invoice_id = Column(String, nullable=True)
    amount_due = Column(Float)
    amount_paid = Column(Float)
    currency = Column(String, default="USD")
    status = Column(String)  # paid, unpaid, void, etc.
    invoice_period_start = Column(DateTime)
    invoice_period_end = Column(DateTime)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    paid_at = Column(DateTime, nullable=True)
    










class LawUpdate(Base):
    __tablename__ = "law_updates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(String, nullable=False)  # AustLII, legislation.gov.au, ASIC, etc.
    jurisdiction = Column(String, nullable=False)  # AU, US, UK, etc.
    law_id = Column(String, nullable=False)  # External ID from the source
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    effective_date = Column(DateTime, nullable=True)
    url = Column(String, nullable=True)
    raw_data = Column(JSON, nullable=True)  # Raw JSON data from the source
    processed = Column(Boolean, default=False)
    processed_at = Column(DateTime, nullable=True)
    category = Column(String, nullable=True)  # amendment, new_law, repeal, etc.
    urgency = Column(String, default="medium")  # low, medium, high, critical
    impact_assessment = Column(JSON, nullable=True)  # Assessment of impact
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    
class BillingPlan(Base):
    __tablename__ = "billing_plans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    plan_name = Column(String, nullable=False)  # e.g., basic, professional, enterprise
    stripe_subscription_id = Column(String, nullable=True)
    monthly_fee = Column(Float, nullable=False)
    status = Column(String, default="active")
    current_period_start = Column(DateTime, nullable=False)
    current_period_end = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    user = relationship("User", backref="billing_plans")


class BlacklistedToken(Base):
    __tablename__ = "blacklisted_tokens"

    token = Column(String, primary_key=True, index=True)
    reason = Column(String)
    blacklisted_at = Column(DateTime, default=datetime.datetime.utcnow)