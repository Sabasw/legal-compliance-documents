from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.db.db_connection import Base
from sqlalchemy.dialects.postgresql import UUID
import uuid
import enum

class BlockchainActionType(enum.Enum):
    """Types of actions that can be tracked in the blockchain"""
    CREATED = "CREATED"
    MODIFIED = "MODIFIED"
    ACCESSED = "ACCESSED"
    DELETED = "DELETED"
    SHARED = "SHARED"
    REVIEWED = "REVIEWED"

class BlockchainDocument(Base):
    """Model for tracking documents in blockchain"""
    __tablename__ = "blockchain_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_hash = Column(String, nullable=False, index=True)  # Blockchain document hash
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    department = Column(String, nullable=True)
    classification = Column(String, nullable=False, default="Confidential")
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    document_metadata = Column(JSON, nullable=True)  # Additional metadata about the document

    # Relationships
    audit_entries = relationship("BlockchainAuditEntry", back_populates="document")
    creator = relationship("User", backref="blockchain_documents")

class BlockchainAuditEntry(Base):
    """Model for blockchain audit trail entries"""
    __tablename__ = "blockchain_audit_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("blockchain_documents.id"), nullable=False)
    action_type = Column(SQLEnum(BlockchainActionType), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    transaction_hash = Column(String, nullable=False)  # Blockchain transaction hash
    previous_version = Column(String, nullable=True)  # For tracking document versions
    action_metadata = Column(JSON, nullable=True)  # Additional action metadata
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    document = relationship("BlockchainDocument", back_populates="audit_entries")
    user = relationship("User", backref="blockchain_audit_entries")

class BlockchainUserKeys(Base):
    """Model for storing user blockchain keys"""
    __tablename__ = "blockchain_user_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), unique=True, nullable=False)
    public_key = Column(String, nullable=False)  # Solana public key
    encrypted_private_key = Column(String, nullable=False)  # Encrypted private key
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", backref="blockchain_keys") 