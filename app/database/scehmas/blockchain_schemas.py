from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
from uuid import UUID

class BlockchainActionType(str, Enum):
    CREATED = "CREATED"
    MODIFIED = "MODIFIED"
    ACCESSED = "ACCESSED"
    DELETED = "DELETED"
    SHARED = "SHARED"
    REVIEWED = "REVIEWED"

class BlockchainDocumentBase(BaseModel):
    original_filename: str
    department: Optional[str] = None
    classification: str = "Confidential"
    document_metadata: Optional[Dict[str, Any]] = None

class BlockchainDocumentCreate(BlockchainDocumentBase):
    pass

class BlockchainDocumentResponse(BlockchainDocumentBase):
    id: UUID
    document_hash: str
    created_by: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True

class BlockchainAuditEntryBase(BaseModel):
    action_type: BlockchainActionType
    previous_version: Optional[str] = None
    action_metadata: Optional[Dict[str, Any]] = None

class BlockchainAuditEntryCreate(BlockchainAuditEntryBase):
    document_id: UUID

class BlockchainAuditEntryResponse(BlockchainAuditEntryBase):
    id: UUID
    document_id: UUID
    user_id: UUID
    transaction_hash: str
    timestamp: datetime

    class Config:
        orm_mode = True

class BlockchainUserKeysBase(BaseModel):
    public_key: str

class BlockchainUserKeysCreate(BlockchainUserKeysBase):
    encrypted_private_key: str
    user_id: UUID

class BlockchainUserKeysResponse(BlockchainUserKeysBase):
    id: UUID
    user_id: UUID
    created_at: datetime
    last_used: Optional[datetime] = None

    class Config:
        orm_mode = True

class BlockchainDocumentHistory(BaseModel):
    document: BlockchainDocumentResponse
    audit_entries: List[BlockchainAuditEntryResponse]

    class Config:
        orm_mode = True 