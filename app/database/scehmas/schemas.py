from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime
import enum


class RegisterUser(BaseModel):
    name: str
    email: EmailStr
    role:str
    password1: str
    password2: str
    
    class Config():
        orm_mode = True
        
class UpdateUser(BaseModel):
    username: str
    title: str
    organization: str
    work_phone: str
    contact_number:str
    email:str
    
    class Config():
        orm_mode = True
        
        
class ShowUser(BaseModel):
    username: str
    email: str
    role:str
    
    class Config():
        orm_mode = True

class UserBasicInfo(BaseModel):
    username: str
    email: str
    role: str

    class Config:
        orm_mode = True

        
class Login(BaseModel):
    email : str
    password : str
    
    class Config():
        orm_mode = True
        
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: str
    
    
class DocumentUpload(BaseModel):
    file_bytes: bytes
    filename: str
    content_type: str

class DocumentAnalysisSchema(BaseModel):
    doc_id: str
    document_type: str
    compliance_status: str
    risk_score: str                      # Must be string!
    risk_profile: Dict[str, Any]         # Not a str!
    statutory_references: List[str]
    key_issues: List[str]
    recommendations: List[str]
    predictive_outcomes: List[str]
    timestamp: datetime                  # Required!

class KnowledgeBaseUpdate(BaseModel):
    documents: List[DocumentUpload]
    rebuild_index: bool = True

class AnalysisRequest(BaseModel):
    text: Optional[str] = None
    doc_id: Optional[str] = None
    doc_type: Optional[str] = None

class SystemConfig(BaseModel):
    max_text_length: int = 100000
    top_k_rules: int = 7
    summary_length: int = 600
    temperature: float = 0.2
    max_kb_distance: float = 1.5
    
class DocumentStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"