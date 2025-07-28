import os
from fastapi import HTTPException, status
from typing import List, Optional, Union
import uuid
from sqlalchemy.orm import Session
from app.database.models.models import Document, User, UserRole

def validate_file_type(filename: str, allowed_extensions: Optional[List[str]] = None) -> bool:
    """
    Validate if the file type is allowed.
    
    Args:
        filename: Name of the file to validate
        allowed_extensions: List of allowed file extensions (defaults to pdf, docx, txt)
        
    Returns:
        bool: True if file type is allowed, False otherwise
    """
    if allowed_extensions is None:
        allowed_extensions = ['.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        
    ext = os.path.splitext(filename)[1].lower()
    return ext in allowed_extensions

from sqlalchemy.orm import Session
def check_document_ownership(
    doc_id: Union[str, uuid.UUID],
    user: User,
    db: Session
) -> Document:
    """
    Check if a user has access rights to a document.
    
    Args:
        doc_id: UUID of the document
        user: User attempting to access the document
        db: Database session
        
    Returns:
        Document: The document object if access is allowed
        
    Raises:
        HTTPException: If document not found or user lacks access rights
    """
    # Convert string UUID to UUID object if needed
    if isinstance(doc_id, str):
        try:
            doc_id = uuid.UUID(doc_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid document ID format"
            )
    
    # Query the document
    document = db.query(Document).filter(Document.id == doc_id).first()
    
    # Check if document exists
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Document not found"
        )
    
    # Check access rights (owner or admin)
    if document.owner_id != user.id and user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this document"
        )
    
    return document

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and other security issues.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Remove potentially dangerous characters
    sanitized = os.path.basename(filename)
    
    # Remove any null bytes
    sanitized = sanitized.replace('\0', '')
    
    # Ensure the filename isn't empty after sanitization
    if not sanitized:
        sanitized = "unnamed_file"
    
    return sanitized 