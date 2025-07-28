from typing import Optional, Dict, Any
from fastapi import Request
import uuid
import logging

from app.database.models.models import AuditLog

logger = logging.getLogger(__name__)

def create_audit_log(
    db, 
    user_id: uuid.UUID, 
    action: str, 
    entity_type: str,
    entity_id: str,
    document_id: Optional[uuid.UUID] = None,
    changes: Optional[Dict[str, Any]] = None,
    request: Optional[Request] = None
) -> AuditLog:
    """
    Create an audit log entry for user actions.
    
    Args:
        db: Database session
        user_id: ID of the user performing the action
        action: Description of the action (e.g., "create", "update", "delete")
        entity_type: Type of entity affected (e.g., "document", "user")
        entity_id: ID of the entity affected
        document_id: Optional ID of related document
        changes: Optional dictionary of changes (before/after)
        request: Optional request object for IP and user agent info
        
    Returns:
        AuditLog: The created audit log entry
    """
    try:
        # Create the audit log entry
        log = AuditLog(
            user_id=user_id,
            document_id=document_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            changes=changes or {},
            ip_address=request.client.host if request else None,
            user_agent=request.headers.get("user-agent") if request else None
        )
        
        # Add to database
        db.add(log)
        db.commit()
        db.refresh(log)
        
        logger.info(f"Audit log created: {action} on {entity_type} {entity_id} by user {user_id}")
        return log
        
    except Exception as e:
        logger.error(f"Failed to create audit log: {str(e)}")
        # Don't fail the main operation if audit logging fails
        db.rollback()
        return None

 