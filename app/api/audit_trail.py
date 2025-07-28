from fastapi import APIRouter, Depends, HTTPException, Request, status, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime
import uuid

from app.database.db.db_connection import get_db
from app.database.models.models import AuditLog
from app.services.enhanced_audit_service import enhanced_audit_service
from app.services.blockchain_audit_service import blockchain_audit_service
from app.utils.dependencies import get_current_user
from app.database.models.models import User

router = APIRouter(
    prefix="/audit-trail",
    tags=["Audit Trail"]
)

@router.post("/create")
async def create_immutable_audit_log(
    action: str,
    entity_type: str,
    entity_id: str,
    document_id: Optional[str] = None,
    changes: Optional[dict] = None,
    critical_level: bool = False,
    request: Request = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create an immutable audit log with blockchain verification
    """
    try:
        result = await enhanced_audit_service.create_immutable_audit_log(
            db=db,
            user_id=str(current_user.id),
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            document_id=document_id,
            changes=changes,
            request=request,
            critical_level=critical_level
        )
        
        return {
            "success": True,
            "message": "Audit log created successfully",
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create audit log: {str(e)}"
        )

@router.get("/verify/{audit_log_id}")
async def verify_audit_integrity(
    audit_log_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Verify the integrity of an audit log entry
    """
    try:
        result = await enhanced_audit_service.verify_audit_integrity(
            audit_log_id=audit_log_id,
            db=db
        )
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to verify audit integrity: {str(e)}"
        )

@router.get("/trail")
async def get_audit_trail(
    entity_id: Optional[str] = Query(None, description="Filter by entity ID"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    include_blockchain: bool = Query(True, description="Include blockchain records"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive audit trail for compliance verification
    """
    try:
        result = await enhanced_audit_service.get_compliance_audit_trail(
            entity_id=entity_id,
            entity_type=entity_type,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            include_blockchain=include_blockchain,
            db=db
        )
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get audit trail: {str(e)}"
        )

@router.get("/export")
async def export_audit_report(
    format: str = Query("json", description="Export format (json/csv)"),
    entity_id: Optional[str] = Query(None, description="Filter by entity ID"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    current_user: User = Depends(get_current_user)
):
    """
    Export comprehensive compliance audit report
    """
    try:
        if format.lower() not in ["json", "csv"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported format. Use 'json' or 'csv'"
            )
        
        report_content = await enhanced_audit_service.export_compliance_report(
            format=format,
            entity_id=entity_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "success": True,
            "format": format,
            "content": report_content,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export audit report: {str(e)}"
        )

@router.get("/blockchain/verify")
async def verify_blockchain_chain(
    current_user: User = Depends(get_current_user)
):
    """
    Verify the integrity of the blockchain audit chain
    """
    try:
        verification_result = blockchain_audit_service.verify_audit_chain()
        
        return {
            "success": True,
            "data": verification_result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to verify blockchain chain: {str(e)}"
        )

@router.get("/blockchain/stats")
async def get_blockchain_stats(
    current_user: User = Depends(get_current_user)
):
    """
    Get blockchain audit trail statistics
    """
    try:
        audit_trail = blockchain_audit_service.get_audit_trail()
        verification_result = blockchain_audit_service.verify_audit_chain()
        
        stats = {
            "total_records": len(audit_trail),
            "chain_valid": verification_result.get("valid", False),
            "last_record": audit_trail[0] if audit_trail else None,
            "verification_details": verification_result
        }
        
        return {
            "success": True,
            "data": stats
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get blockchain stats: {str(e)}"
        )

@router.get("/blockchain/trail")
async def get_blockchain_audit_trail(
    entity_id: Optional[str] = Query(None, description="Filter by entity ID"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    current_user: User = Depends(get_current_user)
):
    """
    Get blockchain-only audit trail
    """
    try:
        records = blockchain_audit_service.get_audit_trail(
            entity_id=entity_id,
            entity_type=entity_type,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "success": True,
            "data": {
                "records": [
                    {
                        "record_id": record.record_id,
                        "timestamp": record.timestamp,
                        "user_id": record.user_id,
                        "action": record.action,
                        "entity_type": record.entity_type,
                        "entity_id": record.entity_id,
                        "document_id": record.document_id,
                        "data_hash": record.data_hash,
                        "previous_hash": record.previous_hash,
                        "nonce": record.nonce
                    }
                    for record in records
                ],
                "total_records": len(records)
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get blockchain audit trail: {str(e)}"
        )

@router.get("/database/trail")
async def get_database_audit_trail(
    entity_id: Optional[str] = Query(None, description="Filter by entity ID"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    limit: int = Query(100, description="Maximum number of records to return"),
    offset: int = Query(0, description="Offset for pagination"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get database-only audit trail
    """
    try:
        from sqlalchemy import select, desc
        
        query = select(AuditLog).order_by(desc(AuditLog.timestamp))
        
        if user_id:
            query = query.where(AuditLog.user_id == uuid.UUID(user_id))
        
        if entity_type:
            query = query.where(AuditLog.entity_type == entity_type)
        
        if entity_id:
            query = query.where(AuditLog.entity_id == entity_id)
        
        # Apply pagination
        query = query.offset(offset).limit(limit)
        
        result = db.execute(query)
        audit_logs = result.scalars().all()
        
        return {
            "success": True,
            "data": {
                "records": [
                    {
                        "id": str(log.id),
                        "user_id": str(log.user_id),
                        "document_id": str(log.document_id) if log.document_id else None,
                        "action": log.action,
                        "entity_type": log.entity_type,
                        "entity_id": log.entity_id,
                        "changes": log.changes,
                        "ip_address": log.ip_address,
                        "user_agent": log.user_agent,
                        "timestamp": log.timestamp.isoformat(),
                        "data_hash": log.data_hash,
                        "blockchain_tx_hash": log.blockchain_tx_hash,
                        "blockchain_block_number": log.blockchain_block_number,
                        "verified": log.verified
                    }
                    for log in audit_logs
                ],
                "total_records": len(audit_logs),
                "pagination": {
                    "limit": limit,
                    "offset": offset
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get database audit trail: {str(e)}"
        )

@router.get("/compliance/status")
async def get_compliance_status(
    entity_id: Optional[str] = Query(None, description="Entity ID to check"),
    current_user: User = Depends(get_current_user)
):
    """
    Get compliance status and audit trail integrity
    """
    try:
        # Get blockchain verification
        blockchain_verification = blockchain_audit_service.verify_audit_chain()
        
        # Get audit trail for entity if specified
        audit_trail = None
        if entity_id:
            audit_trail = await enhanced_audit_service.get_compliance_audit_trail(
                entity_id=entity_id
            )
        
        compliance_status = {
            "blockchain_chain_valid": blockchain_verification.get("valid", False),
            "total_blockchain_records": blockchain_verification.get("total_records", 0),
            "invalid_blockchain_records": blockchain_verification.get("invalid_records", 0),
            "entity_audit_trail": audit_trail,
            "last_verification": datetime.utcnow().isoformat(),
            "compliance_level": "FULL" if blockchain_verification.get("valid", False) else "PARTIAL"
        }
        
        return {
            "success": True,
            "data": compliance_status
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get compliance status: {str(e)}"
        ) 