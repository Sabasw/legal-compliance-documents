import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy.orm import Session
from fastapi import Request

from app.database.models.models import AuditLog
from app.services.blockchain_audit_service import blockchain_audit_service, AuditRecord
from app.utils.audit import create_audit_log

logger = logging.getLogger(__name__)

class EnhancedAuditService:
    """
    Enhanced audit service that combines traditional database logging
    with blockchain-based immutable audit trails
    """
    
    def __init__(self):
        self.blockchain_service = blockchain_audit_service
    
    async def create_immutable_audit_log(
        self,
        db: Session,
        user_id: str,
        action: str,
        entity_type: str,
        entity_id: str,
        document_id: Optional[str] = None,
        changes: Optional[Dict[str, Any]] = None,
        request: Optional[Request] = None,
        critical_level: bool = False
    ) -> Dict[str, Any]:
        """
        Create an audit log with blockchain verification for critical legal records
        
        Args:
            db: Database session
            user_id: ID of the user performing the action
            action: Description of the action
            entity_type: Type of entity affected
            entity_id: ID of the entity affected
            document_id: Optional document ID
            changes: Optional changes dictionary
            request: Optional request object
            critical_level: Whether this is a critical legal record requiring blockchain verification
            
        Returns:
            Dictionary containing audit log details and blockchain verification info
        """
        try:
            # Create traditional database audit log
            db_audit_log = await create_audit_log(
                db=db,
                user_id=uuid.UUID(user_id),
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                document_id=uuid.UUID(document_id) if document_id else None,
                changes=changes,
                request=request
            )
            
            # Create blockchain audit record for critical legal records
            blockchain_record = None
            blockchain_tx_hash = None
            
            if critical_level or self._is_critical_action(action, entity_type):
                blockchain_record = self.blockchain_service.create_audit_record(
                    user_id=user_id,
                    action=action,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    document_id=document_id,
                    changes=changes,
                    ip_address=request.client.host if request else None,
                    user_agent=request.headers.get("user-agent") if request else None
                )
                
                # Add to blockchain
                blockchain_tx_hash = self.blockchain_service.add_to_blockchain(blockchain_record)
                
                # Update database audit log with blockchain info
                if db_audit_log and blockchain_record:
                    db_audit_log.data_hash = blockchain_record.data_hash
                    db_audit_log.blockchain_tx_hash = blockchain_tx_hash
                    db_audit_log.blockchain_block_number = 0  # Would be set from blockchain receipt
                    db_audit_log.verified = True
                    db.commit()
            
            return {
                "audit_log_id": str(db_audit_log.id) if db_audit_log else None,
                "blockchain_record_id": blockchain_record.record_id if blockchain_record else None,
                "blockchain_tx_hash": blockchain_tx_hash,
                "data_hash": blockchain_record.data_hash if blockchain_record else None,
                "verified": bool(blockchain_record),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating immutable audit log: {str(e)}")
            raise
    
    def _is_critical_action(self, action: str, entity_type: str) -> bool:
        """
        Determine if an action is critical and requires blockchain verification
        """
        critical_actions = {
            "document": ["upload", "delete", "modify", "compliance_check", "risk_assessment"],
            "compliance_rule": ["create", "update", "delete", "activate", "deactivate"],
            "user": ["create", "delete", "role_change", "permission_change"],
            "audit": ["export", "delete", "modify"],
            "legal_document": ["create", "modify", "delete", "sign", "verify"],
            "contract": ["create", "modify", "delete", "sign", "execute"],
            "evidence": ["upload", "delete", "modify", "verify"]
        }
        
        return (
            entity_type in critical_actions and 
            action in critical_actions[entity_type]
        )
    
    async def verify_audit_integrity(
        self,
        audit_log_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """
        Verify the integrity of an audit log entry
        """
        try:
            # Get database audit log
            db_audit_log = db.query(AuditLog).filter(AuditLog.id == uuid.UUID(audit_log_id)).first()
            
            if not db_audit_log:
                return {
                    "valid": False,
                    "error": "Audit log not found",
                    "audit_log_id": audit_log_id
                }
            
            # If it has blockchain verification, verify against blockchain
            if db_audit_log.blockchain_tx_hash and db_audit_log.data_hash:
                # Verify against blockchain service
                blockchain_verification = self.blockchain_service.verify_audit_chain()
                
                # Check if the specific record exists in blockchain
                audit_trail = self.blockchain_service.get_audit_trail(
                    entity_id=db_audit_log.entity_id
                )
                
                blockchain_record = next(
                    (r for r in audit_trail if r.data_hash == db_audit_log.data_hash),
                    None
                )
                
                return {
                    "valid": blockchain_verification["valid"] and blockchain_record is not None,
                    "audit_log_id": audit_log_id,
                    "database_record": {
                        "action": db_audit_log.action,
                        "entity_type": db_audit_log.entity_type,
                        "entity_id": db_audit_log.entity_id,
                        "timestamp": db_audit_log.timestamp.isoformat(),
                        "data_hash": db_audit_log.data_hash,
                        "blockchain_tx_hash": db_audit_log.blockchain_tx_hash,
                        "verified": db_audit_log.verified
                    },
                    "blockchain_verification": blockchain_verification,
                    "blockchain_record_found": blockchain_record is not None
                }
            else:
                # Traditional database-only audit log
                return {
                    "valid": True,
                    "audit_log_id": audit_log_id,
                    "database_record": {
                        "action": db_audit_log.action,
                        "entity_type": db_audit_log.entity_type,
                        "entity_id": db_audit_log.entity_id,
                        "timestamp": db_audit_log.timestamp.isoformat(),
                        "verified": False,
                        "note": "Database-only audit log (no blockchain verification)"
                    },
                    "blockchain_verification": None
                }
                
        except Exception as e:
            logger.error(f"Error verifying audit integrity: {str(e)}")
            return {
                "valid": False,
                "error": str(e),
                "audit_log_id": audit_log_id
            }
    
    async def get_compliance_audit_trail(
        self,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_blockchain: bool = True,
        db: Session = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive audit trail for compliance verification
        """
        try:
            # Get database audit logs
            db_audit_logs = []
            if db:
                from sqlalchemy import select, desc
                from app.database.models.models import AuditLog
                
                query = select(AuditLog).order_by(desc(AuditLog.timestamp))
                
                if user_id:
                    query = query.where(AuditLog.user_id == uuid.UUID(user_id))
                
                if entity_type:
                    query = query.where(AuditLog.entity_type == entity_type)
                
                if entity_id:
                    query = query.where(AuditLog.entity_id == entity_id)
                
                if start_date:
                    query = query.where(AuditLog.timestamp >= start_date)
                
                if end_date:
                    query = query.where(AuditLog.timestamp <= end_date)
                
                result = db.execute(query)
                db_audit_logs = result.scalars().all()
            
            # Get blockchain audit trail
            blockchain_records = []
            if include_blockchain:
                blockchain_records = self.blockchain_service.get_audit_trail(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    user_id=user_id,
                    start_date=start_date,
                    end_date=end_date
                )
            
            # Verify blockchain chain integrity
            blockchain_verification = None
            if include_blockchain:
                blockchain_verification = self.blockchain_service.verify_audit_chain()
            
            return {
                "audit_trail": {
                    "database_records": len(db_audit_logs),
                    "blockchain_records": len(blockchain_records),
                    "total_records": len(db_audit_logs) + len(blockchain_records)
                },
                "blockchain_verification": blockchain_verification,
                "database_audit_logs": [
                    {
                        "id": str(log.id),
                        "action": log.action,
                        "entity_type": log.entity_type,
                        "entity_id": log.entity_id,
                        "timestamp": log.timestamp.isoformat(),
                        "data_hash": log.data_hash,
                        "blockchain_tx_hash": log.blockchain_tx_hash,
                        "verified": log.verified
                    }
                    for log in db_audit_logs
                ],
                "blockchain_records": [
                    {
                        "record_id": record.record_id,
                        "action": record.action,
                        "entity_type": record.entity_type,
                        "entity_id": record.entity_id,
                        "timestamp": record.timestamp,
                        "data_hash": record.data_hash,
                        "previous_hash": record.previous_hash
                    }
                    for record in blockchain_records
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting compliance audit trail: {str(e)}")
            return {
                "error": str(e),
                "audit_trail": {"total_records": 0}
            }
    
    async def export_compliance_report(
        self,
        format: str = "json",
        entity_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> str:
        """
        Export comprehensive compliance audit report
        """
        try:
            # Get audit trail
            audit_trail = await self.get_compliance_audit_trail(
                entity_id=entity_id,
                start_date=start_date,
                end_date=end_date
            )
            
            # Get blockchain verification
            blockchain_verification = self.blockchain_service.verify_audit_chain()
            
            # Create comprehensive report
            report_data = {
                "report_metadata": {
                    "generated_at": datetime.utcnow().isoformat(),
                    "report_type": "compliance_audit_report",
                    "format": format,
                    "filters": {
                        "entity_id": entity_id,
                        "start_date": start_date,
                        "end_date": end_date
                    }
                },
                "audit_summary": audit_trail["audit_trail"],
                "blockchain_verification": blockchain_verification,
                "compliance_status": {
                    "audit_chain_valid": blockchain_verification.get("valid", False),
                    "total_records": audit_trail["audit_trail"]["total_records"],
                    "blockchain_records": audit_trail["audit_trail"]["blockchain_records"],
                    "database_records": audit_trail["audit_trail"]["database_records"]
                },
                "detailed_records": {
                    "database_audit_logs": audit_trail.get("database_audit_logs", []),
                    "blockchain_records": audit_trail.get("blockchain_records", [])
                }
            }
            
            if format.lower() == "json":
                import json
                return json.dumps(report_data, indent=2)
            
            elif format.lower() == "csv":
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write summary
                writer.writerow(["Report Type", "Compliance Audit Report"])
                writer.writerow(["Generated At", report_data["report_metadata"]["generated_at"]])
                writer.writerow(["Audit Chain Valid", report_data["compliance_status"]["audit_chain_valid"]])
                writer.writerow(["Total Records", report_data["compliance_status"]["total_records"]])
                writer.writerow([])
                
                # Write database records
                writer.writerow(["Database Audit Logs"])
                writer.writerow([
                    "ID", "Action", "Entity Type", "Entity ID", "Timestamp", 
                    "Data Hash", "Blockchain TX Hash", "Verified"
                ])
                
                for log in report_data["detailed_records"]["database_audit_logs"]:
                    writer.writerow([
                        log["id"], log["action"], log["entity_type"], log["entity_id"],
                        log["timestamp"], log["data_hash"], log["blockchain_tx_hash"], log["verified"]
                    ])
                
                writer.writerow([])
                
                # Write blockchain records
                writer.writerow(["Blockchain Records"])
                writer.writerow([
                    "Record ID", "Action", "Entity Type", "Entity ID", 
                    "Timestamp", "Data Hash", "Previous Hash"
                ])
                
                for record in report_data["detailed_records"]["blockchain_records"]:
                    writer.writerow([
                        record["record_id"], record["action"], record["entity_type"],
                        record["entity_id"], record["timestamp"], record["data_hash"],
                        record["previous_hash"]
                    ])
                
                return output.getvalue()
            
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting compliance report: {str(e)}")
            raise

# Global instance
enhanced_audit_service = EnhancedAuditService() 