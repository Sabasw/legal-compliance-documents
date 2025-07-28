"""
Celery Task Scheduler Service
Handles background tasks for law updates, compliance monitoring, and data processing
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from celery import Celery, Task
from celery.schedules import crontab
from celery.result import AsyncResult
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from app.db.session import AsyncSessionLocal
from app.db.models import Document, User, ComplianceRule, LawUpdate
from app.services.blockchain_service import blockchain_service
from app.config import settings
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Initialize Celery app
celery_app = Celery(
    'solvelex_tasks',
    broker=f'redis://{settings.REDIS_HOST or "localhost"}:{settings.REDIS_PORT or 6379}/0',
    backend=f'redis://{settings.REDIS_HOST or "localhost"}:{settings.REDIS_PORT or 6379}/0',
    include=['app.services.scheduler_service']
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    result_expires=3600,  # 1 hour
)

# Periodic task schedule
celery_app.conf.beat_schedule = {
    'check-law-updates': {
        'task': 'app.services.scheduler_service.check_law_updates',
        'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
    },
    'compliance-monitoring': {
        'task': 'app.services.scheduler_service.monitor_compliance',
        'schedule': crontab(minute='*/30'),  # Every 30 minutes
    },
    'cleanup-expired-documents': {
        'task': 'app.services.scheduler_service.cleanup_expired_documents',
        'schedule': crontab(hour=3, minute=0),  # Daily at 3 AM
    },
    'generate-usage-reports': {
        'task': 'app.services.scheduler_service.generate_usage_reports',
        'schedule': crontab(hour=1, minute=0, day_of_month=1),  # Monthly
    },
    'update-law-database': {
        'task': 'app.services.scheduler_service.update_law_database',
        'schedule': crontab(hour=4, minute=0, day_of_week=1),  # Weekly on Monday
    },
}

class AsyncTask(Task):
    """Base task class for async operations"""
    
    def run_async(self, coro):
        """Run async coroutine in sync context"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

@celery_app.task(bind=True, base=AsyncTask)
def check_law_updates(self):
    """Check for law updates from various legal databases"""
    logger.info("Starting law updates check")
    
    async def _check_updates():
        async with AsyncSessionLocal() as session:
            try:
                # Australian Legal Database APIs
                sources = [
                    {
                        'name': 'AustLII',
                        'url': 'https://www.austlii.edu.au/api/updates',
                        'jurisdiction': 'AU'
                    },
                    {
                        'name': 'Federal Register of Legislation',
                        'url': 'https://www.legislation.gov.au/api/updates',
                        'jurisdiction': 'AU_FEDERAL'
                    },
                    # Add more sources as needed
                ]
                
                updates_found = []
                
                for source in sources:
                    try:
                        logger.info(f"Checking updates from {source['name']}")
                        
                        # Make API call to legal database
                        response = requests.get(
                            source['url'],
                            timeout=30,
                            headers={'User-Agent': 'SolveLex/1.0'}
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            # Process updates
                            for update in data.get('updates', []):
                                law_update = LawUpdate(
                                    source=source['name'],
                                    jurisdiction=source['jurisdiction'],
                                    law_id=update.get('id'),
                                    title=update.get('title'),
                                    description=update.get('description'),
                                    effective_date=datetime.fromisoformat(
                                        update.get('effective_date')
                                    ) if update.get('effective_date') else None,
                                    url=update.get('url'),
                                    raw_data=update,
                                    processed=False,
                                    created_at=datetime.utcnow()
                                )
                                
                                session.add(law_update)
                                updates_found.append(update)
                    
                    except Exception as e:
                        logger.error(f"Error checking {source['name']}: {str(e)}")
                        continue
                
                await session.commit()
                
                # Trigger processing of new updates
                if updates_found:
                    process_law_updates.apply_async(
                        args=[len(updates_found)],
                        countdown=60  # Process after 1 minute
                    )
                
                logger.info(f"Law updates check completed. Found {len(updates_found)} updates")
                return {"status": "success", "updates_found": len(updates_found)}
                
            except Exception as e:
                logger.error(f"Law updates check failed: {str(e)}")
                await session.rollback()
                raise
    
    return self.run_async(_check_updates())

@celery_app.task(bind=True, base=AsyncTask)
def process_law_updates(self, update_count: int):
    """Process newly discovered law updates"""
    logger.info(f"Processing {update_count} law updates")
    
    async def _process_updates():
        async with AsyncSessionLocal() as session:
            try:
                # Get unprocessed updates
                query = select(LawUpdate).where(
                    LawUpdate.processed == False
                ).order_by(LawUpdate.created_at.desc())
                
                result = await session.execute(query)
                updates = result.scalars().all()
                
                processed_count = 0
                
                for update in updates:
                    try:
                        # Analyze impact on existing compliance rules
                        affected_rules = await _analyze_law_impact(update, session)
                        
                        # Update compliance rules if necessary
                        if affected_rules:
                            await _update_compliance_rules(update, affected_rules, session)
                        
                        # Mark as processed
                        update.processed = True
                        update.processed_at = datetime.utcnow()
                        processed_count += 1
                        
                        # Record audit trail
                        await blockchain_service.record_audit(
                            document_id=f"law_update_{update.id}",
                            user_id="system",
                            action="law_update_processed",
                            additional_data={
                                "source": update.source,
                                "jurisdiction": update.jurisdiction,
                                "title": update.title,
                                "affected_rules": len(affected_rules)
                            },
                            session=session
                        )
                        
                    except Exception as e:
                        logger.error(f"Error processing update {update.id}: {str(e)}")
                        continue
                
                await session.commit()
                logger.info(f"Processed {processed_count} law updates")
                return {"status": "success", "processed": processed_count}
                
            except Exception as e:
                logger.error(f"Law updates processing failed: {str(e)}")
                await session.rollback()
                raise
    
    return self.run_async(_process_updates())

@celery_app.task(bind=True, base=AsyncTask)
def monitor_compliance(self):
    """Monitor documents for compliance issues"""
    logger.info("Starting compliance monitoring")
    
    async def _monitor():
        async with AsyncSessionLocal() as session:
            try:
                # Get documents that need compliance check
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                query = select(Document).where(
                    or_(
                        Document.last_compliance_check.is_(None),
                        Document.last_compliance_check < cutoff_time
                    )
                ).limit(100)  # Process in batches
                
                result = await session.execute(query)
                documents = result.scalars().all()
                
                compliance_issues = []
                
                for document in documents:
                    try:
                        # Run compliance analysis
                        issues = await _check_document_compliance(document, session)
                        
                        if issues:
                            compliance_issues.extend(issues)
                            
                            # Send notifications if critical issues found
                            critical_issues = [i for i in issues if i.get('severity') == 'critical']
                            if critical_issues:
                                send_compliance_alert.apply_async(
                                    args=[document.id, critical_issues]
                                )
                        
                        # Update last check time
                        document.last_compliance_check = datetime.utcnow()
                        
                    except Exception as e:
                        logger.error(f"Error checking compliance for document {document.id}: {str(e)}")
                        continue
                
                await session.commit()
                
                logger.info(f"Compliance monitoring completed. Found {len(compliance_issues)} issues")
                return {"status": "success", "issues_found": len(compliance_issues)}
                
            except Exception as e:
                logger.error(f"Compliance monitoring failed: {str(e)}")
                await session.rollback()
                raise
    
    return self.run_async(_monitor())

@celery_app.task(bind=True, base=AsyncTask)
def cleanup_expired_documents(self):
    """Clean up expired documents and temporary files"""
    logger.info("Starting document cleanup")
    
    async def _cleanup():
        async with AsyncSessionLocal() as session:
            try:
                # Get expired documents
                expiry_date = datetime.utcnow() - timedelta(days=settings.DOCUMENT_RETENTION_DAYS)
                
                query = select(Document).where(
                    and_(
                        Document.created_at < expiry_date,
                        Document.archived == False
                    )
                )
                
                result = await session.execute(query)
                documents = result.scalars().all()
                
                cleaned_count = 0
                
                for document in documents:
                    try:
                        # Archive document instead of deleting
                        document.archived = True
                        document.archived_at = datetime.utcnow()
                        
                        # Remove physical files if configured
                        if settings.DELETE_ARCHIVED_FILES:
                            if document.file_path and os.path.exists(document.file_path):
                                os.remove(document.file_path)
                        
                        # Record audit trail
                        await blockchain_service.record_audit(
                            document_id=str(document.id),
                            user_id="system",
                            action="document_archived",
                            additional_data={
                                "reason": "retention_policy",
                                "original_created_at": document.created_at.isoformat()
                            },
                            session=session
                        )
                        
                        cleaned_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error archiving document {document.id}: {str(e)}")
                        continue
                
                await session.commit()
                
                logger.info(f"Document cleanup completed. Archived {cleaned_count} documents")
                return {"status": "success", "archived": cleaned_count}
                
            except Exception as e:
                logger.error(f"Document cleanup failed: {str(e)}")
                await session.rollback()
                raise
    
    return self.run_async(_cleanup())

@celery_app.task(bind=True)
def send_compliance_alert(self, document_id: str, issues: List[Dict[str, Any]]):
    """Send compliance alert notifications"""
    logger.info(f"Sending compliance alert for document {document_id}")
    
    try:
        # Implementation would send email/SMS/webhook notifications
        # For now, just log the alert
        logger.warning(
            f"COMPLIANCE ALERT - Document {document_id} has {len(issues)} critical issues"
        )
        
        return {"status": "success", "document_id": document_id, "issues": len(issues)}
        
    except Exception as e:
        logger.error(f"Failed to send compliance alert: {str(e)}")
        raise

@celery_app.task(bind=True, base=AsyncTask)
def generate_usage_reports(self):
    """Generate monthly usage reports for billing"""
    logger.info("Generating usage reports")
    
    async def _generate_reports():
        async with AsyncSessionLocal() as session:
            try:
                # Get usage data for the previous month
                end_date = datetime.utcnow().replace(day=1)
                start_date = (end_date - timedelta(days=1)).replace(day=1)
                
                # Generate per-user usage reports
                query = select(User)
                result = await session.execute(query)
                users = result.scalars().all()
                
                reports_generated = 0
                
                for user in users:
                    try:
                        # Calculate usage metrics
                        usage_data = await _calculate_user_usage(user, start_date, end_date, session)
                        
                        # Store usage report
                        # Implementation would create UsageReport model
                        
                        reports_generated += 1
                        
                    except Exception as e:
                        logger.error(f"Error generating report for user {user.id}: {str(e)}")
                        continue
                
                logger.info(f"Generated {reports_generated} usage reports")
                return {"status": "success", "reports_generated": reports_generated}
                
            except Exception as e:
                logger.error(f"Usage report generation failed: {str(e)}")
                raise
    
    return self.run_async(_generate_reports())

@celery_app.task(bind=True, base=AsyncTask)
def update_law_database(self):
    """Weekly update of the legal knowledge base"""
    logger.info("Starting weekly law database update")
    
    async def _update_database():
        try:
            # Implementation would update legal knowledge base
            # from authoritative sources
            
            logger.info("Law database update completed")
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"Law database update failed: {str(e)}")
            raise
    
    return self.run_async(_update_database())

# Helper functions
async def _analyze_law_impact(update: LawUpdate, session: AsyncSession) -> List[ComplianceRule]:
    """Analyze impact of law update on existing compliance rules"""
    # Implementation would use NLP to match law updates to compliance rules
    return []

async def _update_compliance_rules(update: LawUpdate, rules: List[ComplianceRule], session: AsyncSession):
    """Update compliance rules based on law changes"""
    # Implementation would update rules based on law changes
    pass

async def _check_document_compliance(document: Document, session: AsyncSession) -> List[Dict[str, Any]]:
    """Check document for compliance issues"""
    # Implementation would run compliance checks
    return []

async def _calculate_user_usage(user: User, start_date: datetime, end_date: datetime, session: AsyncSession) -> Dict[str, Any]:
    """Calculate user usage metrics for billing"""
    # Implementation would calculate usage metrics
    return {}

class SchedulerService:
    """Service class for managing scheduled tasks"""
    
    @staticmethod
    def schedule_document_analysis(document_id: str, delay_seconds: int = 0):
        """Schedule document analysis task"""
        # Implementation would schedule analysis task
        pass
    
    @staticmethod
    def schedule_compliance_check(document_id: str, delay_seconds: int = 0):
        """Schedule compliance check task"""
        # Implementation would schedule compliance check
        pass
    
    @staticmethod
    def get_task_status(task_id: str) -> Dict[str, Any]:
        """Get status of scheduled task"""
        result = AsyncResult(task_id, app=celery_app)
        return {
            "task_id": task_id,
            "status": result.status,
            "result": result.result if result.ready() else None,
            "traceback": result.traceback if result.failed() else None
        }
    
    @staticmethod
    def cancel_task(task_id: str) -> bool:
        """Cancel scheduled task"""
        try:
            celery_app.control.revoke(task_id, terminate=True)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {str(e)}")
            return False

# Global scheduler service instance
scheduler_service = SchedulerService() 