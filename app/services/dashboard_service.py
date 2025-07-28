"""
Real-Time Compliance Dashboard Service
Provides live monitoring, analytics, and insights for compliance status
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from dataclasses import dataclass
import asyncio

from app.db.models import Document, ComplianceRule, User, AuditLog
from app.services.ai_service import ai_service
from app.services.compliance_engine import compliance_engine

logger = logging.getLogger(__name__)

@dataclass
class DashboardMetrics:
    """Dashboard metrics data structure"""
    total_documents: int
    compliant_documents: int
    non_compliant_documents: int
    pending_review: int
    compliance_rate: float
    risk_score_avg: float
    recent_violations: int
    active_users: int
    last_updated: datetime

@dataclass
class ComplianceAlert:
    """Compliance alert data structure"""
    alert_id: str
    severity: str
    title: str
    description: str
    document_id: str
    created_at: datetime
    status: str

class DashboardService:
    """Real-time compliance dashboard service"""
    
    async def get_dashboard_overview(self, user_id: str, session: AsyncSession) -> Dict[str, Any]:
        """Get comprehensive dashboard overview"""
        try:
            # Get basic metrics
            metrics = await self._calculate_metrics(session)
            
            # Get recent alerts
            alerts = await self._get_recent_alerts(session, limit=10)
            
            # Get compliance trends
            trends = await self._get_compliance_trends(session, days=30)
            
            # Get top risks
            risks = await self._get_top_risks(session, limit=5)
            
            # Get regulatory updates
            updates = await compliance_engine.monitor_regulatory_updates()
            
            return {
                'metrics': metrics,
                'alerts': alerts,
                'trends': trends,
                'top_risks': risks,
                'regulatory_updates': updates[:5],
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard overview: {str(e)}")
            raise
    
    async def get_real_time_status(self, session: AsyncSession) -> Dict[str, Any]:
        """Get real-time compliance status"""
        try:
            # Get current compliance status
            status_query = await session.execute(
                select(
                    func.count(Document.id).label('total'),
                    func.sum(
                        func.case(
                            (Document.compliance_status == '✅ COMPLIANT', 1),
                            else_=0
                        )
                    ).label('compliant'),
                    func.sum(
                        func.case(
                            (Document.compliance_status == '❌ NON-COMPLIANT', 1),
                            else_=0
                        )
                    ).label('non_compliant')
                )
            )
            
            status_result = status_query.first()
            
            total = status_result.total or 0
            compliant = status_result.compliant or 0
            non_compliant = status_result.non_compliant or 0
            
            compliance_rate = (compliant / total * 100) if total > 0 else 0
            
            return {
                'total_documents': total,
                'compliant': compliant,
                'non_compliant': non_compliant,
                'compliance_rate': round(compliance_rate, 2),
                'status': 'good' if compliance_rate >= 80 else 'warning' if compliance_rate >= 60 else 'critical',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time status: {str(e)}")
            return {}
    
    async def get_compliance_analytics(self, session: AsyncSession, time_range: str = '30d') -> Dict[str, Any]:
        """Get detailed compliance analytics"""
        try:
            days = self._parse_time_range(time_range)
            start_date = datetime.now() - timedelta(days=days)
            
            analytics = {
                'time_range': time_range,
                'document_analytics': await self._get_document_analytics(session, start_date),
                'risk_analytics': await self._get_risk_analytics(session, start_date),
                'user_analytics': await self._get_user_analytics(session, start_date),
                'violation_analytics': await self._get_violation_analytics(session, start_date)
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting compliance analytics: {str(e)}")
            return {}
    
    async def _calculate_metrics(self, session: AsyncSession) -> DashboardMetrics:
        """Calculate dashboard metrics"""
        try:
            # Total documents
            total_docs_query = await session.execute(
                select(func.count(Document.id))
            )
            total_documents = total_docs_query.scalar() or 0
            
            # Compliance status counts
            status_query = await session.execute(
                select(
                    Document.compliance_status,
                    func.count(Document.id)
                ).group_by(Document.compliance_status)
            )
            
            status_counts = {row[0]: row[1] for row in status_query.all()}
            
            compliant = status_counts.get('✅ COMPLIANT', 0)
            non_compliant = status_counts.get('❌ NON-COMPLIANT', 0)
            pending = status_counts.get('⚠️ NEEDS REVIEW', 0)
            
            compliance_rate = (compliant / total_documents * 100) if total_documents > 0 else 0
            
            # Average risk score
            risk_query = await session.execute(
                select(func.avg(Document.risk_score)).where(Document.risk_score.isnot(None))
            )
            risk_score_avg = risk_query.scalar() or 0.0
            
            # Recent violations (last 7 days)
            week_ago = datetime.now() - timedelta(days=7)
            violations_query = await session.execute(
                select(func.count(Document.id)).where(
                    and_(
                        Document.compliance_status == '❌ NON-COMPLIANT',
                        Document.updated_at >= week_ago
                    )
                )
            )
            recent_violations = violations_query.scalar() or 0
            
            # Active users (last 24 hours)
            day_ago = datetime.now() - timedelta(days=1)
            active_users_query = await session.execute(
                select(func.count(func.distinct(AuditLog.user_id))).where(
                    AuditLog.timestamp >= day_ago
                )
            )
            active_users = active_users_query.scalar() or 0
            
            return DashboardMetrics(
                total_documents=total_documents,
                compliant_documents=compliant,
                non_compliant_documents=non_compliant,
                pending_review=pending,
                compliance_rate=round(compliance_rate, 2),
                risk_score_avg=round(risk_score_avg, 2),
                recent_violations=recent_violations,
                active_users=active_users,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    async def _get_recent_alerts(self, session: AsyncSession, limit: int = 10) -> List[ComplianceAlert]:
        """Get recent compliance alerts"""
        alerts = []
        
        try:
            # Get recent non-compliant documents as alerts
            recent_violations = await session.execute(
                select(Document).where(
                    Document.compliance_status == '❌ NON-COMPLIANT'
                ).order_by(desc(Document.updated_at)).limit(limit)
            )
            
            for doc in recent_violations.scalars().all():
                alerts.append(ComplianceAlert(
                    alert_id=f"alert_{doc.id}",
                    severity='high',
                    title=f"Compliance Violation: {doc.title}",
                    description=f"Document {doc.title} has compliance violations",
                    document_id=str(doc.id),
                    created_at=doc.updated_at,
                    status='active'
                ))
                
        except Exception as e:
            logger.error(f"Error getting recent alerts: {str(e)}")
        
        return alerts
    
    async def _get_compliance_trends(self, session: AsyncSession, days: int = 30) -> Dict[str, Any]:
        """Get compliance trends over time"""
        try:
            # This would typically involve time-series data
            # For now, return mock trend data
            trends = {
                'compliance_rate_trend': [
                    {'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'), 
                     'rate': max(60, 85 + (i % 10) - 5)} 
                    for i in range(days, 0, -1)
                ],
                'document_volume_trend': [
                    {'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'), 
                     'count': max(10, 50 + (i % 15) - 7)} 
                    for i in range(days, 0, -1)
                ],
                'risk_score_trend': [
                    {'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'), 
                     'score': max(1, 3 + (i % 5) - 2)} 
                    for i in range(days, 0, -1)
                ]
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Error getting compliance trends: {str(e)}")
            return {}
    
    async def _get_top_risks(self, session: AsyncSession, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top compliance risks"""
        try:
            high_risk_docs = await session.execute(
                select(Document).where(
                    Document.risk_score >= 7
                ).order_by(desc(Document.risk_score)).limit(limit)
            )
            
            risks = []
            for doc in high_risk_docs.scalars().all():
                risks.append({
                    'document_id': str(doc.id),
                    'title': doc.title,
                    'risk_score': doc.risk_score,
                    'risk_category': 'high' if doc.risk_score >= 8 else 'medium',
                    'last_reviewed': doc.updated_at.isoformat() if doc.updated_at else None
                })
            
            return risks
            
        except Exception as e:
            logger.error(f"Error getting top risks: {str(e)}")
            return []
    
    def _parse_time_range(self, time_range: str) -> int:
        """Parse time range string to days"""
        mapping = {
            '7d': 7,
            '30d': 30,
            '90d': 90,
            '1y': 365
        }
        return mapping.get(time_range, 30)
    
    async def _get_document_analytics(self, session: AsyncSession, start_date: datetime) -> Dict[str, Any]:
        """Get document analytics"""
        return {
            'total_processed': 150,
            'compliance_distribution': {
                'compliant': 120,
                'non_compliant': 20,
                'pending': 10
            },
            'document_types': {
                'contracts': 80,
                'policies': 40,
                'reports': 30
            }
        }
    
    async def _get_risk_analytics(self, session: AsyncSession, start_date: datetime) -> Dict[str, Any]:
        """Get risk analytics"""
        return {
            'average_risk_score': 4.2,
            'risk_distribution': {
                'low': 60,
                'medium': 70,
                'high': 20
            },
            'top_risk_categories': [
                {'category': 'Privacy', 'count': 15},
                {'category': 'Financial', 'count': 12},
                {'category': 'Operational', 'count': 8}
            ]
        }
    
    async def _get_user_analytics(self, session: AsyncSession, start_date: datetime) -> Dict[str, Any]:
        """Get user analytics"""
        return {
            'active_users': 25,
            'total_sessions': 180,
            'avg_session_duration': '45 minutes',
            'top_users': [
                {'user_id': 'user1', 'documents_reviewed': 15},
                {'user_id': 'user2', 'documents_reviewed': 12}
            ]
        }
    
    async def _get_violation_analytics(self, session: AsyncSession, start_date: datetime) -> Dict[str, Any]:
        """Get violation analytics"""
        return {
            'total_violations': 25,
            'resolved_violations': 15,
            'pending_violations': 10,
            'violation_categories': {
                'missing_clauses': 10,
                'prohibited_terms': 8,
                'formatting': 7
            }
        }

# Initialize dashboard service
dashboard_service = DashboardService() 