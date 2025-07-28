"""
Document Version Control System
Tracks document changes, maintains history, and provides version comparison
"""

import logging
import hashlib
import difflib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, and_
from uuid import uuid4
import json
import re

from app.db.models import Document, User, AuditLog
from app.services.blockchain_service import blockchain_service
from config import settings

logger = logging.getLogger(__name__)

@dataclass
class DocumentVersion:
    """Document version information"""
    version_id: str
    document_id: str
    version_number: str
    content_hash: str
    file_path: str
    changes_summary: str
    created_by: str
    created_at: datetime
    file_size: int
    metadata: Dict[str, Any]
    tags: List[str]
    is_major_version: bool
    parent_version_id: Optional[str]

@dataclass
class DocumentDifference:
    """Document difference information"""
    version_from: str
    version_to: str
    change_type: str
    changes: List[Dict[str, Any]]
    similarity_score: float
    summary: str
    affected_sections: List[str]

@dataclass
class VersionHistory:
    """Complete version history for a document"""
    document_id: str
    current_version: str
    total_versions: int
    created_at: datetime
    last_modified: datetime
    versions: List[DocumentVersion]
    contributors: List[str]

class DocumentVersionControl:
    """Advanced document version control system"""
    
    def __init__(self):
        self.version_storage_path = settings.UPLOAD_DIR + "/versions"
        self.supported_formats = ['.pdf', '.docx', '.txt', '.md']
        self.auto_version_triggers = [
            'significant_change',
            'compliance_status_change',
            'scheduled_review',
            'manual_save'
        ]
    
    async def create_document_version(self,
                                    document_id: str,
                                    file_path: str,
                                    user_id: str,
                                    changes_summary: str,
                                    version_type: str = 'minor',
                                    metadata: Optional[Dict[str, Any]] = None,
                                    session: AsyncSession) -> DocumentVersion:
        """Create a new version of a document"""
        
        logger.info(f"Creating new version for document {document_id}")
        
        try:
            # Get current document
            doc_result = await session.execute(
                select(Document).where(Document.id == document_id)
            )
            document = doc_result.scalar_one_or_none()
            
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            # Get latest version number
            latest_version = await self._get_latest_version_number(document_id, session)
            
            # Generate new version number
            new_version_number = self._increment_version_number(latest_version, version_type)
            
            # Calculate content hash
            content_hash = await self._calculate_content_hash(file_path)
            
            # Check if content actually changed
            if await self._is_duplicate_content(document_id, content_hash, session):
                logger.info(f"No content changes detected for document {document_id}")
                return await self._get_latest_version(document_id, session)
            
            # Create version record
            version_id = str(uuid4())
            version_file_path = await self._store_version_file(
                file_path, document_id, new_version_number
            )
            
            file_size = await self._get_file_size(file_path)
            
            version = DocumentVersion(
                version_id=version_id,
                document_id=document_id,
                version_number=new_version_number,
                content_hash=content_hash,
                file_path=version_file_path,
                changes_summary=changes_summary,
                created_by=user_id,
                created_at=datetime.now(),
                file_size=file_size,
                metadata=metadata or {},
                tags=self._extract_version_tags(changes_summary, metadata),
                is_major_version=version_type == 'major',
                parent_version_id=await self._get_parent_version_id(document_id, session)
            )
            
            # Store version in database
            await self._store_version_record(version, session)
            
            # Update document's current version
            await self._update_document_current_version(document, version, session)
            
            # Record audit trail
            await self._record_version_audit(version, user_id, session)
            
            # Analyze changes if previous version exists
            if version.parent_version_id:
                await self._analyze_version_changes(version, session)
            
            logger.info(f"Created version {new_version_number} for document {document_id}")
            return version
            
        except Exception as e:
            logger.error(f"Error creating document version: {str(e)}")
            raise
    
    async def get_document_versions(self,
                                  document_id: str,
                                  limit: int = 50,
                                  include_content: bool = False,
                                  session: AsyncSession) -> VersionHistory:
        """Get version history for a document"""
        
        logger.info(f"Retrieving version history for document {document_id}")
        
        try:
            # Get document
            doc_result = await session.execute(
                select(Document).where(Document.id == document_id)
            )
            document = doc_result.scalar_one_or_none()
            
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            # Get versions from audit logs and metadata
            versions = await self._load_versions_from_storage(document_id, limit, session)
            
            # Get contributors
            contributors = await self._get_document_contributors(document_id, session)
            
            # Build version history
            history = VersionHistory(
                document_id=document_id,
                current_version=versions[0].version_number if versions else "1.0",
                total_versions=len(versions),
                created_at=document.created_at,
                last_modified=document.updated_at,
                versions=versions,
                contributors=contributors
            )
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting document versions: {str(e)}")
            raise
    
    async def compare_versions(self,
                             document_id: str,
                             version_from: str,
                             version_to: str,
                             comparison_type: str = 'detailed',
                             session: AsyncSession) -> DocumentDifference:
        """Compare two versions of a document"""
        
        logger.info(f"Comparing versions {version_from} to {version_to} for document {document_id}")
        
        try:
            # Get both versions
            version_from_obj = await self._get_version_by_number(document_id, version_from, session)
            version_to_obj = await self._get_version_by_number(document_id, version_to, session)
            
            if not version_from_obj or not version_to_obj:
                raise ValueError("One or both versions not found")
            
            # Load content for both versions
            content_from = await self._load_version_content(version_from_obj)
            content_to = await self._load_version_content(version_to_obj)
            
            # Perform comparison
            if comparison_type == 'detailed':
                changes = await self._detailed_text_comparison(content_from, content_to)
            elif comparison_type == 'summary':
                changes = await self._summary_comparison(content_from, content_to)
            else:
                changes = await self._basic_comparison(content_from, content_to)
            
            # Calculate similarity score
            similarity_score = self._calculate_similarity_score(content_from, content_to)
            
            # Generate summary
            summary = await self._generate_comparison_summary(changes, similarity_score)
            
            # Identify affected sections
            affected_sections = self._identify_affected_sections(changes)
            
            difference = DocumentDifference(
                version_from=version_from,
                version_to=version_to,
                change_type=self._classify_change_type(changes, similarity_score),
                changes=changes,
                similarity_score=similarity_score,
                summary=summary,
                affected_sections=affected_sections
            )
            
            # Record comparison audit
            await self._record_comparison_audit(document_id, difference, session)
            
            return difference
            
        except Exception as e:
            logger.error(f"Error comparing versions: {str(e)}")
            raise
    
    async def restore_version(self,
                            document_id: str,
                            version_number: str,
                            user_id: str,
                            reason: str,
                            session: AsyncSession) -> DocumentVersion:
        """Restore a document to a previous version"""
        
        logger.info(f"Restoring document {document_id} to version {version_number}")
        
        try:
            # Get the version to restore
            version_to_restore = await self._get_version_by_number(document_id, version_number, session)
            
            if not version_to_restore:
                raise ValueError(f"Version {version_number} not found")
            
            # Create new version from restored content
            restored_version = await self.create_document_version(
                document_id=document_id,
                file_path=version_to_restore.file_path,
                user_id=user_id,
                changes_summary=f"Restored from version {version_number}: {reason}",
                version_type='major',
                metadata={
                    'restoration': True,
                    'restored_from': version_number,
                    'restoration_reason': reason,
                    'restored_at': datetime.now().isoformat()
                },
                session=session
            )
            
            # Record restoration audit
            await self._record_restoration_audit(
                document_id, version_number, restored_version.version_number, user_id, reason, session
            )
            
            logger.info(f"Successfully restored document {document_id} to version {version_number}")
            return restored_version
            
        except Exception as e:
            logger.error(f"Error restoring version: {str(e)}")
            raise
    
    async def auto_version_check(self,
                               document_id: str,
                               current_content: str,
                               trigger: str,
                               session: AsyncSession) -> Optional[DocumentVersion]:
        """Automatically check if a new version should be created"""
        
        try:
            # Get latest version
            latest_version = await self._get_latest_version(document_id, session)
            
            if not latest_version:
                return None
            
            # Load current version content
            latest_content = await self._load_version_content(latest_version)
            
            # Determine if versioning is needed
            needs_version = await self._should_create_version(
                latest_content, current_content, trigger
            )
            
            if needs_version:
                # Create temporary file for current content
                temp_file_path = await self._create_temp_file(current_content, document_id)
                
                # Create new version
                return await self.create_document_version(
                    document_id=document_id,
                    file_path=temp_file_path,
                    user_id="system",
                    changes_summary=f"Auto-version created due to {trigger}",
                    version_type='minor',
                    metadata={'auto_version': True, 'trigger': trigger},
                    session=session
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in auto version check: {str(e)}")
            return None
    
    async def merge_versions(self,
                           document_id: str,
                           base_version: str,
                           version_a: str,
                           version_b: str,
                           user_id: str,
                           merge_strategy: str = 'manual',
                           session: AsyncSession) -> DocumentVersion:
        """Merge two versions of a document"""
        
        logger.info(f"Merging versions {version_a} and {version_b} for document {document_id}")
        
        try:
            # Get all three versions
            base = await self._get_version_by_number(document_id, base_version, session)
            ver_a = await self._get_version_by_number(document_id, version_a, session)
            ver_b = await self._get_version_by_number(document_id, version_b, session)
            
            if not all([base, ver_a, ver_b]):
                raise ValueError("One or more versions not found for merge")
            
            # Load content
            base_content = await self._load_version_content(base)
            content_a = await self._load_version_content(ver_a)
            content_b = await self._load_version_content(ver_b)
            
            # Perform merge
            if merge_strategy == 'auto':
                merged_content = await self._auto_merge_content(base_content, content_a, content_b)
            else:
                merged_content = await self._manual_merge_content(base_content, content_a, content_b)
            
            # Create temporary file for merged content
            temp_file_path = await self._create_temp_file(merged_content, document_id)
            
            # Create merged version
            merged_version = await self.create_document_version(
                document_id=document_id,
                file_path=temp_file_path,
                user_id=user_id,
                changes_summary=f"Merged versions {version_a} and {version_b}",
                version_type='major',
                metadata={
                    'merge': True,
                    'base_version': base_version,
                    'merged_versions': [version_a, version_b],
                    'merge_strategy': merge_strategy
                },
                session=session
            )
            
            # Record merge audit
            await self._record_merge_audit(document_id, [version_a, version_b], merged_version.version_number, user_id, session)
            
            return merged_version
            
        except Exception as e:
            logger.error(f"Error merging versions: {str(e)}")
            raise
    
    async def get_version_analytics(self,
                                  document_id: str,
                                  time_period: int = 30,
                                  session: AsyncSession) -> Dict[str, Any]:
        """Get analytics for document versions"""
        
        try:
            # Get version history
            history = await self.get_document_versions(document_id, limit=100, session=session)
            
            # Calculate analytics
            analytics = {
                'total_versions': history.total_versions,
                'version_frequency': self._calculate_version_frequency(history.versions, time_period),
                'major_versions': len([v for v in history.versions if v.is_major_version]),
                'contributors': len(history.contributors),
                'average_changes_per_version': self._calculate_average_changes(history.versions),
                'most_active_contributor': self._get_most_active_contributor(history.versions),
                'version_size_trend': self._analyze_size_trend(history.versions),
                'change_patterns': self._analyze_change_patterns(history.versions),
                'collaboration_metrics': self._calculate_collaboration_metrics(history.versions)
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting version analytics: {str(e)}")
            return {}
    
    # Private helper methods
    
    async def _get_latest_version_number(self, document_id: str, session: AsyncSession) -> str:
        """Get the latest version number for a document"""
        # This would query version storage/database
        # For now, return default
        return "0.0"
    
    def _increment_version_number(self, current_version: str, version_type: str) -> str:
        """Increment version number based on type"""
        try:
            if current_version == "0.0":
                return "1.0"
            
            parts = current_version.split('.')
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            
            if version_type == 'major':
                return f"{major + 1}.0"
            else:
                return f"{major}.{minor + 1}"
                
        except Exception:
            return "1.0"
    
    async def _calculate_content_hash(self, file_path: str) -> str:
        """Calculate hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception:
            return hashlib.sha256(file_path.encode()).hexdigest()
    
    async def _is_duplicate_content(self, document_id: str, content_hash: str, session: AsyncSession) -> bool:
        """Check if content hash already exists for document"""
        # This would check against stored version hashes
        return False
    
    async def _store_version_file(self, source_path: str, document_id: str, version_number: str) -> str:
        """Store version file in version storage"""
        import shutil
        import os
        
        # Create version storage directory
        version_dir = f"{self.version_storage_path}/{document_id}"
        os.makedirs(version_dir, exist_ok=True)
        
        # Copy file with version number
        file_ext = os.path.splitext(source_path)[1]
        version_file_path = f"{version_dir}/v{version_number}{file_ext}"
        
        shutil.copy2(source_path, version_file_path)
        return version_file_path
    
    async def _get_file_size(self, file_path: str) -> int:
        """Get file size in bytes"""
        try:
            import os
            return os.path.getsize(file_path)
        except Exception:
            return 0
    
    def _extract_version_tags(self, changes_summary: str, metadata: Optional[Dict[str, Any]]) -> List[str]:
        """Extract tags from changes and metadata"""
        tags = []
        
        # Extract from summary
        if 'compliance' in changes_summary.lower():
            tags.append('compliance')
        if 'major' in changes_summary.lower():
            tags.append('major_change')
        if 'review' in changes_summary.lower():
            tags.append('review')
        
        # Extract from metadata
        if metadata:
            if metadata.get('auto_version'):
                tags.append('auto_generated')
            if metadata.get('restoration'):
                tags.append('restoration')
            if metadata.get('merge'):
                tags.append('merge')
        
        return tags
    
    async def _get_parent_version_id(self, document_id: str, session: AsyncSession) -> Optional[str]:
        """Get parent version ID"""
        latest = await self._get_latest_version(document_id, session)
        return latest.version_id if latest else None
    
    async def _store_version_record(self, version: DocumentVersion, session: AsyncSession):
        """Store version record in database/storage"""
        # This would store the version metadata in database
        pass
    
    async def _update_document_current_version(self, document: Document, version: DocumentVersion, session: AsyncSession):
        """Update document's current version reference"""
        # Update document metadata with current version
        if not document.document_metadata:
            document.document_metadata = {}
        
        document.document_metadata['current_version'] = version.version_number
        document.document_metadata['version_id'] = version.version_id
        document.updated_at = datetime.now()
        
        await session.commit()
    
    async def _record_version_audit(self, version: DocumentVersion, user_id: str, session: AsyncSession):
        """Record version creation in audit trail"""
        await blockchain_service.record_audit(
            document_id=version.document_id,
            user_id=user_id,
            action="version_created",
            additional_data={
                'version_id': version.version_id,
                'version_number': version.version_number,
                'content_hash': version.content_hash,
                'file_size': version.file_size,
                'changes_summary': version.changes_summary,
                'is_major_version': version.is_major_version
            },
            session=session
        )
    
    async def _analyze_version_changes(self, version: DocumentVersion, session: AsyncSession):
        """Analyze changes in this version"""
        # This would perform detailed change analysis
        pass
    
    async def _load_versions_from_storage(self, document_id: str, limit: int, session: AsyncSession) -> List[DocumentVersion]:
        """Load versions from storage"""
        # Mock implementation - would load from actual storage
        return []
    
    async def _get_document_contributors(self, document_id: str, session: AsyncSession) -> List[str]:
        """Get list of contributors to document"""
        # Query audit logs for contributors
        return []
    
    async def _get_version_by_number(self, document_id: str, version_number: str, session: AsyncSession) -> Optional[DocumentVersion]:
        """Get version by number"""
        # Mock implementation
        return None
    
    async def _load_version_content(self, version: DocumentVersion) -> str:
        """Load content from version file"""
        try:
            with open(version.file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return ""
    
    async def _detailed_text_comparison(self, content_from: str, content_to: str) -> List[Dict[str, Any]]:
        """Perform detailed text comparison"""
        changes = []
        
        # Use difflib for line-by-line comparison
        diff = list(difflib.unified_diff(
            content_from.splitlines(keepends=True),
            content_to.splitlines(keepends=True),
            lineterm='',
            n=3
        ))
        
        current_change = None
        for line in diff:
            if line.startswith('@@'):
                if current_change:
                    changes.append(current_change)
                current_change = {
                    'type': 'modification',
                    'location': line,
                    'added_lines': [],
                    'removed_lines': [],
                    'context_lines': []
                }
            elif line.startswith('+') and current_change:
                current_change['added_lines'].append(line[1:])
            elif line.startswith('-') and current_change:
                current_change['removed_lines'].append(line[1:])
            elif line.startswith(' ') and current_change:
                current_change['context_lines'].append(line[1:])
        
        if current_change:
            changes.append(current_change)
        
        return changes
    
    async def _summary_comparison(self, content_from: str, content_to: str) -> List[Dict[str, Any]]:
        """Perform summary comparison"""
        changes = []
        
        # Basic statistics
        changes.append({
            'type': 'statistics',
            'from_length': len(content_from),
            'to_length': len(content_to),
            'character_change': len(content_to) - len(content_from),
            'from_words': len(content_from.split()),
            'to_words': len(content_to.split()),
            'word_change': len(content_to.split()) - len(content_from.split())
        })
        
        return changes
    
    async def _basic_comparison(self, content_from: str, content_to: str) -> List[Dict[str, Any]]:
        """Perform basic comparison"""
        return [{
            'type': 'basic',
            'changed': content_from != content_to,
            'hash_from': hashlib.md5(content_from.encode()).hexdigest(),
            'hash_to': hashlib.md5(content_to.encode()).hexdigest()
        }]
    
    def _calculate_similarity_score(self, content_from: str, content_to: str) -> float:
        """Calculate similarity score between two texts"""
        if not content_from and not content_to:
            return 1.0
        if not content_from or not content_to:
            return 0.0
        
        # Use simple sequence matcher
        matcher = difflib.SequenceMatcher(None, content_from, content_to)
        return matcher.ratio()
    
    async def _generate_comparison_summary(self, changes: List[Dict[str, Any]], similarity_score: float) -> str:
        """Generate human-readable comparison summary"""
        if similarity_score > 0.95:
            return f"Minor changes detected (similarity: {similarity_score:.1%})"
        elif similarity_score > 0.8:
            return f"Moderate changes detected (similarity: {similarity_score:.1%})"
        elif similarity_score > 0.5:
            return f"Significant changes detected (similarity: {similarity_score:.1%})"
        else:
            return f"Major changes detected (similarity: {similarity_score:.1%})"
    
    def _identify_affected_sections(self, changes: List[Dict[str, Any]]) -> List[str]:
        """Identify sections affected by changes"""
        sections = []
        
        for change in changes:
            if change.get('type') == 'modification':
                # Extract section information from context
                context_lines = change.get('context_lines', [])
                for line in context_lines:
                    if any(header in line.lower() for header in ['section', 'clause', 'paragraph', 'article']):
                        sections.append(line.strip())
        
        return list(set(sections))
    
    def _classify_change_type(self, changes: List[Dict[str, Any]], similarity_score: float) -> str:
        """Classify the type of changes"""
        if similarity_score > 0.95:
            return 'minor_edit'
        elif similarity_score > 0.8:
            return 'moderate_revision'
        elif similarity_score > 0.5:
            return 'significant_update'
        else:
            return 'major_rewrite'
    
    async def _record_comparison_audit(self, document_id: str, difference: DocumentDifference, session: AsyncSession):
        """Record comparison in audit trail"""
        await blockchain_service.record_audit(
            document_id=document_id,
            user_id="system",
            action="version_comparison",
            additional_data={
                'version_from': difference.version_from,
                'version_to': difference.version_to,
                'change_type': difference.change_type,
                'similarity_score': difference.similarity_score,
                'changes_count': len(difference.changes)
            },
            session=session
        )
    
    async def _get_latest_version(self, document_id: str, session: AsyncSession) -> Optional[DocumentVersion]:
        """Get latest version of document"""
        # Mock implementation
        return None
    
    async def _should_create_version(self, latest_content: str, current_content: str, trigger: str) -> bool:
        """Determine if a new version should be created"""
        # Calculate similarity
        similarity = self._calculate_similarity_score(latest_content, current_content)
        
        # Version creation thresholds based on trigger
        thresholds = {
            'significant_change': 0.9,
            'compliance_status_change': 0.95,
            'scheduled_review': 0.98,
            'manual_save': 0.99
        }
        
        threshold = thresholds.get(trigger, 0.95)
        return similarity < threshold
    
    async def _create_temp_file(self, content: str, document_id: str) -> str:
        """Create temporary file with content"""
        import tempfile
        import os
        
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"temp_{document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return temp_file_path
    
    async def _record_restoration_audit(self, document_id: str, restored_from: str, new_version: str, user_id: str, reason: str, session: AsyncSession):
        """Record restoration in audit trail"""
        await blockchain_service.record_audit(
            document_id=document_id,
            user_id=user_id,
            action="version_restored",
            additional_data={
                'restored_from_version': restored_from,
                'new_version': new_version,
                'restoration_reason': reason,
                'restored_at': datetime.now().isoformat()
            },
            session=session
        )
    
    async def _auto_merge_content(self, base_content: str, content_a: str, content_b: str) -> str:
        """Automatically merge content"""
        # Simple merge strategy - in production would use more sophisticated algorithms
        return content_b  # For now, prefer content_b
    
    async def _manual_merge_content(self, base_content: str, content_a: str, content_b: str) -> str:
        """Manual merge content (would require UI for conflict resolution)"""
        # For now, return content_b - in production would handle conflicts
        return content_b
    
    async def _record_merge_audit(self, document_id: str, merged_versions: List[str], new_version: str, user_id: str, session: AsyncSession):
        """Record merge in audit trail"""
        await blockchain_service.record_audit(
            document_id=document_id,
            user_id=user_id,
            action="versions_merged",
            additional_data={
                'merged_versions': merged_versions,
                'new_version': new_version,
                'merged_at': datetime.now().isoformat()
            },
            session=session
        )
    
    def _calculate_version_frequency(self, versions: List[DocumentVersion], time_period: int) -> float:
        """Calculate version creation frequency"""
        recent_versions = [
            v for v in versions 
            if v.created_at > datetime.now() - timedelta(days=time_period)
        ]
        return len(recent_versions) / time_period
    
    def _calculate_average_changes(self, versions: List[DocumentVersion]) -> float:
        """Calculate average changes per version"""
        if not versions:
            return 0.0
        
        # Mock calculation - would analyze actual changes
        return len(versions) * 1.5
    
    def _get_most_active_contributor(self, versions: List[DocumentVersion]) -> str:
        """Get most active contributor"""
        if not versions:
            return "Unknown"
        
        contributors = {}
        for version in versions:
            contributors[version.created_by] = contributors.get(version.created_by, 0) + 1
        
        return max(contributors.items(), key=lambda x: x[1])[0] if contributors else "Unknown"
    
    def _analyze_size_trend(self, versions: List[DocumentVersion]) -> Dict[str, Any]:
        """Analyze document size trend"""
        if len(versions) < 2:
            return {'trend': 'insufficient_data'}
        
        sizes = [v.file_size for v in versions[-10:]]  # Last 10 versions
        
        if sizes[-1] > sizes[0]:
            trend = 'increasing'
        elif sizes[-1] < sizes[0]:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'current_size': sizes[-1],
            'size_change': sizes[-1] - sizes[0],
            'average_size': sum(sizes) / len(sizes)
        }
    
    def _analyze_change_patterns(self, versions: List[DocumentVersion]) -> Dict[str, Any]:
        """Analyze change patterns"""
        patterns = {
            'auto_versions': len([v for v in versions if 'auto_version' in v.metadata]),
            'major_versions': len([v for v in versions if v.is_major_version]),
            'restorations': len([v for v in versions if v.metadata.get('restoration')]),
            'merges': len([v for v in versions if v.metadata.get('merge')])
        }
        
        return patterns
    
    def _calculate_collaboration_metrics(self, versions: List[DocumentVersion]) -> Dict[str, Any]:
        """Calculate collaboration metrics"""
        if not versions:
            return {'collaboration_score': 0}
        
        unique_contributors = len(set(v.created_by for v in versions))
        total_versions = len(versions)
        
        collaboration_score = min(1.0, unique_contributors / max(1, total_versions / 5))
        
        return {
            'collaboration_score': collaboration_score,
            'unique_contributors': unique_contributors,
            'versions_per_contributor': total_versions / max(1, unique_contributors)
        }

# Initialize the document version control service
document_version_control = DocumentVersionControl() 