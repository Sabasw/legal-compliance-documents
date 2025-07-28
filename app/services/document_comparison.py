"""
Document Comparison Service
Provides advanced document comparison and difference analysis
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import difflib
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class DocumentComparison:
    """Document comparison result"""
    document1_id: str
    document2_id: str
    similarity_score: float
    differences: List[Dict[str, Any]]
    summary: str
    created_at: datetime

class DocumentComparisonService:
    """Service for comparing legal documents"""
    
    def __init__(self):
        self.comparison_cache = {}
    
    async def compare_documents(self, 
                              doc1_content: str, 
                              doc2_content: str,
                              doc1_id: str = "doc1",
                              doc2_id: str = "doc2") -> DocumentComparison:
        """Compare two documents and return detailed analysis"""
        
        try:
            # Calculate similarity score
            similarity_score = self._calculate_similarity(doc1_content, doc2_content)
            
            # Find differences
            differences = self._find_differences(doc1_content, doc2_content)
            
            # Generate summary
            summary = self._generate_summary(similarity_score, differences)
            
            comparison = DocumentComparison(
                document1_id=doc1_id,
                document2_id=doc2_id,
                similarity_score=similarity_score,
                differences=differences,
                summary=summary,
                created_at=datetime.now()
            )
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing documents: {str(e)}")
            raise
    
    def _calculate_similarity(self, doc1: str, doc2: str) -> float:
        """Calculate similarity score between documents"""
        if not doc1 and not doc2:
            return 1.0
        if not doc1 or not doc2:
            return 0.0
        
        matcher = difflib.SequenceMatcher(None, doc1, doc2)
        return matcher.ratio()
    
    def _find_differences(self, doc1: str, doc2: str) -> List[Dict[str, Any]]:
        """Find detailed differences between documents"""
        differences = []
        
        diff = list(difflib.unified_diff(
            doc1.splitlines(keepends=True),
            doc2.splitlines(keepends=True),
            lineterm='',
            n=3
        ))
        
        current_diff = None
        for line in diff:
            if line.startswith('@@'):
                if current_diff:
                    differences.append(current_diff)
                current_diff = {
                    'type': 'modification',
                    'location': line,
                    'added_lines': [],
                    'removed_lines': [],
                    'context_lines': []
                }
            elif line.startswith('+') and current_diff:
                current_diff['added_lines'].append(line[1:])
            elif line.startswith('-') and current_diff:
                current_diff['removed_lines'].append(line[1:])
            elif line.startswith(' ') and current_diff:
                current_diff['context_lines'].append(line[1:])
        
        if current_diff:
            differences.append(current_diff)
        
        return differences
    
    def _generate_summary(self, similarity_score: float, differences: List[Dict[str, Any]]) -> str:
        """Generate comparison summary"""
        if similarity_score > 0.95:
            return f"Documents are very similar (similarity: {similarity_score:.1%})"
        elif similarity_score > 0.8:
            return f"Documents have moderate differences (similarity: {similarity_score:.1%})"
        elif similarity_score > 0.5:
            return f"Documents have significant differences (similarity: {similarity_score:.1%})"
        else:
            return f"Documents are very different (similarity: {similarity_score:.1%})"

# Initialize service
document_comparison_service = DocumentComparisonService() 