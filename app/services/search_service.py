"""
Semantic Search Service
Advanced legal document search using Elasticsearch and NLP
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from elasticsearch import Elasticsearch, NotFoundError
from sentence_transformers import SentenceTransformer
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from app.db.models import Document, User
from config import settings
from app.services.ai_service import ai_service

logger = logging.getLogger(__name__)

class SemanticSearchService:
    def __init__(self):
        self.es = None
        self.sentence_transformer = None
        self.index_name = "legal_documents"
        self._initialize_search_engine()
    
    def _initialize_search_engine(self):
        """Initialize Elasticsearch and semantic search models"""
        try:
            logger.info("Initializing semantic search engine...")
            
            # Initialize Elasticsearch
            es_host = settings.ELASTICSEARCH_HOST or "localhost"
            es_port = settings.ELASTICSEARCH_PORT or 9200
            
            try:
                self.es = Elasticsearch(
                    hosts=[{"host": es_host, "port": es_port}],
                    timeout=30,
                    max_retries=3,
                    retry_on_timeout=True
                )
                
                # Test connection
                if self.es.ping():
                    logger.info("Elasticsearch connected successfully")
                    self._setup_indices()
                else:
                    logger.warning("Elasticsearch connection failed, using fallback search")
                    self.es = None
                    
            except Exception as e:
                logger.warning(f"Elasticsearch initialization failed: {str(e)}")
                self.es = None
            
            # Initialize sentence transformer for semantic embeddings
            try:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformer loaded for semantic search")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {str(e)}")
            
            logger.info("Semantic search engine initialization completed")
            
        except Exception as e:
            logger.error(f"Search engine initialization failed: {str(e)}")
    
    def _setup_indices(self):
        """Setup Elasticsearch indices with proper mappings"""
        try:
            if not self.es:
                return
            
            # Document index mapping
            document_mapping = {
                "mappings": {
                    "properties": {
                        "document_id": {"type": "keyword"},
                        "title": {
                            "type": "text",
                            "analyzer": "english",
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        "content": {
                            "type": "text",
                            "analyzer": "english"
                        },
                        "summary": {
                            "type": "text",
                            "analyzer": "english"
                        },
                        "document_type": {"type": "keyword"},
                        "jurisdiction": {"type": "keyword"},
                        "case_type": {"type": "keyword"},
                        "tags": {"type": "keyword"},
                        "created_at": {"type": "date"},
                        "user_id": {"type": "keyword"},
                        "compliance_score": {"type": "float"},
                        "risk_level": {"type": "keyword"},
                        "legal_entities": {
                            "type": "nested",
                            "properties": {
                                "text": {"type": "keyword"},
                                "label": {"type": "keyword"},
                                "confidence": {"type": "float"}
                            }
                        },
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 384  # MiniLM embedding dimension
                        },
                        "file_path": {"type": "keyword"},
                        "file_size": {"type": "long"},
                        "language": {"type": "keyword"}
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "legal_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": [
                                    "lowercase",
                                    "stop",
                                    "legal_synonyms",
                                    "stemmer"
                                ]
                            }
                        },
                        "filter": {
                            "legal_synonyms": {
                                "type": "synonym",
                                "synonyms": [
                                    "contract,agreement",
                                    "liability,responsibility",
                                    "breach,violation",
                                    "damages,compensation",
                                    "plaintiff,complainant",
                                    "defendant,respondent"
                                ]
                            }
                        }
                    }
                }
            }
            
            # Create index if it doesn't exist
            if not self.es.indices.exists(index=self.index_name):
                self.es.indices.create(index=self.index_name, body=document_mapping)
                logger.info(f"Created Elasticsearch index: {self.index_name}")
            else:
                logger.info(f"Elasticsearch index already exists: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Failed to setup Elasticsearch indices: {str(e)}")
    
    async def index_document(
        self,
        document_id: str,
        title: str,
        content: str,
        document_type: str = None,
        jurisdiction: str = "AU",
        case_type: str = None,
        tags: List[str] = None,
        user_id: str = None,
        session: AsyncSession = None
    ) -> Dict[str, Any]:
        """Index a document for semantic search"""
        try:
            logger.info(f"Indexing document {document_id}")
            
            # Generate summary using AI service
            summary = ai_service.summarize_text(content)
            
            # Extract entities
            entities = ai_service.extract_entities(content)
            
            # Generate semantic embedding
            embedding = None
            if self.sentence_transformer:
                text_for_embedding = f"{title} {summary} {content[:1000]}"
                embedding = self.sentence_transformer.encode(text_for_embedding).tolist()
            
            # Prepare document for indexing
            doc_body = {
                "document_id": document_id,
                "title": title,
                "content": content,
                "summary": summary,
                "document_type": document_type,
                "jurisdiction": jurisdiction,
                "case_type": case_type,
                "tags": tags or [],
                "created_at": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "legal_entities": [
                    {
                        "text": entity[0],
                        "label": entity[1],
                        "confidence": 0.8
                    }
                    for entity in entities
                ],
                "language": "en"
            }
            
            if embedding:
                doc_body["embedding"] = embedding
            
            # Index in Elasticsearch
            if self.es:
                try:
                    response = self.es.index(
                        index=self.index_name,
                        id=document_id,
                        body=doc_body
                    )
                    
                    logger.info(f"Document {document_id} indexed successfully")
                    return {
                        "status": "success",
                        "document_id": document_id,
                        "elasticsearch_result": response.get("result"),
                        "entities_extracted": len(entities),
                        "has_embedding": embedding is not None
                    }
                    
                except Exception as e:
                    logger.error(f"Elasticsearch indexing failed: {str(e)}")
                    return {
                        "status": "error",
                        "error": str(e),
                        "fallback": "document_not_indexed"
                    }
            else:
                logger.warning("Elasticsearch not available, document not indexed")
                return {
                    "status": "warning",
                    "message": "Elasticsearch not available"
                }
                
        except Exception as e:
            logger.error(f"Document indexing failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def semantic_search(
        self,
        query: str,
        document_type: str = None,
        jurisdiction: str = None,
        case_type: str = None,
        user_id: str = None,
        limit: int = 10,
        min_score: float = 0.5,
        session: AsyncSession = None
    ) -> Dict[str, Any]:
        """Perform semantic search on legal documents"""
        try:
            logger.info(f"Performing semantic search for: {query}")
            start_time = datetime.utcnow()
            
            # Use database search with semantic enhancements
            return await self._enhanced_database_search(
                query, document_type, jurisdiction, case_type,
                user_id, limit, session
            )
                
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return {
                "query": query,
                "results": [],
                "total": 0,
                "error": str(e),
                "search_time": 0
            }
    
    async def _enhanced_database_search(
        self,
        query: str,
        document_type: str = None,
        jurisdiction: str = None,
        case_type: str = None,
        user_id: str = None,
        limit: int = 10,
        session: AsyncSession = None
    ) -> Dict[str, Any]:
        """Enhanced database search with semantic scoring"""
        start_time = datetime.utcnow()
        
        try:
            if not session:
                return {
                    "query": query,
                    "results": [],
                    "total": 0,
                    "error": "No database session available"
                }
            
            # Build database query
            query_obj = select(Document).where(
                or_(
                    Document.title.ilike(f"%{query}%"),
                    Document.content.ilike(f"%{query}%")
                )
            )
            
            # Add filters
            if document_type:
                query_obj = query_obj.where(Document.document_type == document_type)
            if user_id:
                query_obj = query_obj.where(Document.user_id == user_id)
            
            query_obj = query_obj.limit(limit * 2)  # Get more for semantic filtering
            
            result = await session.execute(query_obj)
            documents = result.scalars().all()
            
            # Process results with semantic scoring
            results = []
            query_embedding = None
            
            if self.sentence_transformer:
                query_embedding = self.sentence_transformer.encode(query)
            
            for doc in documents:
                # Calculate relevance score
                score = self._calculate_relevance_score(
                    query, doc, query_embedding
                )
                
                if score >= min_score:
                    # Extract entities for highlighting
                    entities = ai_service.extract_entities(doc.content[:1000] if doc.content else "")
                    
                    results.append({
                        "document_id": str(doc.id),
                        "title": doc.title,
                        "summary": self._generate_snippet(doc.content, query, 200),
                        "document_type": doc.document_type,
                        "score": score,
                        "created_at": doc.created_at.isoformat() if doc.created_at else None,
                        "user_id": str(doc.user_id) if doc.user_id else None,
                        "highlights": self._generate_highlights(doc, query),
                        "legal_entities": [
                            {"text": entity[0], "label": entity[1]}
                            for entity in entities[:5]  # Limit entities
                        ]
                    })
            
            # Sort by score
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:limit]
            
            search_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "query": query,
                "results": results,
                "total": len(results),
                "search_time": search_time,
                "search_method": "enhanced_database",
                "filters_applied": {
                    "document_type": document_type,
                    "jurisdiction": jurisdiction,
                    "case_type": case_type,
                    "user_id": user_id
                }
            }
            
        except Exception as e:
            logger.error(f"Enhanced database search failed: {str(e)}")
            raise
    
    def _calculate_relevance_score(
        self, 
        query: str, 
        document: Document, 
        query_embedding: np.ndarray = None
    ) -> float:
        """Calculate relevance score for a document"""
        try:
            score = 0.0
            query_lower = query.lower()
            
            # Text matching scores
            if document.title:
                title_lower = document.title.lower()
                if query_lower in title_lower:
                    score += 3.0
                # Partial matches
                query_words = query_lower.split()
                title_words = title_lower.split()
                word_matches = sum(1 for word in query_words if word in title_words)
                score += (word_matches / len(query_words)) * 2.0
            
            if document.content:
                content_lower = document.content.lower()
                if query_lower in content_lower:
                    score += 1.0
                # Count occurrences
                occurrences = content_lower.count(query_lower)
                score += min(occurrences * 0.1, 1.0)
            
            # Semantic similarity (if available)
            if query_embedding is not None and self.sentence_transformer and document.content:
                try:
                    doc_text = f"{document.title or ''} {document.content[:500]}"
                    doc_embedding = self.sentence_transformer.encode(doc_text)
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                    score += similarity * 2.0
                    
                except Exception as e:
                    logger.debug(f"Semantic similarity calculation failed: {str(e)}")
            
            # Normalize score
            return min(score, 10.0) / 10.0
            
        except Exception as e:
            logger.error(f"Relevance score calculation failed: {str(e)}")
            return 0.0
    
    def _generate_snippet(self, content: str, query: str, max_length: int = 200) -> str:
        """Generate a relevant snippet from document content"""
        try:
            if not content:
                return ""
            
            query_lower = query.lower()
            content_lower = content.lower()
            
            # Find the first occurrence of the query
            query_pos = content_lower.find(query_lower)
            
            if query_pos != -1:
                # Extract snippet around the query
                start_pos = max(0, query_pos - max_length // 2)
                end_pos = min(len(content), start_pos + max_length)
                
                snippet = content[start_pos:end_pos]
                
                # Clean up snippet boundaries
                if start_pos > 0:
                    # Find word boundary
                    space_pos = snippet.find(' ')
                    if space_pos != -1:
                        snippet = snippet[space_pos + 1:]
                
                if end_pos < len(content):
                    # Find word boundary
                    last_space = snippet.rfind(' ')
                    if last_space != -1:
                        snippet = snippet[:last_space]
                
                return snippet.strip()
            else:
                # Return beginning of content
                snippet = content[:max_length]
                last_space = snippet.rfind(' ')
                if last_space != -1:
                    snippet = snippet[:last_space]
                return snippet.strip()
                
        except Exception as e:
            logger.error(f"Snippet generation failed: {str(e)}")
            return content[:max_length] if content else ""
    
    def _generate_highlights(self, document: Document, query: str) -> Dict[str, List[str]]:
        """Generate search highlights"""
        try:
            highlights = {}
            query_lower = query.lower()
            
            # Title highlights
            if document.title and query_lower in document.title.lower():
                highlights["title"] = [document.title]
            
            # Content highlights
            if document.content:
                content_snippets = []
                content_lower = document.content.lower()
                
                # Find all occurrences
                start = 0
                while True:
                    pos = content_lower.find(query_lower, start)
                    if pos == -1:
                        break
                    
                    # Extract snippet around match
                    snippet_start = max(0, pos - 50)
                    snippet_end = min(len(document.content), pos + len(query) + 50)
                    snippet = document.content[snippet_start:snippet_end]
                    
                    content_snippets.append(snippet)
                    start = pos + 1
                    
                    # Limit number of snippets
                    if len(content_snippets) >= 3:
                        break
                
                if content_snippets:
                    highlights["content"] = content_snippets
            
            return highlights
            
        except Exception as e:
            logger.error(f"Highlight generation failed: {str(e)}")
            return {}
    
    async def advanced_search(
        self,
        search_params: Dict[str, Any],
        session: AsyncSession = None
    ) -> Dict[str, Any]:
        """Advanced search with multiple parameters and filters"""
        try:
            query = search_params.get("query", "")
            document_types = search_params.get("document_types", [])
            jurisdictions = search_params.get("jurisdictions", [])
            case_types = search_params.get("case_types", [])
            date_range = search_params.get("date_range", {})
            tags = search_params.get("tags", [])
            user_id = search_params.get("user_id")
            limit = search_params.get("limit", 20)
            offset = search_params.get("offset", 0)
            
            if not session:
                return {
                    "query": query,
                    "results": [],
                    "total": 0,
                    "error": "No database session available"
                }
            
            # Build complex database query
            query_obj = select(Document)
            conditions = []
            
            # Text search conditions
            if query:
                conditions.append(
                    or_(
                        Document.title.ilike(f"%{query}%"),
                        Document.content.ilike(f"%{query}%")
                    )
                )
            
            # Filter conditions
            if document_types:
                conditions.append(Document.document_type.in_(document_types))
            
            if user_id:
                conditions.append(Document.user_id == user_id)
            
            # Date range filter
            if date_range:
                if date_range.get("start"):
                    conditions.append(Document.created_at >= date_range["start"])
                if date_range.get("end"):
                    conditions.append(Document.created_at <= date_range["end"])
            
            # Apply conditions
            if conditions:
                query_obj = query_obj.where(and_(*conditions))
            
            # Execute query with pagination
            total_query = query_obj
            result = await session.execute(total_query)
            all_documents = result.scalars().all()
            total = len(all_documents)
            
            # Apply pagination
            paginated_docs = all_documents[offset:offset + limit]
            
            # Process results
            results = []
            query_embedding = None
            
            if self.sentence_transformer and query:
                query_embedding = self.sentence_transformer.encode(query)
            
            for doc in paginated_docs:
                score = self._calculate_relevance_score(query, doc, query_embedding) if query else 1.0
                
                results.append({
                    "document_id": str(doc.id),
                    "title": doc.title,
                    "summary": self._generate_snippet(doc.content, query, 200),
                    "document_type": doc.document_type,
                    "score": score,
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                    "highlights": self._generate_highlights(doc, query) if query else {}
                })
            
            # Sort by score if there's a query
            if query:
                results.sort(key=lambda x: x["score"], reverse=True)
            
            return {
                "query": query,
                "results": results,
                "total": total,
                "offset": offset,
                "limit": limit,
                "search_params": search_params
            }
            
        except Exception as e:
            logger.error(f"Advanced search failed: {str(e)}")
            return {
                "query": search_params.get("query", ""),
                "results": [],
                "total": 0,
                "error": str(e)
            }
    
    async def suggest_search_terms(
        self,
        partial_query: str,
        session: AsyncSession = None,
        limit: int = 5
    ) -> List[str]:
        """Suggest search terms based on partial input"""
        try:
            if not session or len(partial_query) < 2:
                return []
            
            # Get suggestions from document titles
            query = select(Document.title).where(
                Document.title.ilike(f"%{partial_query}%")
            ).limit(limit * 2)
            
            result = await session.execute(query)
            titles = result.scalars().all()
            
            # Extract relevant terms
            suggestions = set()
            partial_lower = partial_query.lower()
            
            for title in titles:
                if title:
                    words = title.lower().split()
                    for word in words:
                        if partial_lower in word and len(word) > len(partial_query):
                            suggestions.add(word)
                        
                        if len(suggestions) >= limit:
                            break
            
            return list(suggestions)[:limit]
            
        except Exception as e:
            logger.error(f"Search suggestion failed: {str(e)}")
            return []
    
    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Remove document from search index"""
        try:
            if not self.es:
                return {"status": "warning", "message": "Elasticsearch not available"}
            
            response = self.es.delete(index=self.index_name, id=document_id)
            
            logger.info(f"Document {document_id} removed from search index")
            return {
                "status": "success",
                "document_id": document_id,
                "result": response.get("result")
            }
            
        except NotFoundError:
            return {
                "status": "warning", 
                "message": f"Document {document_id} not found in index"
            }
        except Exception as e:
            logger.error(f"Failed to delete document from index: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_search_analytics(
        self,
        user_id: str = None,
        session: AsyncSession = None
    ) -> Dict[str, Any]:
        """Get search analytics and statistics"""
        try:
            if not session:
                return {"error": "Database session not available"}
            
            # Get document statistics
            query = select(Document)
            if user_id:
                query = query.where(Document.user_id == user_id)
            
            result = await session.execute(query)
            documents = result.scalars().all()
            
            # Calculate statistics
            total_documents = len(documents)
            document_types = {}
            for doc in documents:
                doc_type = doc.document_type or "Unknown"
                document_types[doc_type] = document_types.get(doc_type, 0) + 1
            
            return {
                "total_documents": total_documents,
                "document_types": [
                    {"key": k, "count": v} for k, v in document_types.items()
                ],
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Search analytics failed: {str(e)}")
            return {"error": str(e)}

# Global search service instance
search_service = SemanticSearchService() 