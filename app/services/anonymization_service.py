"""
Data Anonymization Service for AI Training
Protects user-identifiable information in legal documents for AI training
"""

import os
import re
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import spacy
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from config import settings
from app.services.blockchain_service import blockchain_service

logger = logging.getLogger(__name__)

class AnonymizationService:
    def __init__(self):
        self.nlp = None
        self.fake = Faker()
        self.entity_mappings = {}  # Store consistent mappings
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize anonymization engines and models"""
        try:
            logger.info("Initializing anonymization engines...")
            
            # Load SpaCy model for NLP
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("SpaCy model loaded for anonymization")
            except OSError:
                logger.warning("SpaCy model not found for anonymization")
            
            logger.info("Anonymization engines initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize anonymization engines: {str(e)}")
    
    async def anonymize_document(
        self,
        document_id: str,
        text: str,
        anonymization_level: str = "medium",  # low, medium, high
        preserve_structure: bool = True,
        session: AsyncSession = None
    ) -> Dict[str, Any]:
        """Anonymize a legal document for AI training"""
        try:
            logger.info(f"Anonymizing document {document_id}")
            
            # Analyze text for PII
            entities = self._analyze_text(text)
            
            # Apply anonymization based on level
            anonymized_text, anonymization_map = self._anonymize_text(
                text, entities, anonymization_level, preserve_structure
            )
            
            # Additional legal-specific anonymization
            anonymized_text, legal_anonymizations = self._anonymize_legal_entities(
                anonymized_text, preserve_structure
            )
            
            # Combine anonymization maps
            full_anonymization_map = {**anonymization_map, **legal_anonymizations}
            
            # Calculate anonymization score
            anonymization_score = self._calculate_anonymization_score(
                text, anonymized_text, full_anonymization_map
            )
            
            # Record audit trail
            if session:
                await blockchain_service.record_audit(
                    document_id=document_id,
                    user_id="system",
                    action="document_anonymized",
                    additional_data={
                        "anonymization_level": anonymization_level,
                        "anonymization_score": anonymization_score,
                        "entities_anonymized": len(full_anonymization_map),
                        "preserve_structure": preserve_structure
                    },
                    session=session
                )
            
            return {
                "document_id": document_id,
                "anonymized_text": anonymized_text,
                "anonymization_map": full_anonymization_map,
                "anonymization_score": anonymization_score,
                "entities_found": len(entities),
                "entities_anonymized": len(full_anonymization_map),
                "anonymization_level": anonymization_level,
                "preserve_structure": preserve_structure,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Document anonymization failed: {str(e)}")
            return {
                "error": str(e),
                "document_id": document_id,
                "anonymized_text": text,  # Return original if anonymization fails
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _analyze_text(self, text: str) -> List[Dict[str, Any]]:
        """Analyze text for personally identifiable information"""
        try:
            entities = []
            
            # Use SpaCy if available
            if self.nlp:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'MONEY', 'DATE', 'TIME']:
                        entities.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                            'confidence': 0.8
                        })
            
            # Additional pattern-based entity detection
            pattern_entities = self._detect_pattern_entities(text)
            entities.extend(pattern_entities)
            
            # Legal-specific entity detection
            legal_entities = self._detect_legal_entities(text)
            entities.extend(legal_entities)
            
            return entities
            
        except Exception as e:
            logger.error(f"Text analysis failed: {str(e)}")
            return []
    
    def _detect_pattern_entities(self, text: str) -> List[Dict[str, Any]]:
        """Detect entities using regex patterns"""
        entities = []
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'EMAIL',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.9
            })
        
        # Phone numbers
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}\b',
            r'\+\d{1,3}\s*\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        ]
        
        for pattern in phone_patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    'text': match.group(),
                    'label': 'PHONE',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.85
                })
        
        # SSN
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        for match in re.finditer(ssn_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'SSN',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.95
            })
        
        # Credit card numbers
        cc_pattern = r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        for match in re.finditer(cc_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'CREDIT_CARD',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.9
            })
        
        return entities
    
    def _detect_legal_entities(self, text: str) -> List[Dict[str, Any]]:
        """Detect legal-specific entities"""
        entities = []
        
        # Case numbers
        case_patterns = [
            r'\b\d{4}\s?[A-Z]{2,4}\s?\d+\b',  # 2020 HCA 123
            r'\b[A-Z]{2,4}\s?\d{4}\s?\d+\b',  # HCA 2020 123
            r'\bCase\s+No\.?\s*\d+[-/]\d+\b'   # Case No. 123/2020
        ]
        
        for pattern in case_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    'text': match.group(),
                    'label': 'CASE_NUMBER',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9
                })
        
        # Bar numbers
        bar_pattern = r'\bBar\s+(?:No\.?\s*)?\d+\b'
        for match in re.finditer(bar_pattern, text, re.IGNORECASE):
            entities.append({
                'text': match.group(),
                'label': 'BAR_NUMBER',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.9
            })
        
        # Legal firm names
        firm_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:&|and)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Law\s+Firm|Lawyers?|Attorneys?)\b'
        for match in re.finditer(firm_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'LAW_FIRM',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.8
            })
        
        return entities
    
    def _anonymize_text(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        anonymization_level: str,
        preserve_structure: bool
    ) -> Tuple[str, Dict[str, str]]:
        """Anonymize text based on detected entities"""
        try:
            anonymization_map = {}
            anonymized_text = text
            
            # Sort entities by start position (reverse order to maintain positions)
            entities.sort(key=lambda x: x['start'], reverse=True)
            
            for entity in entities:
                original_text = entity['text']
                
                # Generate replacement based on entity type and level
                replacement = self._generate_replacement(
                    entity, anonymization_level, preserve_structure
                )
                
                # Replace in text
                start, end = entity['start'], entity['end']
                anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]
                anonymization_map[original_text] = replacement
            
            return anonymized_text, anonymization_map
            
        except Exception as e:
            logger.error(f"Text anonymization failed: {str(e)}")
            return text, {}
    
    def _generate_replacement(
        self,
        entity: Dict[str, Any],
        anonymization_level: str,
        preserve_structure: bool
    ) -> str:
        """Generate replacement text for an entity"""
        entity_type = entity['label']
        original_text = entity['text']
        
        if anonymization_level == "low":
            # Light masking
            if entity_type in ['EMAIL', 'PHONE', 'SSN', 'CREDIT_CARD']:
                return self._mask_text(original_text, mask_ratio=0.5)
            else:
                return f"[MASKED_{entity_type}]"
        
        elif anonymization_level == "medium":
            # Replacement with synthetic data
            replacements = {
                'PERSON': 'John Doe',
                'EMAIL': 'user@example.com',
                'PHONE': '+1-555-0123',
                'ORG': 'ABC Company',
                'GPE': 'City, State',
                'MONEY': '$10,000',
                'DATE': '2020-01-01',
                'SSN': 'XXX-XX-1234',
                'CREDIT_CARD': '4XXX-XXXX-XXXX-1234',
                'CASE_NUMBER': 'CASE-2020-001',
                'BAR_NUMBER': 'Bar No. 12345',
                'LAW_FIRM': 'Legal Associates Law Firm'
            }
            
            return replacements.get(entity_type, f"[{entity_type}]")
        
        else:  # high
            # Complete redaction
            return f"[REDACTED_{entity_type}]"
    
    def _mask_text(self, text: str, mask_ratio: float = 0.5, mask_char: str = "*") -> str:
        """Mask part of the text"""
        if len(text) <= 2:
            return mask_char * len(text)
        
        chars_to_mask = int(len(text) * mask_ratio)
        start_visible = (len(text) - chars_to_mask) // 2
        
        masked = (
            text[:start_visible] + 
            mask_char * chars_to_mask + 
            text[start_visible + chars_to_mask:]
        )
        
        return masked
    
    def _anonymize_legal_entities(
        self,
        text: str,
        preserve_structure: bool = True
    ) -> Tuple[str, Dict[str, str]]:
        """Additional anonymization for legal-specific content"""
        anonymization_map = {}
        
        # Monetary amounts
        amount_pattern = r'\$[\d,]+(?:\.\d{2})?'
        for match in re.finditer(amount_pattern, text):
            original = match.group()
            anonymized = self._generate_similar_amount(original) if preserve_structure else "[AMOUNT]"
            text = text.replace(original, anonymized, 1)
            anonymization_map[original] = anonymized
        
        # Addresses
        address_patterns = [
            r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)\b',
            r'\b[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b'
        ]
        
        for pattern in address_patterns:
            for match in re.finditer(pattern, text):
                original = match.group()
                anonymized = "123 Main Street" if preserve_structure else "[ADDRESS]"
                text = text.replace(original, anonymized, 1)
                anonymization_map[original] = anonymized
        
        # Signature blocks
        signature_pattern = r'/s/\s*[A-Za-z\s]+(?:\n|$)'
        for match in re.finditer(signature_pattern, text):
            original = match.group()
            anonymized = "/s/ [SIGNATURE]"
            text = text.replace(original, anonymized, 1)
            anonymization_map[original] = anonymized
        
        return text, anonymization_map
    
    def _generate_similar_amount(self, original_amount: str) -> str:
        """Generate a similar monetary amount for anonymization"""
        try:
            # Extract numeric value
            numeric_str = original_amount.replace('$', '').replace(',', '')
            value = float(numeric_str)
            
            # Generate similar magnitude amount
            if value < 1000:
                anonymized_value = round((value * 0.8) + (value * 0.4 * self.fake.random.random()), 2)
            elif value < 100000:
                anonymized_value = round((value * 0.9) + (value * 0.2 * self.fake.random.random()), 2)
            else:
                anonymized_value = round((value * 0.95) + (value * 0.1 * self.fake.random.random()), 2)
            
            return f"${anonymized_value:,.2f}"
            
        except:
            return "$[AMOUNT]"
    
    def _calculate_anonymization_score(
        self,
        original_text: str,
        anonymized_text: str,
        anonymization_map: Dict[str, str]
    ) -> float:
        """Calculate anonymization effectiveness score"""
        try:
            if not original_text:
                return 0.0
            
            # Calculate percentage of text that was anonymized
            total_chars_anonymized = sum(
                len(original) for original in anonymization_map.keys()
            )
            
            anonymization_ratio = total_chars_anonymized / len(original_text)
            
            # Factor in number of different entity types
            entity_type_bonus = min(0.2, len(anonymization_map) * 0.02)
            
            # Calculate final score (0-1 scale)
            score = min(1.0, anonymization_ratio + entity_type_bonus)
            
            return round(score, 3)
            
        except Exception as e:
            logger.error(f"Anonymization score calculation failed: {str(e)}")
            return 0.0
    
    async def verify_anonymization(
        self,
        anonymized_text: str,
        minimum_score: float = 0.7
    ) -> Dict[str, Any]:
        """Verify that anonymization meets privacy standards"""
        try:
            # Re-analyze anonymized text for remaining PII
            remaining_entities = self._analyze_text(anonymized_text)
            
            # Check for common PII patterns that might have been missed
            privacy_violations = self._check_privacy_violations(anonymized_text)
            
            # Calculate privacy score
            privacy_score = 1.0 - (len(remaining_entities) * 0.1) - (len(privacy_violations) * 0.2)
            privacy_score = max(0.0, privacy_score)
            
            is_safe = privacy_score >= minimum_score and len(privacy_violations) == 0
            
            return {
                "is_safe_for_training": is_safe,
                "privacy_score": round(privacy_score, 3),
                "remaining_pii_count": len(remaining_entities),
                "privacy_violations": privacy_violations,
                "remaining_entities": remaining_entities,
                "verified_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Anonymization verification failed: {str(e)}")
            return {
                "is_safe_for_training": False,
                "privacy_score": 0.0,
                "error": str(e)
            }
    
    def _check_privacy_violations(self, text: str) -> List[str]:
        """Check for privacy violations in anonymized text"""
        violations = []
        
        # Check for email patterns
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            violations.append("Email address detected")
        
        # Check for phone patterns
        if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):
            violations.append("Phone number pattern detected")
        
        # Check for SSN patterns
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
            violations.append("SSN pattern detected")
        
        # Check for credit card patterns
        if re.search(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', text):
            violations.append("Credit card pattern detected")
        
        return violations

# Global anonymization service instance
anonymization_service = AnonymizationService() 