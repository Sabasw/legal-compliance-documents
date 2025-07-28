"""
Enhanced AI Service for Legal Predictive Analytics
Integrates BERT, legal outcome prediction, and advanced NLP capabilities
"""

import os
import logging
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertForSequenceClassification
)
import torch
import spacy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from config import settings
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.models import Document, CaseOutcome, LegalPrecedent

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

class LegalAIService:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.vectorizers = {}
        self.nlp = None
        self.sentence_transformer = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models and components"""
        try:
            logger.info("Initializing AI models...")
            
            # Load SpaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("SpaCy model loaded successfully")
            except OSError:
                logger.warning("SpaCy model not found. Some features may be limited.")
            
            # Load sentence transformer for embeddings
            try:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformer loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {str(e)}")
            
            # Initialize BERT for legal text classification
            self._initialize_bert_models()
            
            # Initialize legal outcome prediction models
            self._initialize_prediction_models()
            
            # Initialize text vectorizers
            self._initialize_vectorizers()
            
            logger.info("AI models initialization completed")
            
        except Exception as e:
            logger.error(f"AI models initialization failed: {str(e)}")
    
    def _initialize_bert_models(self):
        """Initialize BERT models for legal text analysis"""
        try:
            # Legal BERT model for contract analysis
            model_name = "nlpaueb/legal-bert-base-uncased"
            
            try:
                self.tokenizers['legal_bert'] = AutoTokenizer.from_pretrained(model_name)
                self.models['legal_bert'] = AutoModel.from_pretrained(model_name)
                logger.info("Legal BERT model loaded successfully")
            except Exception:
                # Fallback to standard BERT
                self.tokenizers['bert'] = BertTokenizer.from_pretrained('bert-base-uncased')
                self.models['bert'] = BertForSequenceClassification.from_pretrained(
                    'bert-base-uncased',
                    num_labels=5  # 5 risk levels
                )
                logger.info("Standard BERT model loaded as fallback")
            
            # Initialize sentiment analysis pipeline
            self.models['sentiment'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
        except Exception as e:
            logger.error(f"BERT models initialization failed: {str(e)}")
    
    def _initialize_prediction_models(self):
        """Initialize predictive models for legal outcomes"""
        try:
            # Load pre-trained models if they exist
            model_paths = {
                'case_outcome': 'models/case_outcome_predictor.pkl',
                'risk_assessment': 'models/risk_assessment_model.pkl',
                'compliance_checker': 'models/compliance_checker.pkl'
            }
            
            for model_name, path in model_paths.items():
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    logger.info(f"Loaded {model_name} model from {path}")
                else:
                    # Initialize new models
                    if model_name == 'case_outcome':
                        self.models[model_name] = GradientBoostingClassifier(
                            n_estimators=100,
                            learning_rate=0.1,
                            max_depth=6,
                            random_state=42
                        )
                    elif model_name == 'risk_assessment':
                        self.models[model_name] = RandomForestClassifier(
                            n_estimators=100,
                            max_depth=10,
                            random_state=42
                        )
                    elif model_name == 'compliance_checker':
                        self.models[model_name] = GradientBoostingClassifier(
                            n_estimators=50,
                            learning_rate=0.1,
                            max_depth=4,
                            random_state=42
                        )
                    
                    logger.info(f"Initialized new {model_name} model")
            
        except Exception as e:
            logger.error(f"Prediction models initialization failed: {str(e)}")
    
    def _initialize_vectorizers(self):
        """Initialize text vectorizers"""
        try:
            vectorizer_paths = {
                'tfidf_legal': 'models/tfidf_legal_vectorizer.pkl',
                'tfidf_risk': 'models/tfidf_risk_vectorizer.pkl'
            }
            
            for vectorizer_name, path in vectorizer_paths.items():
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        self.vectorizers[vectorizer_name] = pickle.load(f)
                    logger.info(f"Loaded {vectorizer_name} vectorizer")
                else:
                    self.vectorizers[vectorizer_name] = TfidfVectorizer(
                        max_features=5000,
                        stop_words='english',
                        ngram_range=(1, 3),
                        min_df=2,
                        max_df=0.95
                    )
                    logger.info(f"Initialized new {vectorizer_name} vectorizer")
            
        except Exception as e:
            logger.error(f"Vectorizers initialization failed: {str(e)}")
    
    async def predict_case_outcome(
        self, 
        case_text: str, 
        case_type: str = None,
        jurisdiction: str = "AU",
        session: AsyncSession = None
    ) -> Dict[str, Any]:
        """Predict legal case outcome using AI models"""
        try:
            logger.info("Predicting case outcome")
            
            # Extract features from case text
            features = await self._extract_case_features(case_text, case_type, jurisdiction)
            
            # Get BERT embeddings
            bert_features = self._get_bert_embeddings(case_text)
            
            # Combine features
            combined_features = np.concatenate([features, bert_features])
            
            # Predict outcome if model is trained
            if 'case_outcome' in self.models and hasattr(self.models['case_outcome'], 'predict_proba'):
                try:
                    # Reshape for single prediction
                    features_reshaped = combined_features.reshape(1, -1)
                    
                    # Get prediction probabilities
                    probabilities = self.models['case_outcome'].predict_proba(features_reshaped)[0]
                    predicted_class = self.models['case_outcome'].predict(features_reshaped)[0]
                    
                    # Map classes to outcomes
                    outcome_mapping = {
                        0: "Favorable",
                        1: "Partially Favorable", 
                        2: "Neutral",
                        3: "Unfavorable",
                        4: "Highly Unfavorable"
                    }
                    
                    prediction = {
                        "predicted_outcome": outcome_mapping.get(predicted_class, "Unknown"),
                        "confidence": float(max(probabilities)),
                        "probabilities": {
                            outcome_mapping[i]: float(prob) 
                            for i, prob in enumerate(probabilities)
                        }
                    }
                    
                except Exception as e:
                    # Fallback to rule-based prediction
                    prediction = self._rule_based_outcome_prediction(case_text, case_type)
                    logger.warning(f"Used rule-based prediction due to model error: {str(e)}")
            else:
                # Use rule-based prediction
                prediction = self._rule_based_outcome_prediction(case_text, case_type)
            
            # Add legal precedent analysis
            precedents = await self._find_similar_precedents(case_text, session)
            
            # Risk assessment
            risk_assessment = await self._assess_legal_risk(case_text, case_type)
            
            return {
                "prediction": prediction,
                "risk_assessment": risk_assessment,
                "similar_precedents": precedents,
                "features_analyzed": {
                    "text_length": len(case_text),
                    "case_type": case_type,
                    "jurisdiction": jurisdiction,
                    "sentiment": self._analyze_sentiment(case_text),
                    "key_entities": self._extract_entities(case_text)
                },
                "confidence_factors": {
                    "model_confidence": prediction.get("confidence", 0.0),
                    "precedent_similarity": len(precedents) > 0,
                    "feature_quality": "high" if len(case_text) > 1000 else "medium"
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Case outcome prediction failed: {str(e)}")
            return {
                "error": str(e),
                "prediction": {"predicted_outcome": "Unknown", "confidence": 0.0},
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _extract_case_features(
        self, 
        case_text: str, 
        case_type: str = None, 
        jurisdiction: str = "AU"
    ) -> np.ndarray:
        """Extract numerical features from case text"""
        features = []
        
        # Text statistics
        features.append(len(case_text))
        features.append(len(case_text.split()))
        features.append(len(sent_tokenize(case_text)))
        
        # Legal terminology frequency
        legal_terms = [
            'contract', 'breach', 'damages', 'liability', 'negligence',
            'jurisdiction', 'precedent', 'statute', 'regulation', 'compliance'
        ]
        
        text_lower = case_text.lower()
        for term in legal_terms:
            features.append(text_lower.count(term))
        
        # Case type encoding
        case_types = ['contract', 'tort', 'criminal', 'family', 'corporate']
        for ct in case_types:
            features.append(1 if case_type and ct in case_type.lower() else 0)
        
        # Jurisdiction encoding
        jurisdictions = ['AU', 'US', 'UK', 'CA']
        for j in jurisdictions:
            features.append(1 if jurisdiction == j else 0)
        
        return np.array(features, dtype=float)
    
    def _get_bert_embeddings(self, text: str) -> np.ndarray:
        """Get BERT embeddings for text"""
        try:
            if self.sentence_transformer:
                # Using sentence transformer for embeddings
                embeddings = self.sentence_transformer.encode([text])
                return embeddings[0]
            elif 'legal_bert' in self.models:
                # Using Legal BERT
                tokenizer = self.tokenizers['legal_bert']
                model = self.models['legal_bert']
                
                inputs = tokenizer(
                    text, 
                    return_tensors='pt', 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                )
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                
                return embeddings
            else:
                # Fallback to TF-IDF if no BERT available
                if 'tfidf_legal' in self.vectorizers:
                    try:
                        tfidf_matrix = self.vectorizers['tfidf_legal'].transform([text])
                        return tfidf_matrix.toarray()[0]
                    except:
                        pass
                
                # Ultimate fallback: simple numerical features
                return np.array([
                    len(text), len(text.split()), 
                    text.count('.'), text.count('!')
                ])
                
        except Exception as e:
            logger.error(f"BERT embeddings failed: {str(e)}")
            return np.zeros(384)  # Default embedding size
    
    def _rule_based_outcome_prediction(self, case_text: str, case_type: str = None) -> Dict[str, Any]:
        """Rule-based outcome prediction as fallback"""
        text_lower = case_text.lower()
        
        # Simple rule-based scoring
        positive_indicators = ['agree', 'comply', 'resolution', 'settlement', 'cooperation']
        negative_indicators = ['breach', 'violation', 'dispute', 'conflict', 'damages']
        
        positive_score = sum(1 for indicator in positive_indicators if indicator in text_lower)
        negative_score = sum(1 for indicator in negative_indicators if indicator in text_lower)
        
        if positive_score > negative_score:
            outcome = "Favorable"
            confidence = min(0.8, 0.5 + (positive_score - negative_score) * 0.1)
        elif negative_score > positive_score:
            outcome = "Unfavorable"
            confidence = min(0.8, 0.5 + (negative_score - positive_score) * 0.1)
        else:
            outcome = "Neutral"
            confidence = 0.5
        
        return {
            "predicted_outcome": outcome,
            "confidence": confidence,
            "method": "rule_based",
            "positive_indicators": positive_score,
            "negative_indicators": negative_score
        }
    
    async def _find_similar_precedents(
        self, 
        case_text: str, 
        session: AsyncSession = None
    ) -> List[Dict[str, Any]]:
        """Find similar legal precedents"""
        try:
            if not session:
                return []
            
            # Get case embeddings
            case_embedding = self._get_bert_embeddings(case_text)
            
            # Query similar precedents from database
            # This would require a proper precedents table with embeddings
            # For now, return mock data
            return [
                {
                    "case_name": "Smith v. Jones (2020)",
                    "similarity_score": 0.85,
                    "outcome": "Favorable",
                    "key_points": ["Contract interpretation", "Good faith obligation"]
                },
                {
                    "case_name": "ABC Corp v. XYZ Ltd (2019)",
                    "similarity_score": 0.78,
                    "outcome": "Partially Favorable",
                    "key_points": ["Breach of contract", "Damages calculation"]
                }
            ]
            
        except Exception as e:
            logger.error(f"Precedent search failed: {str(e)}")
            return []
    
    async def _assess_legal_risk(self, case_text: str, case_type: str = None) -> Dict[str, Any]:
        """Assess legal risk factors"""
        try:
            # Extract risk features
            risk_features = self._extract_risk_features(case_text, case_type)
            
            # Predict risk level
            if ('risk_assessment' in self.models and 
                hasattr(self.models['risk_assessment'], 'predict_proba')):
                try:
                    risk_proba = self.models['risk_assessment'].predict_proba([risk_features])[0]
                    risk_level = self.models['risk_assessment'].predict([risk_features])[0]
                    
                    risk_mapping = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
                    
                    return {
                        "risk_level": risk_mapping.get(risk_level, "Unknown"),
                        "risk_score": float(max(risk_proba)),
                        "risk_factors": self._identify_risk_factors(case_text),
                        "mitigation_suggestions": self._suggest_risk_mitigation(case_text, case_type)
                    }
                except Exception:
                    pass
            
            # Fallback rule-based risk assessment
            return self._rule_based_risk_assessment(case_text, case_type)
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {str(e)}")
            return {"risk_level": "Unknown", "risk_score": 0.0}
    
    def _extract_risk_features(self, case_text: str, case_type: str = None) -> List[float]:
        """Extract features for risk assessment"""
        features = []
        text_lower = case_text.lower()
        
        # High-risk keywords
        high_risk_terms = [
            'breach', 'violation', 'penalty', 'fine', 'damages', 
            'lawsuit', 'litigation', 'criminal', 'fraud'
        ]
        
        for term in high_risk_terms:
            features.append(text_lower.count(term))
        
        # Text complexity
        features.append(len(case_text))
        features.append(len(case_text.split()))
        
        # Sentiment score
        sentiment = self._analyze_sentiment(case_text)
        features.append(sentiment.get('compound', 0.0))
        
        return features
    
    def _rule_based_risk_assessment(self, case_text: str, case_type: str = None) -> Dict[str, Any]:
        """Rule-based risk assessment"""
        text_lower = case_text.lower()
        
        high_risk_terms = ['criminal', 'fraud', 'felony', 'prison']
        medium_risk_terms = ['breach', 'violation', 'lawsuit', 'penalty']
        
        high_risk_score = sum(1 for term in high_risk_terms if term in text_lower)
        medium_risk_score = sum(1 for term in medium_risk_terms if term in text_lower)
        
        if high_risk_score > 0:
            risk_level = "High"
            risk_score = min(0.9, 0.7 + high_risk_score * 0.1)
        elif medium_risk_score > 0:
            risk_level = "Medium"
            risk_score = min(0.7, 0.4 + medium_risk_score * 0.1)
        else:
            risk_level = "Low"
            risk_score = 0.2
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "method": "rule_based"
        }
    
    def _identify_risk_factors(self, case_text: str) -> List[str]:
        """Identify specific risk factors in the text"""
        risk_factors = []
        text_lower = case_text.lower()
        
        risk_patterns = {
            "Contractual breach": ["breach", "violation", "non-compliance"],
            "Financial exposure": ["damages", "penalty", "fine", "compensation"],
            "Regulatory issues": ["regulation", "compliance", "authority"],
            "Litigation risk": ["lawsuit", "litigation", "court", "legal action"]
        }
        
        for factor, keywords in risk_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                risk_factors.append(factor)
        
        return risk_factors
    
    def _suggest_risk_mitigation(self, case_text: str, case_type: str = None) -> List[str]:
        """Suggest risk mitigation strategies"""
        suggestions = []
        text_lower = case_text.lower()
        
        if 'contract' in text_lower:
            suggestions.append("Review contract terms and conditions")
            suggestions.append("Seek legal counsel for contract interpretation")
        
        if 'compliance' in text_lower:
            suggestions.append("Conduct compliance audit")
            suggestions.append("Implement compliance monitoring system")
        
        if 'breach' in text_lower:
            suggestions.append("Document all communications")
            suggestions.append("Attempt negotiated resolution")
        
        return suggestions or ["Seek professional legal advice"]
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of legal text"""
        try:
            if 'sentiment' in self.models:
                result = self.models['sentiment'](text)[0]
                return {
                    "label": result['label'],
                    "score": result['score'],
                    "compound": result['score'] if result['label'] == 'POSITIVE' else -result['score']
                }
            else:
                # Fallback sentiment analysis
                positive_words = ['good', 'positive', 'favorable', 'agree', 'comply']
                negative_words = ['bad', 'negative', 'unfavorable', 'breach', 'violate']
                
                text_lower = text.lower()
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)
                
                if pos_count > neg_count:
                    return {"label": "POSITIVE", "score": 0.6, "compound": 0.3}
                elif neg_count > pos_count:
                    return {"label": "NEGATIVE", "score": 0.6, "compound": -0.3}
                else:
                    return {"label": "NEUTRAL", "score": 0.5, "compound": 0.0}
                    
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return {"label": "NEUTRAL", "score": 0.5, "compound": 0.0}
    
    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from legal text"""
        try:
            if self.nlp:
                doc = self.nlp(text)
                return [
                    {"text": ent.text, "label": ent.label_, "description": spacy.explain(ent.label_)}
                    for ent in doc.ents
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LAW', 'MONEY', 'DATE']
                ]
            else:
                # Simple pattern-based entity extraction
                import re
                entities = []
                
                # Money patterns
                money_pattern = r'\$[\d,]+(?:\.\d{2})?'
                for match in re.finditer(money_pattern, text):
                    entities.append({
                        "text": match.group(),
                        "label": "MONEY",
                        "description": "Monetary value"
                    })
                
                # Date patterns
                date_pattern = r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b'
                for match in re.finditer(date_pattern, text):
                    entities.append({
                        "text": match.group(),
                        "label": "DATE",
                        "description": "Date"
                    })
                
                return entities
                
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            return []
    
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities - compatible with prototype interface"""
        entities = self._extract_entities(text)
        return [(ent["text"], ent["label"]) for ent in entities]
    
    def summarize_text(self, text: str) -> str:
        """Summarize legal text - compatible with prototype interface"""
        try:
            if self.nlp:
                doc = self.nlp(text)
                sentences = list(doc.sents)
                if sentences:
                    # Simple extractive summarization - return first sentence
                    return sentences[0].text
            
            # Fallback: return first sentence
            sentences = sent_tokenize(text)
            return sentences[0] if sentences else ""
            
        except Exception as e:
            logger.error(f"Text summarization failed: {str(e)}")
            return text[:200] + "..." if len(text) > 200 else text

# Global AI service instance
ai_service = LegalAIService() 