# analyzer.py
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from groq import Groq
from sentence_transformers import SentenceTransformer
from app.core.predictive import PredictiveAnalytics
from app.core.document import DocumentProcessor
from app.core.knowledge2 import KnowledgeBase
from app.core.xai import XAITracker
from app.database.db.db_connection import SessionLocal
from app.database.models.models import Document, DocumentAnalysis, RiskScore
from app.services.blockchain_service import blockchain_service
from app.services.billing_service import billing_service  
from app.services.jurisdiction import jurisdiction_service
import os
import logging
import re
import faiss
import numpy as np
import torch
import hashlib
from app.config import settings
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class ComplianceAnalyzer:
    def __init__(self, groq_client: Groq, model: SentenceTransformer, predictive_analytics: PredictiveAnalytics):
        self.client = groq_client
        self.model = model
        self.predictive_analytics = predictive_analytics
        self.document_processor = DocumentProcessor()
        self.blockchain_logger = blockchain_service
        self.billing_system = billing_service
        self.jurisdiction_handler = jurisdiction_service
            
        # Initialize KnowledgeBase
        self.knowledge_base = KnowledgeBase()
        self.knowledge_base.initialize_model(model)  # Use the shared model
        
        # Load existing knowledge base if available
        if os.path.exists(settings.FAISS_INDEX_PATH):
            self.knowledge_base.load(
                settings.FAISS_INDEX_PATH,
                settings.FAISS_CHUNKS_PATH
            )    
        
        logger.info("Enterprise compliance analyzer initialized")

    def analyze_document(self, doc_path: str, user_id: str = None, ip_address: str = None, doc_id: str = None, doc_type: Optional[str] = None) -> Dict[str, Any]:
        """Complete document analysis pipeline with enterprise features"""
        doc_hash = ""
        try:
            # Initialize tracking and extract content
            xai = XAITracker()
            text = self.document_processor.extract_text(doc_path)
            if not text:
                raise ValueError("Text extraction failed")

            doc_hash = DocumentProcessor.document_hash(text)
            text = text[:settings.MAX_TEXT_LENGTH]
            logger.info(f"Analyzing document {doc_path} (hash: {doc_hash[:8]}...)")

            # Detect jurisdiction
            jurisdiction = self.jurisdiction_handler.detect_jurisdiction(ip_address, text)
            jurisdiction_config = self.jurisdiction_handler.get_jurisdiction_config(jurisdiction)
            xai.log_decision("Jurisdiction", jurisdiction, 0.9)

            # Track usage and billing
            transaction_info = None
            if user_id:
                try:
                    transaction_info = self.billing_system.track_usage(
                        user_id,
                        os.path.splitext(doc_path)[1][1:],  # file extension as doc type
                        jurisdiction
                    )
                    xai.log_decision("Billing", transaction_info, 0.95)
                except Exception as e:
                    logger.error(f"Billing failed: {str(e)}")
                    xai.log_decision("Billing", "Failed", 0.0)

            # Document classification
            doc_type = doc_type or self._classify_document(text)
            xai.log_decision("Document Type", doc_type, 0.85)
            logger.info(f"Classified as: {doc_type}")

            # Get jurisdiction-specific legislation
            legislation = self.jurisdiction_handler.get_relevant_legislation(jurisdiction, doc_type)
            xai.log_decision("Relevant Legislation", legislation, 0.9)

            # Predictive analytics
            risk_profile = self.predictive_analytics.generate_risk_profile(text, doc_type)
            predictions = self.predictive_analytics.predict_legal_outcomes(text, doc_type)

            risk_fig = self.predictive_analytics.plot_risk_profile(risk_profile)
            risk_viz_path = f"risk_profile_{doc_hash[:8]}_{int(datetime.now().timestamp())}.png"
            risk_fig.savefig(risk_viz_path, bbox_inches='tight', dpi=120)
            plt.close(risk_fig)

            xai.attach_visualization("Risk Profile", risk_viz_path)
            xai.log_decision("Risk Profile", risk_profile, 0.85)
            xai.log_decision("Predicted Outcomes", predictions, 0.9)

            # Knowledge base lookup
            kb_results = []
            if self.knowledge_base.index is not None:
                kb_results = self.knowledge_base.query(text)
                xai.set_kb_references(kb_results)
                xai.log_decision("Relevant Rules Found", len(kb_results), min(len(kb_results)/5, 1.0))

            # Generate summary and analysis
            summary = self._generate_summary(text, doc_type, kb_results, predictions, jurisdiction_config)
            xai.log_decision("Summary Quality", len(summary.split()) > 100, 0.8)

            analysis = self._perform_compliance_analysis(text, doc_type, kb_results, summary, predictions, jurisdiction_config)
            xai.log_decision("Compliance Status", analysis['status'], 0.9)
            xai.log_decision("Risk Level", analysis['risk_score'], 0.8)

            # Log to blockchain
            blockchain_proof = None
            try:
                blockchain_proof = self.blockchain_logger.log_compliance_result(
                    doc_hash,
                    analysis['status'],
                    jurisdiction
                )
                xai.log_decision("Blockchain Proof", blockchain_proof, 1.0)
            except Exception as e:
                logger.error(f"Blockchain logging failed: {str(e)}")
                xai.log_decision("Blockchain Proof", "Failed", 0.0)

            # Generate final report
            report = xai.generate_report(
                doc_path=doc_path,
                doc_type=doc_type,
                summary=summary,
                analysis=analysis['full_analysis'],
                factors={
                    "Statutory References": analysis['statutory_refs'],
                    "Critical Issues": analysis['issues'],
                    "Predictive Outcomes": predictions,
                    "Risk Profile": risk_profile,
                    "Jurisdiction": jurisdiction_config,
                    "Blockchain Proof": blockchain_proof,
                    "Transaction Info": transaction_info,
                    "Relevant Legislation": legislation
                },
                visualizations=[risk_viz_path]
            )

            # Store results in database if doc_id is provided
            if doc_id:
                db = SessionLocal()
                try:
                    # Create or update document analysis
                    db_analysis = db.query(DocumentAnalysis).filter(
                        DocumentAnalysis.document_id == doc_id
                    ).first()
                    
                    if not db_analysis:
                        db_analysis = DocumentAnalysis(
                            document_id=doc_id,
                            compliance_status=analysis['status'],
                            risk_score=analysis['risk_score'],
                            risk_profile=risk_profile,
                            statutory_references=analysis['statutory_refs'],
                            key_issues=analysis['issues'],
                            recommendations=self._extract_recommendations(analysis['full_analysis']),
                            predictive_outcomes=predictions,
                            summary=summary,
                            full_analysis=analysis['full_analysis'],
                            jurisdiction=jurisdiction,
                            blockchain_proof=blockchain_proof
                        )
                        db.add(db_analysis)
                    else:
                        # Update existing analysis
                        db_analysis.compliance_status = analysis['status']
                        db_analysis.risk_score = analysis['risk_score']
                        db_analysis.risk_profile = risk_profile
                        db_analysis.statutory_references = analysis['statutory_refs']
                        db_analysis.key_issues = analysis['issues']
                        db_analysis.recommendations = self._extract_recommendations(analysis['full_analysis'])
                        db_analysis.predictive_outcomes = predictions
                        db_analysis.summary = summary
                        db_analysis.full_analysis = analysis['full_analysis']
                        db_analysis.jurisdiction = jurisdiction
                        db_analysis.blockchain_proof = blockchain_proof
                    
                    db.commit()
                finally:
                    db.close()

            logger.info(f"Analysis completed for {doc_path}")

            now = datetime.utcnow()

            return {
                "document_hash": doc_hash,
                "document_id": str(doc_id),
                "doc_id": str(doc_id),  # Required for DocumentAnalysis
                "document_type": doc_type or "unknown",
                "compliance_status": analysis.get("status") or "unknown",
                "risk_score": str(analysis.get("risk_score") or "Unknown"),
                "risk_profile": risk_profile if isinstance(risk_profile, dict) else {},
                "report_path": report.get("text_path"),
                "visualization_paths": report.get("viz_paths"),
                "statutory_references": analysis.get("statutory_refs") if isinstance(analysis.get("statutory_refs"), list) else [],
                "key_issues": analysis.get("issues") if isinstance(analysis.get("issues"), list) else [],
                "predictive_outcomes": predictions if isinstance(predictions, list) else [],
                "recommendations": self._extract_recommendations(analysis.get("full_analysis", "")),
                "summary": summary,
                "full_analysis": analysis.get("full_analysis", ""),
                "kb_references": [r[0] for r in kb_results[:3]] if kb_results else [],
                "jurisdiction": jurisdiction_config,
                "blockchain_proof": blockchain_proof,
                "transaction_info": transaction_info,
                "legislation": legislation,
                "timestamp": now
            }


        except Exception as e:
            logger.error(f"Analysis failed for {doc_path}: {str(e)}")
            return {
                "error": str(e),
                "document_hash": doc_hash,
                "document_id": doc_id,
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }

    def get_analysis(self, doc_id: str) -> Dict[str, Any]:
        """Retrieve stored analysis by ID from database"""
        db = SessionLocal()
        try:
            analysis = db.query(DocumentAnalysis).filter(
                DocumentAnalysis.document_id == doc_id
            ).first()
            
            if not analysis:
                raise ValueError(f"Analysis not found for document ID: {doc_id}")
            
            return {
                "doc_id": doc_id,
                "compliance_status": analysis.compliance_status,
                "risk_score": str(analysis.risk_score),
                "risk_profile": analysis.risk_profile,
                "statutory_references": analysis.statutory_references,
                "key_issues": analysis.key_issues,
                "recommendations": analysis.recommendations,
                "predictive_outcomes": analysis.predictive_outcomes,
                "summary": analysis.summary,
                "full_analysis": analysis.full_analysis,
                "jurisdiction": analysis.jurisdiction,
                "blockchain_proof": analysis.blockchain_proof,
                "timestamp": analysis.created_at.isoformat()
            }
        finally:
            db.close()

    def list_analyses(self) -> List[Dict[str, Any]]:
        """List all stored analyses from database"""
        db = SessionLocal()
        try:
            analyses = db.query(DocumentAnalysis).all()
            return [{
                "doc_id": analysis.document_id,
                "compliance_status": analysis.compliance_status,
                "risk_score": str(analysis.risk_score),
                "risk_profile": analysis.risk_profile,
                "statutory_references": analysis.statutory_references,
                "key_issues": analysis.key_issues,
                "recommendations": analysis.recommendations,
                "predictive_outcomes": analysis.predictive_outcomes,
                "summary": analysis.summary,
                "full_analysis": analysis.full_analysis,
                "jurisdiction": analysis.jurisdiction,
                "blockchain_proof": analysis.blockchain_proof,
                "timestamp": analysis.created_at.isoformat()
            } for analysis in analyses]
        finally:
            db.close()

    @property
    def kb_index(self):
        """Backward compatible access to knowledge base index"""
        return self.knowledge_base.index if hasattr(self.knowledge_base, 'index') else None

    def rebuild_knowledge_base(self, doc_paths: List[str]):
        """Rebuild the knowledge base from documents"""
        if not hasattr(self, 'knowledge_base'):
            raise AttributeError("Knowledge base not initialized in analyzer")
        
        try:
            logger.info(f"Rebuilding knowledge base with {len(doc_paths)} documents")
            
            # Verify documents exist
            valid_paths = [p for p in doc_paths if os.path.exists(p)]
            if not valid_paths:
                raise ValueError("No valid document paths provided")
                
            # Rebuild knowledge base
            self.knowledge_base.build_from_documents(valid_paths)
            
            # Save the rebuilt knowledge base
            self.knowledge_base.save(
                settings.FAISS_INDEX_PATH,
                settings.FAISS_CHUNKS_PATH
            )
            
            logger.info("Knowledge base rebuilt successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rebuild knowledge base: {str(e)}", exc_info=True)
            raise RuntimeError(f"Knowledge base rebuild failed: {str(e)}")

    # For backward compatibility
    @property
    def kb_chunks(self):
        return self.knowledge_base.chunks if hasattr(self, 'knowledge_base') else None

    def _classify_document(self, text: str) -> str:
        """Comprehensive document classification"""
        # First check for court ruling patterns
        if self._is_court_ruling(text):
            return "court_ruling"

        # Try knowledge base classification if available
        if self.kb_index and self.kb_chunks:
            doc_type = self._classify_with_knowledge_base(text)
            if doc_type in ['contract', 'court_ruling', 'regulatory_filing', 'policy']:
                return doc_type

        # Fall back to LLM classification
        return self._classify_with_groq(text[:2000])

    def _is_court_ruling(self, text: str) -> bool:
        """Enhanced court ruling detection"""
        first_page = text[:2000]
        court_patterns = [
            r'\[20\d{2}\] [A-Z]+ \d+',
            r'\b\d{4} [A-Z]+ \d+\b',
            r'\b[A-Z]{2,3} \d+\b'
        ]
        for pattern in court_patterns:
            if re.search(pattern, first_page):
                return True
        return False

    def _classify_with_knowledge_base(self, text: str) -> str:
        """Knowledge-based classification"""
        classification_patterns = {
            "contract": ["contract", "agreement", "clause", "party", "term"],
            "court_ruling": ["judgment", "court", "ruling", "decision", "appeal"],
            "regulatory_filing": ["filing", "disclosure", "report", "submit", "regulation"],
            "policy": ["policy", "procedure", "guideline", "compliance", "standard"]
        }

        doc_embedding = self.model.encode([text[:1000]])
        distances, indices = self.kb_index.search(doc_embedding, 5)
        retrieved_chunks = [self.kb_chunks[i] for i in indices[0] if 0 <= i < len(self.kb_chunks)]

        scores = {doc_type: 0 for doc_type in classification_patterns.keys()}
        for chunk in retrieved_chunks:
            chunk_lower = chunk.lower()
            for doc_type, keywords in classification_patterns.items():
                scores[doc_type] += sum(keyword in chunk_lower for keyword in keywords)

        if any(scores.values()):
            return max(scores.items(), key=lambda x: x[1])[0]
        return "unknown"

    def _classify_with_groq(self, text: str) -> str:
        """LLM-based document classification"""
        prompt = f"""Classify this legal document excerpt:
        {text[:2000]}

        Options: contract, court_ruling, regulatory_filing, policy, unknown
        Respond ONLY with the document type from the options provided."""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=20
            )
            classification = response.choices[0].message.content.strip().lower()
            return classification if classification in ['contract', 'court_ruling', 'regulatory_filing', 'policy'] else "unknown"
        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            return "unknown"

    def _generate_summary(self, text: str, doc_type: str, kb_results: list, predictions: list, jurisdiction_config: dict) -> str:
        """Generate comprehensive legal summary with jurisdiction context"""
        kb_context = "\n".join(
            f"Relevant Rule {i+1}: {rule[0][:300]} (Relevance: {rule[1]:.2f})"
            for i, rule in enumerate(kb_results[:3])
        ) if kb_results else "No highly relevant rules identified"

        doc_type_info = {
            "contract": {
                "focus_areas": [
                    "Missing essential clauses",
                    "Ambiguous terms",
                    f"{jurisdiction_config['name']} Consumer Law compliance",
                    "Contract law requirements",
                    "Blockchain audit trails",
                    "RBAC provisions",
                    "Predictive dispute analysis"
                ],
                "examples": jurisdiction_config['compliance_rules'].get('contract', [])
            },
            "court_ruling": {
                "focus_areas": [
                    "Judicial reasoning",
                    "Statutory interpretation",
                    "Precedent alignment",
                    "Jurisdictional issues",
                    "Evidence handling",
                    "Court procedures",
                    "Predictive outcome analysis"
                ],
                "examples": jurisdiction_config['compliance_rules'].get('court_ruling', [])
            },
            "regulatory_filing": {
                "focus_areas": [
                    "Disclosure completeness",
                    "Reporting accuracy",
                    "Timeliness requirements",
                    f"{jurisdiction_config['regulator']} compliance",
                    "Audit trails",
                    "Data governance",
                    "Regulatory risk assessment"
                ],
                "examples": jurisdiction_config['compliance_rules'].get('regulatory_filing', [])
            },
            "policy": {
                "focus_areas": [
                    "Policy currency",
                    "Compliance risks",
                    "Workplace health and safety",
                    "Privacy compliance",
                    "Access controls",
                    "Policy enforcement",
                    "Impact analysis"
                ],
                "examples": jurisdiction_config['compliance_rules'].get('policy', [])
            },
            "unknown": {
                "focus_areas": ["General legal compliance"],
                "examples": []
            }
        }.get(doc_type, {
            "focus_areas": ["General legal compliance"],
            "examples": []
        })

        focus_areas = "\n".join(f"- {area}" for area in doc_type_info.get("focus_areas", []))
        examples = "\n".join(f"- {ex}" for ex in doc_type_info.get("examples", []))

        prediction_context = "\nPredicted Outcomes:\n" + "\n".join(
            f"- {pred[0]} (confidence: {pred[1]:.0%})"
            for pred in predictions[:3]
        ) if predictions else "\nNo strong predictive outcomes identified"

        prompt = f"""As a {jurisdiction_config['name']} legal compliance expert, generate a comprehensive summary of this {doc_type}:

        Document Excerpt:
        {text[:4000]}

        Key Focus Areas for {doc_type} in {jurisdiction_config['name']}:
        {focus_areas}

        Relevant Compliance Context:
        {kb_context}

        {prediction_context}

        Jurisdiction-Specific Legislation:
        {", ".join(jurisdiction_config['compliance_rules'].values())}

        Create a detailed summary (500-700 words) covering:
        1. Document purpose and key parties
        2. Main legal obligations under {jurisdiction_config['name']} law
        3. Compliance risks and issues specific to jurisdiction
        4. Relevant {jurisdiction_config['name']} legal frameworks
        5. Blockchain and RBAC considerations
        6. Predicted outcomes and risk factors
        7. Recommended review areas"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=800,
                temperature=0.3,
            )
            summary = response.choices[0].message.content
            if len(summary.split()) < 100:
                raise ValueError("Summary too short")
            return summary
        except Exception as e:
            logger.warning(f"Summary generation failed: {str(e)}")
            return self._fallback_summary(text, doc_type, jurisdiction_config)

    def _fallback_summary(self, text: str, doc_type: str, jurisdiction_config: dict) -> str:
        """Generate fallback summary when LLM fails"""
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        key_sentences = sentences[:10] + sentences[-5:] if len(sentences) > 15 else sentences
        return (
            f"Basic Summary of {doc_type} for {jurisdiction_config['name']}:\n\n" +
            "\n".join(f"- {s}" for s in key_sentences) +
            "\n\n[Full analysis unavailable due to summary generation error]"
        )

    def _perform_compliance_analysis(self, text: str, doc_type: str,
                                   kb_results: list, summary: str,
                                   predictions: list, jurisdiction_config: dict) -> Dict[str, Any]:
        """Comprehensive compliance analysis with jurisdiction awareness"""
        kb_context = "\n".join(
            f"{i+1}. {rule[0][:300]} (Relevance: {rule[1]:.2f})"
            for i, rule in enumerate(kb_results[:3])
        ) if kb_results else "No highly relevant compliance rules found"

        doc_type_config = {
            "contract": {
                "focus_areas": [
                    "Missing essential clauses",
                    "Ambiguous terms",
                    f"{jurisdiction_config['name']} Consumer Law compliance",
                    "Contract law requirements",
                    "Blockchain audit trails",
                    "RBAC provisions",
                    "Predictive dispute analysis"
                ],
                "examples": jurisdiction_config['compliance_rules'].get('contract', [])
            },
            "court_ruling": {
                "focus_areas": [
                    "Judicial reasoning",
                    "Statutory interpretation",
                    "Precedent alignment",
                    "Jurisdictional issues",
                    "Evidence handling",
                    "Court procedures",
                    "Predictive outcome analysis"
                ],
                "examples": jurisdiction_config['compliance_rules'].get('court_ruling', [])
            },
            "regulatory_filing": {
                "focus_areas": [
                    "Disclosure completeness",
                    "Reporting accuracy",
                    "Timeliness requirements",
                    f"{jurisdiction_config['regulator']} compliance",
                    "Audit trails",
                    "Data governance",
                    "Regulatory risk assessment"
                ],
                "examples": jurisdiction_config['compliance_rules'].get('regulatory_filing', [])
            },
            "policy": {
                "focus_areas": [
                    "Policy currency",
                    "Compliance risks",
                    "Workplace health and safety",
                    "Privacy compliance",
                    "Access controls",
                    "Policy enforcement",
                    "Impact analysis"
                ],
                "examples": jurisdiction_config['compliance_rules'].get('policy', [])
            },
            "unknown": {
                "focus_areas": ["General legal compliance"],
                "examples": []
            }
        }.get(doc_type, {
            "focus_areas": ["General legal compliance"],
            "examples": []
        })

        focus_areas = "\n".join(f"- {area}" for area in doc_type_config.get("focus_areas", []))
        examples = "\n".join(f"- {ex}" for ex in doc_type_config.get("examples", []))

        prediction_context = "\nPredicted Outcomes:\n" + "\n".join(
            f"- {pred[0]} (confidence: {pred[1]:.0%})"
            for pred in predictions[:3]
        ) if predictions else "\nNo strong predictive outcomes available"

        prompt = f"""As a {jurisdiction_config['name']} legal compliance expert, analyze this {doc_type}:

        Document Summary:
        {summary}

        Relevant Compliance Rules:
        {kb_context}

        {prediction_context}

        Key Focus Areas for {jurisdiction_config['name']}:
        {focus_areas}

        Jurisdiction-Specific Legislation:
        {", ".join(jurisdiction_config['compliance_rules'].values())}

        Provide detailed analysis in this EXACT format:

        COMPLIANCE VERDICT: [✅ COMPLIANT/⚠️ NEEDS REVIEW/❌ NON-COMPLIANT]
        RISK SCORE: [Low/Medium/High/Critical]

        STATUTORY REFERENCES:
        - [Full Act name Year (Jurisdiction) s Number]

        KEY ISSUES:
        - [Bullet point 1]
        - [Bullet point 2]

        RECOMMENDATIONS:
        - [Recommended action 1]
        - [Recommended action 2]

        Include analysis of:
        - Blockchain and audit trail provisions
        - Role-based access controls
        - Predicted risk factors
        - Compliance gaps specific to {jurisdiction_config['name']}"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
                temperature=0.2
            )
            analysis_text = response.choices[0].message.content

            return self._parse_analysis_results(analysis_text, text, summary, jurisdiction_config)
        except Exception as e:
            logger.error(f"Compliance analysis failed: {str(e)}")
            return self._fallback_analysis(text, doc_type, jurisdiction_config)

    def _parse_analysis_results(self, analysis_text: str, full_text: str, summary: str, jurisdiction_config: dict) -> Dict[str, Any]:
        """Parse and validate analysis results with jurisdiction context"""
        # Parse compliance status
        status = self._parse_compliance_status(analysis_text)

        # Parse risk score
        risk_score = self._parse_risk_score(analysis_text)

        # Extract statutory references
        statutory_refs = self._extract_statutory_references(analysis_text)
        if len(statutory_refs) < 3:  # Fallback to full text if few references found
            text_refs = self._extract_statutory_references(full_text[:20000])
            statutory_refs.extend(ref for ref in text_refs if ref not in statutory_refs)

        # Filter and validate references
        statutory_refs = self._filter_valid_references(statutory_refs, jurisdiction_config)[:20]

        # Extract issues and recommendations
        issues = self._extract_issues(analysis_text) or ["No critical issues identified"]

        return {
            "status": status,
            "risk_score": risk_score,
            "statutory_refs": statutory_refs,
            "issues": issues,
            "summary": summary,
            "full_analysis": analysis_text
        }

    def _fallback_analysis(self, text: str, doc_type: str, jurisdiction_config: dict) -> Dict[str, Any]:
        """Generate fallback analysis when main analysis fails"""
        statutory_refs = self._extract_statutory_references(text[:20000])
        return {
            "status": "⚠️ NEEDS REVIEW",
            "risk_score": "Medium",
            "statutory_refs": self._filter_valid_references(statutory_refs, jurisdiction_config)[:15],
            "issues": ["Full analysis failed - basic references extracted"],
            "summary": text[:500] + "... [truncated]",
            "full_analysis": f"Analysis failed - partial results shown for {jurisdiction_config['name']}"
        }

    def _parse_compliance_status(self, analysis_text: str) -> str:
        """Robust compliance status parsing"""
        status_patterns = {
            "✅ COMPLIANT": r'COMPLIANCE VERDICT:\s*([✅✅]+)',
            "⚠️ NEEDS REVIEW": r'COMPLIANCE VERDICT:\s*([⚠️⚠️]+)',
            "❌ NON-COMPLIANT": r'COMPLIANCE VERDICT:\s*([❌❌]+)'
        }

        for status, pattern in status_patterns.items():
            if re.search(pattern, analysis_text, re.IGNORECASE):
                return status

        # Fallback patterns
        if re.search(r'\bcompliant\b', analysis_text, re.IGNORECASE):
            return "✅ COMPLIANT"
        elif re.search(r'non.?compliant', analysis_text, re.IGNORECASE):
            return "❌ NON-COMPLIANT"

        return "⚠️ NEEDS REVIEW"

    def _parse_risk_score(self, analysis_text: str) -> str:
        """Comprehensive risk score parsing"""
        risk_pattern = r'RISK SCORE:\s*([A-Za-z]+)'
        match = re.search(risk_pattern, analysis_text, re.IGNORECASE)
        if match:
            risk = match.group(1).capitalize()
            if risk in ["Low", "Medium", "High", "Critical"]:
                return risk

        # Fallback risk assessment
        if re.search(r'high risk|critical', analysis_text, re.IGNORECASE):
            return "High"
        elif re.search(r'medium risk', analysis_text, re.IGNORECASE):
            return "Medium"
        elif re.search(r'low risk', analysis_text, re.IGNORECASE):
            return "Low"

        return "Unknown"

    def _extract_statutory_references(self, text: str) -> List[str]:
        """Comprehensive statutory reference extraction for Australian law"""
        patterns = [
            # Standard Act reference: Privacy Act 1988 (Cth) s 6
            r'(\b[A-Z][A-Za-z\s]+?\b)\s+(\d{4})\s*(?:\(([A-Za-z]{2,4})\))?\s+[Ss](?:ection|\.?)\s*(\d+[A-Z]*)',

            # Act without section: Corporations Act 2001 (Cth)
            r'(\b[A-Z][A-Za-z\s]+?\b)\s+(\d{4})\s*(?:\(([A-Za-z]{2,4})\))',

            # Regulation reference: Corporations Regulations 2001 (Cth) reg 7.1
            r'(\b[A-Z][A-Za-z\s]+?\b)\s+Regulations?\s+(\d{4})\s*(?:\(([A-Za-z]{2,4})\))?\s+[Rr]eg(?:ulation)?\s*([\d.]+)',

            # Short form: CA s 911A
            r'\b([A-Z]{2,})\s+[Ss](?:ection|\.?)\s*(\d+[A-Z]*)',

            # Act names without year: Privacy Act, Corporations Act
            r'\b([A-Z][A-Za-z]+?\s+Act)\b'
        ]

        references = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                groups = match.groups()
                act = groups[0].strip() if groups[0] else ""
                year = groups[1] if len(groups) > 1 and groups[1] else ""
                jur = groups[2] if len(groups) > 2 and groups[2] else ""
                section = groups[3] if len(groups) > 3 and groups[3] else ""

                ref = act
                if year:
                    ref += f" {year}"
                if jur:
                    ref += f" ({jur})"
                if section:
                    ref += f" s {section}"

                references.append(ref)

        # Handle common abbreviations
        abbrev_map = {
            'CA': 'Corporations Act 2001 (Cth)',
            'PA': 'Privacy Act 1988 (Cth)',
            'FTA': 'Fair Trading Act 1999 (Vic)',
            'ACL': 'Australian Consumer Law',
            'SIS': 'Superannuation Industry (Supervision) Act 1993 (Cth)'
        }

        for abbrev, full_name in abbrev_map.items():
            if f"{abbrev} s " in text or f"{abbrev}," in text:
                references.append(full_name)

        return sorted(list(set(ref for ref in references if len(ref.split()) >= 2)))

    def _filter_valid_references(self, references: List[str], jurisdiction_config: dict) -> List[str]:
        """Validate and filter statutory references with jurisdiction awareness"""
        valid_refs = []
        year_pattern = r'(19|20)\d{2}'
        act_pattern = r'[A-Z][A-Za-z]+\s+[A-Za-z]+'

        for ref in references:
            ref = ref.strip()

            # Must contain "Act" or "Regulation"
            if not any(word in ref for word in ['Act', 'Regulation', 'Legislation']):
                continue

            # Must have a year or be a known abbreviation
            if not re.search(year_pattern, ref):
                if not any(abbrev in ref for abbrev in ['ACL', 'UEL']):  # Common abbreviations
                    continue

            # Must have proper act naming
            if not re.search(act_pattern, ref):
                continue

            # Check jurisdiction if specified
            if jurisdiction_config and jurisdiction_config.get('strict_jurisdiction', False):
                if not any(jurisdiction_config['jurisdiction_code'].lower() in ref.lower() or 
                          jurisdiction_config['name'].lower() in ref.lower()):
                    continue

            valid_refs.append(ref)

        return valid_refs

    def _extract_issues(self, text: str) -> List[str]:
        """Advanced issue extraction with hierarchical parsing"""
        issues = []
        current_issue = ""
        in_issues_section = False

        lines = text.split('\n')
        for line in lines:
            line = line.strip()

            # Detect issues section
            if "KEY ISSUES:" in line or "CRITICAL ISSUES:" in line:
                in_issues_section = True
                continue

            # Detect end of section
            if in_issues_section and ("RECOMMENDATIONS:" in line or "STATUTORY REFERENCES:" in line):
                break

            # Process issue lines
            if in_issues_section:
                if line.startswith(("-", "*", "•")) or re.match(r'^\d+\.', line):
                    if current_issue:
                        issues.append(current_issue.strip())
                    current_issue = re.sub(r'^[-•*]\s*|\d+\.\s*', '', line)
                elif current_issue and line:
                    current_issue += " " + line

        if current_issue:
            issues.append(current_issue.strip())

        # Fallback to simple pattern if no structured issues found
        if not issues:
            issue_pattern = r'(?:non.?compli|missing|inadequate|fail|breach|risk)[\w\s-]+?(?=[.!?])'
            issues = list(set(m.group(0) for m in re.finditer(issue_pattern, text, re.IGNORECASE)))

        return issues[:10]  # Return top 10 issues max

    def _extract_recommendations(self, analysis_text: str) -> List[str]:
        """Extract recommendations from analysis text"""
        recommendations = []
        current_rec = ""
        in_rec_section = False

        lines = analysis_text.split('\n')
        for line in lines:
            line = line.strip()

            if "RECOMMENDATIONS:" in line:
                in_rec_section = True
                continue

            if in_rec_section and ("STATUTORY REFERENCES:" in line or "FULL ANALYSIS:" in line):
                break

            if in_rec_section:
                if line.startswith(("-", "*", "•")) or re.match(r'^\d+\.', line):
                    if current_rec:
                        recommendations.append(current_rec.strip())
                    current_rec = re.sub(r'^[-•*]\s*|\d+\.\s*', '', line)
                elif current_rec and line:
                    current_rec += " " + line

        if current_rec:
            recommendations.append(current_rec.strip())

        return recommendations or [
            "Review document with legal counsel",
            "Address identified compliance gaps"
        ]