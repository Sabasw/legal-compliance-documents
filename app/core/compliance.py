import os
import re
import faiss
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
from groq import Groq
from sentence_transformers import SentenceTransformer
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict
import logging
import hashlib
from app.core.predictive import PredictiveAnalytics
from app.core.knowledge import KnowledgeBase
from app.core.document import DocumentProcessor, Document
from app.config2 import CONFIG
from app.core.xai import XAITracker

logger = logging.getLogger(__name__)

class ComplianceAnalyzer:
    def __init__(self, groq_client: Groq, model: SentenceTransformer, predictive_analytics: PredictiveAnalytics):
        self.client = groq_client
        self.model = model
        self.predictive_analytics = predictive_analytics
        self.kb_index = None
        self.kb_chunks = []
        logger.info("Compliance analyzer initialized")

    def load_knowledge_base(self, index_path: str, chunks_path: str) -> bool:
        """Load knowledge base with validation"""
        self.kb_index, self.kb_chunks = KnowledgeBase.load_knowledge_base(index_path, chunks_path)
        success = self.kb_index is not None
        if not success:
            logger.warning("Proceeding without knowledge base")
        return success

    def analyze_document(self, doc_path: str) -> Dict[str, Any]:
        """Complete document analysis pipeline with enhanced error handling"""
        doc_hash = ""
        try:
            xai = XAITracker()
            text = DocumentProcessor.extract_text(doc_path)
            if not text:
                raise ValueError("Text extraction failed")

            doc_hash = DocumentProcessor.document_hash(text)
            text = text[:CONFIG['MAX_TEXT_LENGTH']]
            logger.info(f"Analyzing document {doc_path} (hash: {doc_hash[:8]}...)")

            doc_type = self._classify_document(text)
            xai.log_decision("Document Type", doc_type, 0.85)
            logger.info(f"Classified as: {doc_type}")

            risk_profile = self.predictive_analytics.generate_risk_profile(text, doc_type)
            predictions = self.predictive_analytics.predict_legal_outcomes(text, doc_type)

            risk_fig = self.predictive_analytics.plot_risk_profile(risk_profile)
            risk_viz_path = f"risk_profile_{doc_hash[:8]}_{int(datetime.now().timestamp())}.png"
            risk_fig.savefig(risk_viz_path, bbox_inches='tight', dpi=120)
            plt.close(risk_fig)

            xai.attach_visualization("Risk Profile", risk_viz_path)
            xai.log_decision("Risk Profile", risk_profile, 0.85)
            xai.log_decision("Predicted Outcomes", predictions, 0.9)

            kb_results = []
            if self.kb_index and self.kb_chunks:
                kb_results = KnowledgeBase.query_knowledge_base(
                    text, self.model, self.kb_index, self.kb_chunks, doc_type, CONFIG['TOP_K_RULES']
                )
                xai.set_kb_references(kb_results)
                xai.log_decision("Relevant Rules Found", len(kb_results), min(len(kb_results)/5, 1.0))

            summary = self._generate_summary(text, doc_path, kb_results, doc_type, predictions)
            xai.log_decision("Summary Quality", len(summary.split()) > 100, 0.8)

            analysis = self._perform_compliance_analysis(text, doc_type, kb_results, summary, predictions)
            xai.log_decision("Compliance Status", analysis['status'], 0.9)
            xai.log_decision("Risk Level", analysis['risk_score'], 0.8)

            report = xai.generate_report(
                doc_path=doc_path,
                doc_type=doc_type,
                summary=summary,
                analysis=analysis['full_analysis'],
                factors={
                    "Statutory References": analysis['statutory_refs'],
                    "Critical Issues": analysis['issues'],
                    "Predictive Outcomes": predictions,
                    "Risk Profile": risk_profile
                },
                visualizations=[risk_viz_path]
            )

            logger.info(f"Analysis completed for {doc_path}")
            return {
                "document_hash": doc_hash,
                "document_type": doc_type,
                "compliance_status": analysis['status'],
                "risk_score": analysis['risk_score'],
                "risk_profile": risk_profile,
                "report_path": report['text_path'],
                "visualization_paths": report['viz_paths'],
                "statutory_analysis": analysis['statutory_refs'],
                "predictive_outcomes": predictions,
                "summary": summary,
                "full_analysis": analysis['full_analysis'],
                "kb_references": [r[0] for r in kb_results[:3]],
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Analysis failed for {doc_path}: {str(e)}")
            return {
                "error": str(e),
                "document_hash": doc_hash,
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }

    def _classify_document(self, text: str) -> str:
        """Comprehensive document classification"""
        if self._is_court_ruling(text):
            return "court_ruling"

        if self.kb_index and self.kb_chunks:
            doc_type = self._classify_with_knowledge_base(text)
            if doc_type in CONFIG['DOCUMENT_TYPES']:
                return doc_type

        return self._classify_with_groq(text[:2000])

    def _is_court_ruling(self, text: str) -> bool:
        """Enhanced court ruling detection"""
        first_page = text[:2000]
        for pattern in CONFIG['COURT_PATTERNS']:
            if re.search(pattern, first_page):
                return True
        return False

    def _classify_with_knowledge_base(self, text: str) -> str:
        """Knowledge-based classification"""
        classification_patterns = {
            "contract": ["contract", "agreement", "clause", "party", "term"],
            "court_ruling": ["judgment", "court", "ruling", "decision", "appeal", "transfer"],
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

        Options: {", ".join(CONFIG['DOCUMENT_TYPES'])}
        Respond ONLY with the document type from the options provided."""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=20
            )
            classification = response.choices[0].message.content.strip().lower()
            return classification if classification in CONFIG['DOCUMENT_TYPES'] else "unknown"
        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            return "unknown"

    def _generate_summary(self, text: str, doc_path: str, kb_results: list,
                         doc_type: str, predictions: list) -> str:
        """Generate comprehensive legal summary"""
        kb_context = "\n".join(
            f"Relevant Rule {i+1}: {rule[0][:300]} (Relevance: {rule[1]:.2f})"
            for i, rule in enumerate(kb_results[:3])
        ) if kb_results else "No highly relevant rules identified"

        doc_type_info = CONFIG['DOC_TYPE_PROMPTS'].get(doc_type, CONFIG['DOC_TYPE_PROMPTS']['unknown'])
        focus_areas = "\n".join(f"- {area}" for area in doc_type_info.get("focus_areas", []))
        examples = "\n".join(f"- {ex}" for ex in doc_type_info.get("examples", []))

        prediction_context = "\nPredicted Outcomes:\n" + "\n".join(
            f"- {pred[0]} (confidence: {pred[1]:.0%})"
            for pred in predictions[:3]
        ) if predictions else "\nNo strong predictive outcomes identified"

        prompt = f"""As an Australian legal compliance expert, generate a comprehensive summary of this {doc_type}:

        Document Excerpt:
        {text[:4000]}

        Key Focus Areas for {doc_type}:
        {focus_areas}

        Relevant Compliance Context:
        {kb_context}

        {prediction_context}

        Example References:
        {examples}

        Create a detailed summary (500-700 words) covering:
        1. Document purpose and key parties
        2. Main legal obligations and requirements
        3. Compliance risks and issues
        4. Relevant Australian legal frameworks
        5. Case transfer implications (if applicable)
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
            return self._fallback_summary(text, doc_type)

    def _fallback_summary(self, text: str, doc_type: str) -> str:
        """Generate fallback summary when LLM fails"""
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        key_sentences = sentences[:10] + sentences[-5:] if len(sentences) > 15 else sentences
        return (
            f"Basic Summary of {doc_type}:\n\n" +
            "\n".join(f"- {s}" for s in key_sentences) +
            "\n\n[Full analysis unavailable due to summary generation error]"
        )

    def _perform_compliance_analysis(self, text: str, doc_type: str,
                                   kb_results: list, summary: str,
                                   predictions: list) -> Dict[str, Any]:
        """Comprehensive compliance analysis with doc-type specific templates"""
        kb_context = "\n".join(
            f"{i+1}. {rule[0][:300]} (Relevance: {rule[1]:.2f})"
            for i, rule in enumerate(kb_results[:3])
        ) if kb_results else "No highly relevant compliance rules found"

        doc_type_config = CONFIG['DOC_TYPE_PROMPTS'].get(doc_type, CONFIG['DOC_TYPE_PROMPTS']['unknown'])
        focus_areas = "\n".join(f"- {area}" for area in doc_type_config.get("focus_areas", []))
        examples = "\n".join(f"- {ex}" for ex in doc_type_config.get("examples", []))

        prediction_context = "\nPredicted Outcomes:\n" + "\n".join(
            f"- {pred[0]} (confidence: {pred[1]:.0%})"
            for pred in predictions[:3]
        ) if predictions else "\nNo strong predictive outcomes available"

        if doc_type == "court_ruling":
            prompt = f"""As an Australian court compliance expert, analyze this ruling:

            Document Summary:
            {summary}

            Relevant Compliance Rules:
            {kb_context}

            {prediction_context}

            Key Focus Areas:
            {focus_areas}

            Example References:
            {examples}

            Provide detailed analysis in this EXACT format:

            COMPLIANCE VERDICT: [✅ COMPLIANT/⚠️ NEEDS REVIEW/❌ NON-COMPLIANT]
            RISK SCORE: [Low/Medium/High/Critical]

            JURISDICTIONAL ANALYSIS:
            - [Key jurisdictional considerations]
            - [Court hierarchy implications]

            CASE TRANSFER IMPLICATIONS:
            - [Impact on existing orders]
            - [Procedural consequences]

            RECOMMENDATIONS:
            - [Action 1]
            - [Action 2]"""
        else:
            prompt = f"""As an Australian legal compliance expert, analyze this {doc_type}:

            Document Summary:
            {summary}

            Relevant Compliance Rules:
            {kb_context}

            {prediction_context}

            Key Focus Areas:
            {focus_areas}

            Example References:
            {examples}

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
            - [Recommended action 2]"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
                temperature=0.2
            )
            analysis_text = response.choices[0].message.content
            return self._parse_analysis_results(analysis_text, text, summary, doc_type)
        except Exception as e:
            logger.error(f"Compliance analysis failed: {str(e)}")
            return self._fallback_analysis(text, doc_type)

    def _parse_analysis_results(self, analysis_text: str, full_text: str, summary: str, doc_type: str) -> Dict[str, Any]:
        """Parse and validate analysis results with doc-type awareness"""
        status = self._parse_compliance_status(analysis_text)
        risk_score = self._parse_risk_score(analysis_text)
        statutory_refs = self._extract_statutory_references(analysis_text)

        if len(statutory_refs) < 3:
            text_refs = self._extract_statutory_references(full_text[:20000])
            statutory_refs.extend(ref for ref in text_refs if ref not in statutory_refs)

        statutory_refs = self._filter_valid_references(statutory_refs)[:20]

        if doc_type == "court_ruling":
            issues = self._extract_court_ruling_issues(analysis_text, full_text)
        else:
            issues = self._extract_issues(analysis_text)

        return {
            "status": status,
            "risk_score": risk_score,
            "statutory_refs": statutory_refs,
            "issues": issues or ["No critical issues identified"],
            "summary": summary,
            "full_analysis": analysis_text
        }

    def _extract_court_ruling_issues(self, analysis_text: str, full_text: str) -> List[str]:
        """Specialized issue extraction for court rulings"""
        issues = []

        # Extract from structured analysis
        in_issues_section = False
        for line in analysis_text.split('\n'):
            line = line.strip()
            if "JURISDICTIONAL ANALYSIS:" in line or "CASE TRANSFER IMPLICATIONS:" in line:
                in_issues_section = True
                continue
            if in_issues_section and line.startswith(("-", "*", "•")):
                issues.append(re.sub(r'^[-•*]\s*', '', line))
            if in_issues_section and "RECOMMENDATIONS:" in line:
                break

        # Fallback patterns for transfer cases
        if len(issues) < 2:
            transfer_patterns = [
                r'(transfer|removal)\s+(of|to)\s+[A-Z]',
                r'jurisdict(ion|ional)\s+(issue|consideration)',
                r'enforceability\s+of\s+[a-z]+\s+order',
                r'conflict\s+(of|between)\s+laws?'
            ]
            for pattern in transfer_patterns:
                for match in re.finditer(pattern, full_text[:5000], re.IGNORECASE):
                    context = full_text[max(0, match.start()-50):match.end()+50]
                    issues.append(f"{match.group().title()}... {context[:100]}...")

        return list(set(issues[:5]))  # Dedupe and limit

    def _fallback_analysis(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Generate fallback analysis when main analysis fails"""
        statutory_refs = self._extract_statutory_references(text[:20000])

        if doc_type == "court_ruling":
            issues = [
                "Full analysis failed - potential jurisdictional issues",
                "Case transfer implications require review"
            ]
        else:
            issues = ["Full analysis failed - basic references extracted"]

        return {
            "status": CONFIG['COMPLIANCE_LABELS']['NEEDS_REVIEW'],
            "risk_score": "Medium",
            "statutory_refs": self._filter_valid_references(statutory_refs)[:15],
            "issues": issues,
            "summary": text[:500] + "... [truncated]",
            "full_analysis": "Analysis failed - partial results shown"
        }

    def _parse_compliance_status(self, analysis_text: str) -> str:
        """Robust compliance status parsing"""
        status_patterns = {
            CONFIG['COMPLIANCE_LABELS']['COMPLIANT']: r'COMPLIANCE VERDICT:\s*([✅✅]+)',
            CONFIG['COMPLIANCE_LABELS']['NEEDS_REVIEW']: r'COMPLIANCE VERDICT:\s*([⚠️⚠️]+)',
            CONFIG['COMPLIANCE_LABELS']['NON_COMPLIANT']: r'COMPLIANCE VERDICT:\s*([❌❌]+)'
        }

        for status, pattern in status_patterns.items():
            if re.search(pattern, analysis_text, re.IGNORECASE):
                return status

        if re.search(r'\bcompliant\b', analysis_text, re.IGNORECASE):
            return CONFIG['COMPLIANCE_LABELS']['COMPLIANT']
        elif re.search(r'non.?compliant', analysis_text, re.IGNORECASE):
            return CONFIG['COMPLIANCE_LABELS']['NON_COMPLIANT']

        return CONFIG['COMPLIANCE_LABELS']['NEEDS_REVIEW']

    def _parse_risk_score(self, analysis_text: str) -> str:
        """Comprehensive risk score parsing"""
        risk_pattern = r'RISK SCORE:\s*([A-Za-z]+)'
        match = re.search(risk_pattern, analysis_text, re.IGNORECASE)
        if match:
            risk = match.group(1).capitalize()
            if risk in CONFIG['RISK_LEVELS']:
                return risk

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
            r'(\b[A-Z][A-Za-z\s]+?\b)\s+(\d{4})\s*(?:\(([A-Za-z]{2,4})\))?\s+[Ss](?:ection|\.?)\s*(\d+[A-Z]*)',
            r'(\b[A-Z][A-Za-z\s]+?\b)\s+(\d{4})\s*(?:\(([A-Za-z]{2,4})\))',
            r'(\b[A-Z][A-Za-z\s]+?\b)\s+Regulations?\s+(\d{4})\s*(?:\(([A-Za-z]{2,4})\))?\s+[Rr]eg(?:ulation)?\s*([\d.]+)',
            r'\b([A-Z]{2,})\s+[Ss](?:ection|\.?)\s*(\d+[A-Z]*)',
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

        abbrev_map = {
            'CA': 'Corporations Act 2001 (Cth)',
            'PA': 'Privacy Act 1988 (Cth)',
            'FTA': 'Fair Trading Act 1999 (Vic)',
            'ACL': 'Australian Consumer Law',
            'SIS': 'Superannuation Industry (Supervision) Act 1993 (Cth)',
            'JUD': 'Judiciary Act 1903 (Cth)'
        }

        for abbrev, full_name in abbrev_map.items():
            if f"{abbrev} s " in text or f"{abbrev}," in text:
                references.append(full_name)

        return sorted(list(set(ref for ref in references if len(ref.split()) >= 2)))

    def _filter_valid_references(self, references: List[str]) -> List[str]:
        """Validate and filter statutory references"""
        valid_refs = []
        year_pattern = r'(19|20)\d{2}'
        act_pattern = r'[A-Z][A-Za-z]+\s+[A-Za-z]+'

        for ref in references:
            ref = ref.strip()

            if not any(word in ref for word in ['Act', 'Regulation', 'Legislation']):
                continue

            if not re.search(year_pattern, ref):
                if not any(abbrev in ref for abbrev in ['ACL', 'UEL', 'JUD']):
                    continue

            if not re.search(act_pattern, ref):
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

            if "KEY ISSUES:" in line or "CRITICAL ISSUES:" in line:
                in_issues_section = True
                continue

            if in_issues_section and ("RECOMMENDATIONS:" in line or "STATUTORY REFERENCES:" in line):
                break

            if in_issues_section:
                if line.startswith(("-", "*", "•")) or re.match(r'^\d+\.', line):
                    if current_issue:
                        issues.append(current_issue.strip())
                    current_issue = re.sub(r'^[-•*]\s*|\d+\.\s*', '', line)
                elif current_issue and line:
                    current_issue += " " + line

        if current_issue:
            issues.append(current_issue.strip())

        if not issues:
            issue_pattern = r'(?:non.?compli|missing|inadequate|fail|breach|risk)[\w\s-]+?(?=[.!?])'
            issues = list(set(m.group(0) for m in re.finditer(issue_pattern, text, re.IGNORECASE)))

        return issues[:10]