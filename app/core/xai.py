# xai.py
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from pathlib import Path
import hashlib
import re
import torch
from groq import Groq
from sentence_transformers import SentenceTransformer
from .document import DocumentProcessor
from .predictive import PredictiveAnalytics
from app.config2 import CONFIG

logger = logging.getLogger(__name__)


# class XAITracker:
#     def __init__(self):
#         """Initialize XAI tracker with decision factors and visualization tracking"""
#         self.decision_factors = {}
#         self.kb_references = []
#         self.visualizations = []
#         self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         logger.info("XAI tracker initialized")

#     def log_decision(self, factor: str, value: Any, confidence: float):
#         """Track decision factor with validation"""
#         if not 0 <= confidence <= 1:
#             logger.warning(f"Invalid confidence {confidence} for factor {factor}")
#             confidence = max(0, min(1, confidence))

#         self.decision_factors[factor] = {
#             "value": value,
#             "confidence": round(confidence, 2),
#             "weight": round(confidence * 100),
#             "timestamp": datetime.now().isoformat()
#         }

#     def set_kb_references(self, references: list):
#         """Set knowledge base references with validation"""
#         self.kb_references = [
#             (ref[0][:500], float(ref[1]))  # Truncate and ensure float
#             for ref in references if len(ref) >= 2 and ref[1] > 0
#         ][:10]  # Keep top 10

#     def attach_visualization(self, name: str, path: str):
#         """Track visualization with validation"""
#         if not os.path.exists(path):
#             logger.warning(f"Visualization file not found: {path}")
#             return

#         self.visualizations.append({
#             "name": name,
#             "path": os.path.abspath(path),
#             "timestamp": datetime.now().isoformat()
#         })

#     def generate_visualization(self) -> Optional[str]:
#         """Generate decision factors visualization"""
#         if not self.decision_factors:
#             logger.warning("No decision factors to visualize")
#             return None

#         try:
#             # Prepare data
#             data = {
#                 k: v['weight']
#                 for k, v in self.decision_factors.items()
#                 if isinstance(v['weight'], (int, float))
#             }
#             if not data:
#                 return None

#             df = pd.DataFrame.from_dict(data, orient='index', columns=['Weight'])
#             df = df.sort_values('Weight', ascending=True)

#             # Create figure
#             plt.figure(figsize=(10, 6))
#             ax = df.plot(kind='barh', color='#3498db', edgecolor='black', alpha=0.7)

#             # Customize appearance
#             plt.title('Compliance Decision Factors', pad=20, fontsize=14)
#             plt.xlabel('Weight (%)', labelpad=10)
#             plt.ylabel('Factor', labelpad=10)
#             plt.grid(axis='x', alpha=0.3)
#             plt.tight_layout()

#             # Add value annotations
#             for i, (_, row) in enumerate(df.iterrows()):
#                 ax.text(row['Weight'] + 1, i, f"{row['Weight']:.0f}%",
#                        va='center', fontsize=10)

#             # Save to file
#             viz_dir = Path("visualizations")
#             viz_dir.mkdir(exist_ok=True)
#             img_path = str(viz_dir / f"decision_factors_{self.timestamp}.png")
#             plt.savefig(img_path, dpi=120, bbox_inches='tight')
#             plt.close()

#             return img_path

#         except Exception as e:
#             logger.error(f"Failed to generate visualization: {str(e)}")
#             return None

#     def generate_report(self, doc_path: str, doc_type: str, summary: str,
#                        analysis: str, factors: Dict[str, Any],
#                        visualizations: list = None) -> Dict[str, Any]:
#         """Generate comprehensive compliance report with enterprise features"""
#         try:
#             # Prepare metadata
#             doc_name = os.path.basename(doc_path)
#             report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             jurisdiction = factors.get("Jurisdiction", {})
#             legislation = factors.get("Relevant Legislation", [])

#             # Prepare statutory references
#             statutory_refs = [
#                 ref for ref in factors.get("Statutory References", [])
#                 if isinstance(ref, str) and ('Act' in ref or 'Regulation' in ref)
#             ][:20]  # Limit to top 20

#             # Prepare issues and recommendations
#             issues = [
#                 issue for issue in factors.get("Critical Issues", [])
#                 if isinstance(issue, str)
#             ][:10]  # Limit to top 10

#             recommendations = self._extract_recommendations(analysis)[:5]

#             # Prepare predictive outcomes
#             predictions = factors.get("Predictive Outcomes", [])
#             prediction_items = []
#             for pred in predictions[:3]:
#                 item = f"- {pred['prediction']} (confidence: {pred['confidence']:.0%})"
#                 if pred.get('source'):
#                     item += f"\n  Source: {pred['source']}"
#                 if pred.get('reference'):
#                     item += f" | Reference: {pred['reference']}"
#                 prediction_items.append(item)

#             # Prepare blockchain proof
#             blockchain_proof = factors.get("Blockchain Proof")
#             blockchain_section = ""
#             if blockchain_proof and isinstance(blockchain_proof, dict):
#                 blockchain_section = (
#                     f"\nBlockchain Confirmation:"
#                     f"\n- TX Hash: {blockchain_proof.get('tx_hash')}"
#                     f"\n- Block: {blockchain_proof.get('block_number')}"
#                     f"\n- Timestamp: {blockchain_proof.get('timestamp')}"
#                 )

#             # Prepare transaction info
#             transaction_info = factors.get("Transaction Info")
#             billing_section = ""
#             if transaction_info and isinstance(transaction_info, dict):
#                 billing_section = (
#                     f"\nBilling Information:"
#                     f"\n- Transaction ID: {transaction_info.get('transaction_id')}"
#                     f"\n- Amount: {transaction_info.get('amount')} {transaction_info.get('currency')}"
#                     f"\n- Tier: {transaction_info.get('tier', 'N/A')}"
#                 )

#             # Generate report content
#             report_content = f"""⚖️ LEGAL COMPLIANCE REPORT - {jurisdiction.get('name', 'Unknown Jurisdiction')} ⚖️
# ===========================================================
# Generated: {report_date}
# Document: {doc_name}
# Type: {doc_type.upper()}
# Jurisdiction: {jurisdiction.get('name', 'Unknown')}
# Document Hash: {DocumentProcessor.document_hash(summary)[:12]}

# JURISDICTION OVERVIEW
# =====================
# - Legal System: {jurisdiction.get('legal_system', 'Unknown')}
# - Currency: {jurisdiction.get('currency', 'N/A')}
# - Current Time: {jurisdiction.get('current_time', 'N/A')}
# - Relevant Legislation: {", ".join(legislation) or 'None specified'}

# DOCUMENT SUMMARY
# ================
# {summary}

# COMPLIANCE VERDICT
# ==================
# {self._extract_verdict(analysis)}

# RISK ASSESSMENT
# ===============
# {self._format_risk_profile(factors.get("Risk Profile", {}))}

# STATUTORY REFERENCES
# ====================
# {" ".join(f"- {ref}" for ref in statutory_refs) if statutory_refs else "- No statutory references identified"}

# CRITICAL ISSUES
# ===============
# {" ".join(f"- {issue}" for issue in issues) if issues else "- No critical issues identified"}

# RECOMMENDATIONS
# ===============
# {" ".join(f"- {rec}" for rec in recommendations) if recommendations else "- No specific recommendations provided"}

# PREDICTIVE OUTCOMES
# ===================
# {" ".join(prediction_items) if prediction_items else "- No strong predictive outcomes"}

# {blockchain_section}
# {billing_section}

# FULL ANALYSIS
# =============
# {analysis}
# """

#             # Save report to file
#             report_path = f"compliance_report_{self.timestamp}.txt"
#             with open(report_path, "w", encoding="utf-8") as f:
#                 f.write(report_content)

#             # Generate and attach visualizations
#             decision_viz = self.generate_visualization()
#             viz_paths = []
#             if decision_viz:
#                 self.attach_visualization("Decision Factors", decision_viz)

#             # Include any additional visualizations
#             if visualizations:
#                 for viz in visualizations:
#                     if os.path.exists(viz):
#                         viz_name = os.path.basename(viz).split('.')[0].replace('_', ' ').title()
#                         self.attach_visualization(viz_name, viz)

#             viz_paths = [viz['path'] for viz in self.visualizations]

#             logger.info(f"Report generated at {report_path}")
#             return {
#                 "text_path": report_path,
#                 "viz_paths": viz_paths,
#                 "timestamp": datetime.now().isoformat()
#             }

#         except Exception as e:
#             logger.error(f"Report generation failed: {str(e)}")
#             return {
#                 "error": str(e),
#                 "timestamp": datetime.now().isoformat()
#             }

#     def _format_risk_profile(self, risk_profile: Dict[str, float]) -> str:
#         """Format risk profile for report"""
#         if not risk_profile:
#             return "- Risk assessment not available"

#         risk_items = []
#         for risk, score in risk_profile.items():
#             if isinstance(score, (int, float)):
#                 # Color code based on risk level
#                 if score >= 0.75:
#                     indicator = "🔴"
#                 elif score >= 0.5:
#                     indicator = "🟠"
#                 elif score >= 0.25:
#                     indicator = "🟡"
#                 else:
#                     indicator = "🟢"

#                 risk_items.append(f"{indicator} {risk.replace('_', ' ').title()}: {score:.0%}")

#         return "\n".join(risk_items) if risk_items else "- Risk assessment not available"

# # ========================
# # SYSTEM INITIALIZATION (Updated)
# # ========================
# def initialize_system() -> Tuple[Groq, SentenceTransformer, PredictiveAnalytics]:
#     """Initialize system components with comprehensive error handling"""
#     try:
#         # Configure CUDA if available
#         os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         logger.info(f"Using device: {device.upper()}")

#         # Initialize Groq client
#         groq_client = Groq(api_key="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")
#         logger.info("Groq client initialized")

#         # Load sentence transformer model
#         try:
#             model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
#             logger.info(f"Model loaded on {device.upper()}")
#         except Exception as e:
#             logger.warning(f"Failed to load model on {device}: {str(e)}")
#             model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
#             logger.info("Model loaded on CPU")

#         # Initialize predictive analytics with citations
#         predictive_analytics = PredictiveAnalytics(model)
#         logger.info("Predictive analytics initialized")

#         return groq_client, model, predictive_analytics

#     except Exception as e:
#         logger.error(f"System initialization failed: {str(e)}")
#         raise RuntimeError(f"Initialization failed: {str(e)}")


class XAITracker:
    def __init__(self):
        self.decision_factors = {}
        self.kb_references = []
        self.visualizations = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("XAI tracker initialized")

    def log_decision(self, factor: str, value: Any, confidence: float):
        """Track decision factor with validation"""
        if not 0 <= confidence <= 1:
            logger.warning(f"Invalid confidence {confidence} for factor {factor}")
            confidence = max(0, min(1, confidence))

        self.decision_factors[factor] = {
            "value": value,
            "confidence": round(confidence, 2),
            "weight": round(confidence * 100),
            "timestamp": datetime.now().isoformat()
        }

    def set_kb_references(self, references: list):
        """Set knowledge base references with validation"""
        self.kb_references = [
            (ref[0][:500], float(ref[1]))  # Truncate and ensure float
            for ref in references if len(ref) >= 2 and ref[1] > 0
        ][:10]  # Keep top 10

    def attach_visualization(self, name: str, path: str):
        """Track visualization with validation"""
        if os.path.exists(path):
            self.visualizations.append({
                "name": name,
                "path": os.path.abspath(path),
                "timestamp": datetime.now().isoformat()
            })
        else:
            logger.warning(f"Visualization file not found: {path}")

    def generate_visualization(self) -> str:
        """Generate decision factors visualization"""
        if not self.decision_factors:
            return ""

        data = {
            k: v['weight']
            for k, v in self.decision_factors.items()
            if isinstance(v['weight'], (int, float))
        }
        if not data:
            return ""

        df = pd.DataFrame.from_dict(data, orient='index', columns=['Weight'])
        df = df.sort_values('Weight', ascending=True)

        plt.figure(figsize=(10, 6))
        ax = df.plot(kind='barh', color='#3498db', edgecolor='black', alpha=0.7)

        plt.title('Compliance Decision Factors', pad=20, fontsize=14)
        plt.xlabel('Weight (%)', labelpad=10)
        plt.ylabel('Factor', labelpad=10)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        for i, (_, row) in enumerate(df.iterrows()):
            ax.text(row['Weight'] + 1, i, f"{row['Weight']:.0f}%",
                   va='center', fontsize=10)

        img_path = f"decision_factors_{self.timestamp}.png"
        plt.savefig(img_path, dpi=120, bbox_inches='tight')
        plt.close()

        return img_path

    def generate_report(self, doc_path: str, doc_type: str, summary: str,
                       analysis: str, factors: Dict[str, list],
                       visualizations: list = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        try:
            doc_name = os.path.basename(doc_path)
            report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            statutory_refs = [
                ref for ref in factors.get("Statutory References", [])
                if isinstance(ref, str) and ('Act' in ref or 'Regulation' in ref)
            ][:20]

            issues = [
                issue for issue in factors.get("Critical Issues", [])
                if isinstance(issue, str)
            ][:10]

            recommendations = self._extract_recommendations(analysis, doc_type)[:5]

            predictions = factors.get("Predictive Outcomes", [])
            prediction_items = [
                f"- {pred[0]} (confidence: {pred[1]:.0%})"
                for pred in predictions
                if isinstance(pred, (list, tuple)) and len(pred) >= 2
            ][:3]

            risk_profile = factors.get("Risk Profile", {})
            risk_items = [
                f"- {k}: {v:.0%}"
                for k, v in risk_profile.items()
                if isinstance(v, (int, float))
            ]

            kb_refs = [
                f"{i+1}. {ref[0][:100]}... (relevance: {ref[1]:.2f})"
                for i, ref in enumerate(self.kb_references[:3])
            ]

            risk_section = "\n".join(risk_items) if risk_items else "- Risk assessment not available"
            decision_factors = "\n".join(
                f"- {k}: {v['value']} (confidence: {v['confidence']:.0%})"
                for k, v in self.decision_factors.items()
            )
            statutory_section = "\n".join(f"- {ref}" for ref in statutory_refs) if statutory_refs else "- No statutory references identified"
            issues_section = "\n".join(f"- {issue}" for issue in issues) if issues else "- No critical issues identified"
            recommendations_section = "\n".join(f"- {rec}" for rec in recommendations) if recommendations else "- No specific recommendations provided"
            predictions_section = "\n".join(prediction_items) if prediction_items else "- No strong predictive outcomes"
            kb_section = "\n".join(kb_refs) if kb_refs else "- No highly relevant references found"

            report_content = f"""⚖️ AUSTRALIAN LEGAL COMPLIANCE REPORT ⚖️
===========================================
Generated: {report_date}
Document: {doc_name}
Type: {doc_type.upper()}
Document Hash: {DocumentProcessor.document_hash(summary)[:12]}

DOCUMENT SUMMARY
================
{summary}

COMPLIANCE VERDICT
==================
{self._extract_verdict(analysis)}

RISK ASSESSMENT
===============
{risk_section}

DECISION FACTORS
================
{decision_factors}

STATUTORY REFERENCES
====================
{statutory_section}

CRITICAL ISSUES
===============
{issues_section}

RECOMMENDATIONS
===============
{recommendations_section}

PREDICTIVE OUTCOMES
===================
{predictions_section}

KNOWLEDGE REFERENCES
====================
{kb_section}

FULL ANALYSIS
=============
{analysis}
"""

            report_path = f"compliance_report_{self.timestamp}.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)

            decision_viz = self.generate_visualization()
            viz_paths = []
            if decision_viz:
                self.attach_visualization("Decision Factors", decision_viz)

            if visualizations:
                for viz in visualizations:
                    if os.path.exists(viz):
                        viz_name = os.path.basename(viz).split('.')[0].replace('_', ' ').title()
                        self.attach_visualization(viz_name, viz)

            viz_paths = [viz['path'] for viz in self.visualizations]

            logger.info(f"Report generated at {report_path}")
            return {
                "text_path": report_path,
                "viz_paths": viz_paths,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _extract_recommendations(self, analysis_text: str, doc_type: str) -> List[str]:
        """Structured recommendation extraction with doc-type templates"""
        recommendations = []
        in_rec_section = False

        for line in analysis_text.split('\n'):
            line = line.strip()
            if "RECOMMENDATIONS:" in line:
                in_rec_section = True
                continue
            if in_rec_section and line.startswith(("-", "*", "•", "1.", "2.")):
                rec = re.sub(r'^[-•*]\s*|\d+\.\s*', '', line)
                recommendations.append(rec)

        if not recommendations:
            if doc_type == "court_ruling":
                return [
                    "Verify jurisdictional requirements for the transfer",
                    "Assess impact on existing orders and procedures",
                    "Consult with all parties about transfer implications",
                    "Review comparable transfer cases for precedent"
                ]
            elif doc_type == "contract":
                return ["Review with legal counsel", "Clarify ambiguous terms"]
            else:
                return ["Consult legal expert for case-specific advice"]

        cleaned_recs = []
        for rec in recommendations[:5]:
            rec = rec.split('.')[0]
            rec = re.sub(r'in terms of blockchain|rbac', '', rec, flags=re.IGNORECASE)
            if len(rec) > 10 and not rec.startswith(("http", "www")):
                cleaned_recs.append(rec.strip())

        return cleaned_recs or ["Consult legal expert for case-specific advice"]

    def _extract_verdict(self, analysis_text: str) -> str:
        """Extract and format compliance verdict"""
        verdict_pattern = r'COMPLIANCE VERDICT:\s*([✅⚠️❌]+.*)'
        match = re.search(verdict_pattern, analysis_text)
        if match:
            return match.group(1).strip()

        if "COMPLIANT" in analysis_text.upper():
            return CONFIG['COMPLIANCE_LABELS']['COMPLIANT']
        elif "NON-COMPLIANT" in analysis_text.upper():
            return CONFIG['COMPLIANCE_LABELS']['NON_COMPLIANT']

        return CONFIG['COMPLIANCE_LABELS']['NEEDS_REVIEW']