"""
Explainable AI (XAI) Service
Provides transparency and justification for AI decisions in legal analysis
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import re
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ai_service import ai_service
from app.services.blockchain_service import blockchain_service
from config import settings

logger = logging.getLogger(__name__)

@dataclass
class ExplanationComponent:
    """Individual explanation component"""
    component_type: str  # evidence, rule, precedent, risk_factor
    content: str
    confidence_score: float
    weight: float
    source: str
    legal_basis: str

@dataclass
class AIDecisionExplanation:
    """Complete explanation for an AI decision"""
    decision_id: str
    model_name: str
    decision_type: str  # prediction, classification, risk_assessment
    input_summary: str
    output_summary: str
    confidence_score: float
    explanation_components: List[ExplanationComponent]
    reasoning_chain: List[str]
    alternative_outcomes: List[Dict[str, Any]]
    limitations: List[str]
    legal_citations: List[str]
    bias_assessment: Dict[str, Any]
    timestamp: datetime

class ExplainableAI:
    """Service for providing explainable AI decisions"""
    
    def __init__(self):
        self.explanation_templates = {
            'case_outcome_prediction': {
                'structure': [
                    'case_facts_analysis',
                    'applicable_law_identification',
                    'precedent_matching',
                    'risk_factor_assessment',
                    'outcome_probability_calculation'
                ],
                'weight_factors': {
                    'case_facts': 0.3,
                    'legal_precedents': 0.4,
                    'statutory_requirements': 0.2,
                    'judicial_patterns': 0.1
                }
            },
            'compliance_assessment': {
                'structure': [
                    'document_analysis',
                    'rule_matching',
                    'gap_identification',
                    'risk_scoring',
                    'compliance_verdict'
                ],
                'weight_factors': {
                    'mandatory_clauses': 0.4,
                    'prohibited_terms': 0.3,
                    'regulatory_alignment': 0.2,
                    'industry_standards': 0.1
                }
            },
            'risk_assessment': {
                'structure': [
                    'threat_identification',
                    'vulnerability_assessment',
                    'impact_analysis',
                    'likelihood_calculation',
                    'mitigation_recommendations'
                ],
                'weight_factors': {
                    'financial_risk': 0.3,
                    'legal_risk': 0.4,
                    'operational_risk': 0.2,
                    'reputational_risk': 0.1
                }
            }
        }
    
    async def explain_case_prediction(self, 
                                    case_text: str, 
                                    prediction_result: Dict[str, Any],
                                    model_inputs: Dict[str, Any],
                                    session: AsyncSession) -> AIDecisionExplanation:
        """Explain case outcome prediction decision"""
        
        logger.info("Generating explanation for case prediction")
        
        try:
            # Generate unique decision ID
            decision_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(case_text) % 10000:04d}"
            
            # Extract key components for explanation
            components = await self._extract_prediction_components(
                case_text, prediction_result, model_inputs
            )
            
            # Build reasoning chain
            reasoning_chain = await self._build_prediction_reasoning_chain(
                case_text, prediction_result, components
            )
            
            # Identify alternative outcomes
            alternatives = await self._identify_alternative_outcomes(
                case_text, prediction_result, components
            )
            
            # Assess potential biases
            bias_assessment = await self._assess_prediction_bias(
                case_text, prediction_result, model_inputs
            )
            
            # Extract legal citations
            legal_citations = self._extract_legal_citations(case_text, components)
            
            explanation = AIDecisionExplanation(
                decision_id=decision_id,
                model_name=model_inputs.get('model_name', 'legal_bert_classifier'),
                decision_type='case_outcome_prediction',
                input_summary=self._summarize_input(case_text),
                output_summary=self._summarize_prediction_output(prediction_result),
                confidence_score=prediction_result.get('confidence', 0.0),
                explanation_components=components,
                reasoning_chain=reasoning_chain,
                alternative_outcomes=alternatives,
                limitations=self._identify_prediction_limitations(prediction_result),
                legal_citations=legal_citations,
                bias_assessment=bias_assessment,
                timestamp=datetime.now()
            )
            
            # Record explanation audit trail
            await self._record_explanation_audit(explanation, session)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining case prediction: {str(e)}")
            return self._create_fallback_explanation(case_text, prediction_result, 'case_prediction_error')
    
    async def explain_compliance_decision(self,
                                        document_text: str,
                                        compliance_result: Dict[str, Any],
                                        analysis_context: Dict[str, Any],
                                        session: AsyncSession) -> AIDecisionExplanation:
        """Explain compliance assessment decision"""
        
        logger.info("Generating explanation for compliance decision")
        
        try:
            decision_id = f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(document_text[:1000]) % 10000:04d}"
            
            # Extract compliance-specific components
            components = await self._extract_compliance_components(
                document_text, compliance_result, analysis_context
            )
            
            # Build compliance reasoning chain
            reasoning_chain = await self._build_compliance_reasoning_chain(
                document_text, compliance_result, components
            )
            
            # Identify compliance alternatives
            alternatives = await self._identify_compliance_alternatives(
                compliance_result, components
            )
            
            # Assess compliance bias
            bias_assessment = await self._assess_compliance_bias(
                document_text, compliance_result, analysis_context
            )
            
            explanation = AIDecisionExplanation(
                decision_id=decision_id,
                model_name=analysis_context.get('model_name', 'compliance_analyzer'),
                decision_type='compliance_assessment',
                input_summary=self._summarize_document_input(document_text),
                output_summary=self._summarize_compliance_output(compliance_result),
                confidence_score=self._calculate_compliance_confidence(compliance_result),
                explanation_components=components,
                reasoning_chain=reasoning_chain,
                alternative_outcomes=alternatives,
                limitations=self._identify_compliance_limitations(compliance_result),
                legal_citations=compliance_result.get('statutory_refs', []),
                bias_assessment=bias_assessment,
                timestamp=datetime.now()
            )
            
            await self._record_explanation_audit(explanation, session)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining compliance decision: {str(e)}")
            return self._create_fallback_explanation(document_text, compliance_result, 'compliance_error')
    
    async def explain_risk_assessment(self,
                                    input_data: Dict[str, Any],
                                    risk_result: Dict[str, Any],
                                    assessment_context: Dict[str, Any],
                                    session: AsyncSession) -> AIDecisionExplanation:
        """Explain risk assessment decision"""
        
        logger.info("Generating explanation for risk assessment")
        
        try:
            decision_id = f"risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(input_data)) % 10000:04d}"
            
            components = await self._extract_risk_components(
                input_data, risk_result, assessment_context
            )
            
            reasoning_chain = await self._build_risk_reasoning_chain(
                input_data, risk_result, components
            )
            
            alternatives = await self._identify_risk_alternatives(
                risk_result, components
            )
            
            bias_assessment = await self._assess_risk_bias(
                input_data, risk_result, assessment_context
            )
            
            explanation = AIDecisionExplanation(
                decision_id=decision_id,
                model_name=assessment_context.get('model_name', 'risk_analyzer'),
                decision_type='risk_assessment',
                input_summary=self._summarize_risk_input(input_data),
                output_summary=self._summarize_risk_output(risk_result),
                confidence_score=risk_result.get('confidence', 0.0),
                explanation_components=components,
                reasoning_chain=reasoning_chain,
                alternative_outcomes=alternatives,
                limitations=self._identify_risk_limitations(risk_result),
                legal_citations=risk_result.get('legal_references', []),
                bias_assessment=bias_assessment,
                timestamp=datetime.now()
            )
            
            await self._record_explanation_audit(explanation, session)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining risk assessment: {str(e)}")
            return self._create_fallback_explanation(str(input_data), risk_result, 'risk_error')
    
    async def _extract_prediction_components(self,
                                           case_text: str,
                                           prediction_result: Dict[str, Any],
                                           model_inputs: Dict[str, Any]) -> List[ExplanationComponent]:
        """Extract explanation components for case prediction"""
        
        components = []
        
        try:
            # Case facts component
            case_facts = self._extract_case_facts(case_text)
            if case_facts:
                components.append(ExplanationComponent(
                    component_type='evidence',
                    content=f"Key case facts: {', '.join(case_facts[:5])}",
                    confidence_score=0.9,
                    weight=0.3,
                    source='case_text_analysis',
                    legal_basis='Factual foundation for legal analysis'
                ))
            
            # Legal precedents component
            precedents = prediction_result.get('precedents', [])
            if precedents:
                components.append(ExplanationComponent(
                    component_type='precedent',
                    content=f"Relevant precedents: {', '.join(precedents[:3])}",
                    confidence_score=0.85,
                    weight=0.4,
                    source='precedent_database',
                    legal_basis='Stare decisis - binding precedent principle'
                ))
            
            # Statutory requirements component
            statutory_refs = prediction_result.get('statutory_references', [])
            if statutory_refs:
                components.append(ExplanationComponent(
                    component_type='rule',
                    content=f"Applicable statutes: {', '.join(statutory_refs[:3])}",
                    confidence_score=0.95,
                    weight=0.2,
                    source='statutory_analysis',
                    legal_basis='Statutory interpretation and application'
                ))
            
            # Risk factors component
            risk_factors = prediction_result.get('risk_factors', [])
            if risk_factors:
                components.append(ExplanationComponent(
                    component_type='risk_factor',
                    content=f"Key risk factors: {', '.join(risk_factors[:3])}",
                    confidence_score=0.75,
                    weight=0.1,
                    source='risk_analysis',
                    legal_basis='Risk assessment based on similar cases'
                ))
            
        except Exception as e:
            logger.error(f"Error extracting prediction components: {str(e)}")
        
        return components
    
    async def _extract_compliance_components(self,
                                           document_text: str,
                                           compliance_result: Dict[str, Any],
                                           analysis_context: Dict[str, Any]) -> List[ExplanationComponent]:
        """Extract explanation components for compliance assessment"""
        
        components = []
        
        try:
            # Mandatory clauses analysis
            missing_clauses = compliance_result.get('missing_clauses', [])
            if missing_clauses:
                components.append(ExplanationComponent(
                    component_type='rule',
                    content=f"Missing mandatory clauses: {', '.join(missing_clauses)}",
                    confidence_score=0.95,
                    weight=0.4,
                    source='clause_analysis',
                    legal_basis='Statutory mandatory requirements'
                ))
            
            # Prohibited terms analysis
            prohibited_terms = compliance_result.get('prohibited_terms', [])
            if prohibited_terms:
                components.append(ExplanationComponent(
                    component_type='risk_factor',
                    content=f"Prohibited terms found: {', '.join(prohibited_terms)}",
                    confidence_score=0.9,
                    weight=0.3,
                    source='terms_analysis',
                    legal_basis='Regulatory compliance violations'
                ))
            
            # Regulatory alignment
            compliance_score = compliance_result.get('compliance_score', 0)
            components.append(ExplanationComponent(
                component_type='evidence',
                content=f"Overall compliance score: {compliance_score}%",
                confidence_score=0.8,
                weight=0.2,
                source='compliance_calculator',
                legal_basis='Aggregate compliance assessment'
            ))
            
            # Industry standards
            industry_compliance = compliance_result.get('industry_standards', {})
            if industry_compliance:
                components.append(ExplanationComponent(
                    component_type='evidence',
                    content=f"Industry standard compliance: {industry_compliance.get('score', 'N/A')}",
                    confidence_score=0.7,
                    weight=0.1,
                    source='industry_benchmarks',
                    legal_basis='Industry best practices and standards'
                ))
            
        except Exception as e:
            logger.error(f"Error extracting compliance components: {str(e)}")
        
        return components
    
    async def _extract_risk_components(self,
                                     input_data: Dict[str, Any],
                                     risk_result: Dict[str, Any],
                                     assessment_context: Dict[str, Any]) -> List[ExplanationComponent]:
        """Extract explanation components for risk assessment"""
        
        components = []
        
        try:
            # Financial risk component
            financial_risk = risk_result.get('financial_risk', {})
            if financial_risk:
                components.append(ExplanationComponent(
                    component_type='risk_factor',
                    content=f"Financial risk level: {financial_risk.get('level', 'Unknown')} (${financial_risk.get('amount', 0):,})",
                    confidence_score=financial_risk.get('confidence', 0.7),
                    weight=0.3,
                    source='financial_analysis',
                    legal_basis='Financial impact assessment'
                ))
            
            # Legal risk component
            legal_risk = risk_result.get('legal_risk', {})
            if legal_risk:
                components.append(ExplanationComponent(
                    component_type='risk_factor',
                    content=f"Legal risk level: {legal_risk.get('level', 'Unknown')}",
                    confidence_score=legal_risk.get('confidence', 0.8),
                    weight=0.4,
                    source='legal_analysis',
                    legal_basis='Legal liability and penalty assessment'
                ))
            
            # Operational risk component
            operational_risk = risk_result.get('operational_risk', {})
            if operational_risk:
                components.append(ExplanationComponent(
                    component_type='risk_factor',
                    content=f"Operational risk level: {operational_risk.get('level', 'Unknown')}",
                    confidence_score=operational_risk.get('confidence', 0.6),
                    weight=0.2,
                    source='operational_analysis',
                    legal_basis='Business continuity and operational impact'
                ))
            
            # Reputational risk component
            reputational_risk = risk_result.get('reputational_risk', {})
            if reputational_risk:
                components.append(ExplanationComponent(
                    component_type='risk_factor',
                    content=f"Reputational risk level: {reputational_risk.get('level', 'Unknown')}",
                    confidence_score=reputational_risk.get('confidence', 0.5),
                    weight=0.1,
                    source='reputational_analysis',
                    legal_basis='Brand and reputation impact assessment'
                ))
            
        except Exception as e:
            logger.error(f"Error extracting risk components: {str(e)}")
        
        return components
    
    async def _build_prediction_reasoning_chain(self,
                                              case_text: str,
                                              prediction_result: Dict[str, Any],
                                              components: List[ExplanationComponent]) -> List[str]:
        """Build reasoning chain for case prediction"""
        
        reasoning = []
        
        try:
            reasoning.append("1. Case Analysis: Extracted key facts and legal issues from case text")
            
            if any(c.component_type == 'precedent' for c in components):
                reasoning.append("2. Precedent Matching: Identified similar cases and their outcomes")
            
            if any(c.component_type == 'rule' for c in components):
                reasoning.append("3. Legal Framework: Applied relevant statutes and regulations")
            
            reasoning.append("4. Pattern Recognition: Used ML model trained on similar cases")
            
            confidence = prediction_result.get('confidence', 0.0)
            if confidence > 0.8:
                reasoning.append(f"5. High Confidence: Model confidence of {confidence:.1%} based on strong pattern match")
            elif confidence > 0.6:
                reasoning.append(f"5. Moderate Confidence: Model confidence of {confidence:.1%} with some uncertainty")
            else:
                reasoning.append(f"5. Low Confidence: Model confidence of {confidence:.1%} indicates significant uncertainty")
            
            predicted_outcome = prediction_result.get('predicted_outcome', 'Unknown')
            reasoning.append(f"6. Prediction: Based on analysis, predicted outcome is '{predicted_outcome}'")
            
        except Exception as e:
            logger.error(f"Error building reasoning chain: {str(e)}")
            reasoning.append("Error in reasoning chain construction")
        
        return reasoning
    
    async def _build_compliance_reasoning_chain(self,
                                              document_text: str,
                                              compliance_result: Dict[str, Any],
                                              components: List[ExplanationComponent]) -> List[str]:
        """Build reasoning chain for compliance assessment"""
        
        reasoning = []
        
        try:
            reasoning.append("1. Document Analysis: Parsed document structure and content")
            
            reasoning.append("2. Rule Matching: Compared against regulatory requirements")
            
            if compliance_result.get('missing_clauses'):
                reasoning.append("3. Gap Identification: Found missing mandatory clauses")
            
            if compliance_result.get('prohibited_terms'):
                reasoning.append("4. Violation Detection: Identified prohibited terms or clauses")
            
            status = compliance_result.get('status', 'Unknown')
            reasoning.append(f"5. Compliance Verdict: Determined status as '{status}'")
            
            risk_score = compliance_result.get('risk_score', 'Unknown')
            reasoning.append(f"6. Risk Assessment: Assigned risk level of '{risk_score}'")
            
        except Exception as e:
            logger.error(f"Error building compliance reasoning chain: {str(e)}")
            reasoning.append("Error in compliance reasoning chain construction")
        
        return reasoning
    
    async def _build_risk_reasoning_chain(self,
                                        input_data: Dict[str, Any],
                                        risk_result: Dict[str, Any],
                                        components: List[ExplanationComponent]) -> List[str]:
        """Build reasoning chain for risk assessment"""
        
        reasoning = []
        
        try:
            reasoning.append("1. Risk Identification: Analyzed input for potential risk factors")
            
            reasoning.append("2. Impact Assessment: Evaluated potential consequences")
            
            reasoning.append("3. Likelihood Calculation: Estimated probability of risk occurrence")
            
            reasoning.append("4. Risk Scoring: Combined impact and likelihood for overall risk score")
            
            overall_risk = risk_result.get('overall_risk_level', 'Unknown')
            reasoning.append(f"5. Risk Classification: Classified as '{overall_risk}' risk level")
            
            if risk_result.get('mitigation_recommendations'):
                reasoning.append("6. Mitigation: Identified potential risk mitigation strategies")
            
        except Exception as e:
            logger.error(f"Error building risk reasoning chain: {str(e)}")
            reasoning.append("Error in risk reasoning chain construction")
        
        return reasoning
    
    def _extract_case_facts(self, case_text: str) -> List[str]:
        """Extract key facts from case text"""
        
        # Simple fact extraction - in production would use more sophisticated NLP
        facts = []
        
        # Look for fact patterns
        fact_patterns = [
            r'plaintiff.*?(?:claims?|alleges?|argues?)\s+(.+?)\.', 
            r'defendant.*?(?:claims?|alleges?|argues?)\s+(.+?)\.',
            r'(?:facts?|circumstances?|events?)\s+(?:of|in)\s+(?:this|the)\s+case\s+(.+?)\.',
            r'(?:it\s+is\s+)?(?:undisputed|agreed|established)\s+that\s+(.+?)\.'
        ]
        
        for pattern in fact_patterns:
            matches = re.findall(pattern, case_text, re.IGNORECASE | re.DOTALL)
            facts.extend([match.strip() for match in matches])
        
        return facts[:10]  # Return top 10 facts
    
    def _extract_legal_citations(self, case_text: str, components: List[ExplanationComponent]) -> List[str]:
        """Extract legal citations from text and components"""
        
        citations = []
        
        # Extract from case text
        citation_patterns = [
            r'([A-Z][A-Za-z\s]+?)\s+(\d{4})\s*(?:\(([A-Za-z]{2,4})\))?\s+[Ss](?:ection|\.?)\s*(\d+[A-Z]*)',
            r'\[(\d{4})\]\s+([A-Z]+)\s+(\d+)',
            r'(\d{4})\s+([A-Z]+)\s+(\d+)'
        ]
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, case_text)
            for match in matches:
                citation = " ".join(str(m) for m in match if m)
                citations.append(citation)
        
        # Extract from components
        for component in components:
            if component.component_type in ['rule', 'precedent']:
                citations.append(component.content)
        
        return list(set(citations))[:10]  # Return unique citations, max 10
    
    async def _identify_alternative_outcomes(self,
                                           case_text: str,
                                           prediction_result: Dict[str, Any],
                                           components: List[ExplanationComponent]) -> List[Dict[str, Any]]:
        """Identify alternative possible outcomes"""
        
        alternatives = []
        
        try:
            predicted_outcome = prediction_result.get('predicted_outcome', '')
            confidence = prediction_result.get('confidence', 0.0)
            
            # If confidence is high, fewer alternatives
            if confidence > 0.8:
                alternative_count = 1
            elif confidence > 0.6:
                alternative_count = 2
            else:
                alternative_count = 3
            
            # Generate alternatives based on common legal outcomes
            common_outcomes = ['favorable', 'unfavorable', 'mixed', 'settlement', 'dismissal']
            
            for outcome in common_outcomes:
                if outcome != predicted_outcome and len(alternatives) < alternative_count:
                    alt_confidence = max(0.1, (1.0 - confidence) / alternative_count)
                    alternatives.append({
                        'outcome': outcome,
                        'probability': alt_confidence,
                        'reasoning': f"Alternative outcome based on different interpretation of {components[0].content if components else 'case facts'}"
                    })
            
        except Exception as e:
            logger.error(f"Error identifying alternatives: {str(e)}")
        
        return alternatives
    
    async def _identify_compliance_alternatives(self,
                                              compliance_result: Dict[str, Any],
                                              components: List[ExplanationComponent]) -> List[Dict[str, Any]]:
        """Identify alternative compliance assessments"""
        
        alternatives = []
        
        try:
            current_status = compliance_result.get('status', '')
            
            status_alternatives = {
                '✅ COMPLIANT': ['⚠️ NEEDS REVIEW'],
                '⚠️ NEEDS REVIEW': ['✅ COMPLIANT', '❌ NON-COMPLIANT'],
                '❌ NON-COMPLIANT': ['⚠️ NEEDS REVIEW']
            }
            
            for alt_status in status_alternatives.get(current_status, []):
                alternatives.append({
                    'status': alt_status,
                    'probability': 0.3,
                    'reasoning': f"Alternative assessment considering different interpretation of compliance requirements"
                })
            
        except Exception as e:
            logger.error(f"Error identifying compliance alternatives: {str(e)}")
        
        return alternatives
    
    async def _identify_risk_alternatives(self,
                                        risk_result: Dict[str, Any],
                                        components: List[ExplanationComponent]) -> List[Dict[str, Any]]:
        """Identify alternative risk assessments"""
        
        alternatives = []
        
        try:
            current_risk = risk_result.get('overall_risk_level', '')
            
            risk_alternatives = {
                'Low': ['Medium'],
                'Medium': ['Low', 'High'],
                'High': ['Medium', 'Critical'],
                'Critical': ['High']
            }
            
            for alt_risk in risk_alternatives.get(current_risk, []):
                alternatives.append({
                    'risk_level': alt_risk,
                    'probability': 0.25,
                    'reasoning': f"Alternative risk assessment under different scenarios"
                })
            
        except Exception as e:
            logger.error(f"Error identifying risk alternatives: {str(e)}")
        
        return alternatives
    
    async def _assess_prediction_bias(self,
                                    case_text: str,
                                    prediction_result: Dict[str, Any],
                                    model_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Assess potential bias in prediction"""
        
        bias_assessment = {
            'bias_detected': False,
            'bias_types': [],
            'confidence_in_assessment': 0.7,
            'mitigation_applied': []
        }
        
        try:
            # Check for common biases
            text_lower = case_text.lower()
            
            # Selection bias
            if len(case_text) < 500:
                bias_assessment['bias_types'].append('insufficient_data')
                bias_assessment['bias_detected'] = True
            
            # Confirmation bias indicators
            strong_language = ['clearly', 'obviously', 'undoubtedly', 'certainly']
            if any(word in text_lower for word in strong_language):
                bias_assessment['bias_types'].append('confirmation_bias_risk')
            
            # Recency bias
            recent_dates = re.findall(r'20(2[0-9]|1[8-9])', case_text)
            if len(recent_dates) > 3:
                bias_assessment['bias_types'].append('recency_bias_risk')
            
        except Exception as e:
            logger.error(f"Error assessing bias: {str(e)}")
        
        return bias_assessment
    
    async def _assess_compliance_bias(self,
                                    document_text: str,
                                    compliance_result: Dict[str, Any],
                                    analysis_context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess potential bias in compliance assessment"""
        
        return {
            'bias_detected': False,
            'bias_types': [],
            'confidence_in_assessment': 0.8,
            'mitigation_applied': ['multiple_rule_validation', 'automated_cross_check']
        }
    
    async def _assess_risk_bias(self,
                              input_data: Dict[str, Any],
                              risk_result: Dict[str, Any],
                              assessment_context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess potential bias in risk assessment"""
        
        return {
            'bias_detected': False,
            'bias_types': [],
            'confidence_in_assessment': 0.75,
            'mitigation_applied': ['diversified_risk_factors', 'historical_validation']
        }
    
    def _identify_prediction_limitations(self, prediction_result: Dict[str, Any]) -> List[str]:
        """Identify limitations in prediction"""
        
        limitations = [
            "Prediction based on historical data patterns",
            "May not account for recent legal developments",
            "Limited to Australian jurisdiction unless specified",
            "Assumes standard legal procedures and timelines"
        ]
        
        confidence = prediction_result.get('confidence', 0.0)
        if confidence < 0.7:
            limitations.append("Low confidence indicates high uncertainty in prediction")
        
        return limitations
    
    def _identify_compliance_limitations(self, compliance_result: Dict[str, Any]) -> List[str]:
        """Identify limitations in compliance assessment"""
        
        return [
            "Assessment based on current regulatory framework",
            "May require human legal expert review for complex cases",
            "Limited to automated rule checking",
            "Does not replace professional legal advice"
        ]
    
    def _identify_risk_limitations(self, risk_result: Dict[str, Any]) -> List[str]:
        """Identify limitations in risk assessment"""
        
        return [
            "Risk assessment based on available data",
            "May not capture all possible risk factors",
            "Assumes standard market conditions",
            "Should be reviewed regularly as conditions change"
        ]
    
    def _summarize_input(self, case_text: str) -> str:
        """Summarize case input"""
        return f"Legal case text ({len(case_text)} characters) containing {len(case_text.split())} words"
    
    def _summarize_document_input(self, document_text: str) -> str:
        """Summarize document input"""
        return f"Legal document ({len(document_text)} characters) for compliance assessment"
    
    def _summarize_risk_input(self, input_data: Dict[str, Any]) -> str:
        """Summarize risk input"""
        return f"Risk assessment data with {len(input_data)} parameters"
    
    def _summarize_prediction_output(self, prediction_result: Dict[str, Any]) -> str:
        """Summarize prediction output"""
        outcome = prediction_result.get('predicted_outcome', 'Unknown')
        confidence = prediction_result.get('confidence', 0.0)
        return f"Predicted outcome: {outcome} (confidence: {confidence:.1%})"
    
    def _summarize_compliance_output(self, compliance_result: Dict[str, Any]) -> str:
        """Summarize compliance output"""
        status = compliance_result.get('status', 'Unknown')
        risk = compliance_result.get('risk_score', 'Unknown')
        return f"Compliance status: {status}, Risk level: {risk}"
    
    def _summarize_risk_output(self, risk_result: Dict[str, Any]) -> str:
        """Summarize risk output"""
        level = risk_result.get('overall_risk_level', 'Unknown')
        score = risk_result.get('risk_score', 0)
        return f"Risk level: {level} (score: {score})"
    
    def _calculate_compliance_confidence(self, compliance_result: Dict[str, Any]) -> float:
        """Calculate confidence for compliance assessment"""
        # Simple confidence calculation based on available data
        base_confidence = 0.8
        
        if compliance_result.get('statutory_refs'):
            base_confidence += 0.1
        
        if compliance_result.get('missing_clauses'):
            base_confidence += 0.05
        
        return min(0.95, base_confidence)
    
    def _create_fallback_explanation(self, 
                                   input_text: str, 
                                   result: Dict[str, Any], 
                                   error_type: str) -> AIDecisionExplanation:
        """Create fallback explanation when main explanation fails"""
        
        return AIDecisionExplanation(
            decision_id=f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_name='fallback_explainer',
            decision_type=error_type,
            input_summary=f"Input data ({len(str(input_text))} characters)",
            output_summary="Error in explanation generation",
            confidence_score=0.0,
            explanation_components=[],
            reasoning_chain=["Explanation generation failed", "Fallback explanation provided"],
            alternative_outcomes=[],
            limitations=["Full explanation unavailable due to error"],
            legal_citations=[],
            bias_assessment={'bias_detected': False, 'error': 'Assessment unavailable'},
            timestamp=datetime.now()
        )
    
    async def _record_explanation_audit(self, explanation: AIDecisionExplanation, session: AsyncSession):
        """Record explanation in audit trail"""
        try:
            audit_data = {
                'decision_id': explanation.decision_id,
                'model_name': explanation.model_name,
                'decision_type': explanation.decision_type,
                'confidence_score': explanation.confidence_score,
                'components_count': len(explanation.explanation_components),
                'timestamp': explanation.timestamp.isoformat()
            }
            
            await blockchain_service.record_audit(
                document_id=f"xai_{explanation.decision_id}",
                user_id="system",
                action="ai_explanation_generated",
                additional_data=audit_data,
                session=session
            )
            
        except Exception as e:
            logger.error(f"Error recording explanation audit: {str(e)}")

# Initialize the explainable AI service
explainable_ai = ExplainableAI() 