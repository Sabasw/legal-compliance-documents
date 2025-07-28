"""
Industry-Specific Compliance Modules
Healthcare, Corporate Governance, Real Estate compliance validators
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
import re

logger = logging.getLogger(__name__)

@dataclass
class ComplianceViolation:
    """Compliance violation details"""
    violation_type: str
    severity: str
    description: str
    section: str
    recommendation: str
    legal_reference: str

@dataclass
class IndustryComplianceResult:
    """Industry compliance assessment result"""
    industry: str
    overall_status: str
    compliance_score: float
    violations: List[ComplianceViolation]
    recommendations: List[str]
    regulatory_requirements: List[str]

class HealthcareComplianceValidator:
    """Healthcare industry compliance validator"""
    
    def __init__(self):
        self.hipaa_requirements = [
            "patient_privacy_protection",
            "data_encryption",
            "access_controls",
            "audit_logging",
            "breach_notification",
            "business_associate_agreements"
        ]
        
        self.privacy_keywords = [
            "patient", "medical record", "health information", 
            "phi", "protected health information", "hipaa"
        ]
    
    async def validate_healthcare_compliance(self, 
                                           document_text: str,
                                           document_type: str = "general") -> IndustryComplianceResult:
        """Validate healthcare document compliance"""
        try:
            violations = []
            
            # Check HIPAA compliance
            hipaa_violations = await self._check_hipaa_compliance(document_text)
            violations.extend(hipaa_violations)
            
            # Check medical data handling
            data_violations = await self._check_medical_data_handling(document_text)
            violations.extend(data_violations)
            
            # Check business associate requirements
            if document_type == "business_associate_agreement":
                ba_violations = await self._check_business_associate_requirements(document_text)
                violations.extend(ba_violations)
            
            # Calculate compliance score
            compliance_score = self._calculate_healthcare_score(violations)
            
            # Generate recommendations
            recommendations = self._generate_healthcare_recommendations(violations)
            
            return IndustryComplianceResult(
                industry="healthcare",
                overall_status=self._determine_status(compliance_score),
                compliance_score=compliance_score,
                violations=violations,
                recommendations=recommendations,
                regulatory_requirements=self.hipaa_requirements
            )
            
        except Exception as e:
            logger.error(f"Error in healthcare compliance validation: {str(e)}")
            raise
    
    async def _check_hipaa_compliance(self, text: str) -> List[ComplianceViolation]:
        """Check HIPAA compliance requirements"""
        violations = []
        text_lower = text.lower()
        
        # Check for privacy policy
        if any(keyword in text_lower for keyword in self.privacy_keywords):
            if "privacy policy" not in text_lower:
                violations.append(ComplianceViolation(
                    violation_type="missing_privacy_policy",
                    severity="high",
                    description="Document mentions health information but lacks privacy policy reference",
                    section="Privacy Requirements",
                    recommendation="Include privacy policy statement and HIPAA compliance notice",
                    legal_reference="45 CFR 164.520"
                ))
        
        # Check for encryption requirements
        if "data" in text_lower and "encrypt" not in text_lower:
            violations.append(ComplianceViolation(
                violation_type="missing_encryption",
                severity="critical",
                description="Data handling mentioned without encryption requirements",
                section="Security Requirements",
                recommendation="Specify data encryption requirements for PHI",
                legal_reference="45 CFR 164.312(a)(2)(iv)"
            ))
        
        # Check for PII/PHI exposure - SSN patterns
        ssn_pattern = r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'
        if re.search(ssn_pattern, text):
            violations.append(ComplianceViolation(
                violation_type="exposed_ssn",
                severity="critical",
                description="Social Security Number detected without proper protection",
                section="PHI Protection",
                recommendation="Remove or encrypt SSN data, implement access controls",
                legal_reference="45 CFR 164.514"
            ))
        
        # Check for employee/staff data handling
        if "employee" in text_lower and ("ssn" in text_lower or "social security" in text_lower):
            violations.append(ComplianceViolation(
                violation_type="employee_phi_exposure",
                severity="high",
                description="Employee personal information exposed without proper safeguards",
                section="Workforce Security",
                recommendation="Implement workforce security procedures and access controls",
                legal_reference="45 CFR 164.308(a)(3)"
            ))
        
        return violations
    
    async def _check_medical_data_handling(self, text: str) -> List[ComplianceViolation]:
        """Check medical data handling compliance"""
        violations = []
        text_lower = text.lower()
        
        # Check for minimum necessary standard
        if "medical record" in text_lower and "minimum necessary" not in text_lower:
            violations.append(ComplianceViolation(
                violation_type="missing_minimum_necessary",
                severity="medium",
                description="Medical records mentioned without minimum necessary standard",
                section="Data Access",
                recommendation="Include minimum necessary access provisions",
                legal_reference="45 CFR 164.502(b)"
            ))
        
        # Check for governance issues that might affect healthcare (cross-industry concerns)
        if "director" in text_lower and ("disclose" in text_lower or "conflict" in text_lower):
            violations.append(ComplianceViolation(
                violation_type="governance_conflict_disclosure",
                severity="high",
                description="Director conflict disclosure requirements mentioned without proper framework",
                section="Governance and Compliance",
                recommendation="Implement conflict of interest disclosure procedures",
                legal_reference="Healthcare governance best practices"
            ))
        
        return violations
    
    async def _check_business_associate_requirements(self, text: str) -> List[ComplianceViolation]:
        """Check business associate agreement requirements"""
        violations = []
        
        required_clauses = [
            "permitted uses", "required uses", "safeguards", 
            "reporting", "return or destruction", "authorized agents"
        ]
        
        for clause in required_clauses:
            if clause not in text.lower():
                violations.append(ComplianceViolation(
                    violation_type=f"missing_{clause.replace(' ', '_')}",
                    severity="high",
                    description=f"Missing required clause: {clause}",
                    section="Business Associate Requirements",
                    recommendation=f"Add {clause} clause to agreement",
                    legal_reference="45 CFR 164.504(e)"
                ))
        
        return violations
    
    def _calculate_healthcare_score(self, violations: List[ComplianceViolation]) -> float:
        """Calculate healthcare compliance score"""
        if not violations:
            return 100.0
        
        penalty_map = {"critical": 25, "high": 15, "medium": 10, "low": 5}
        total_penalty = sum(penalty_map.get(v.severity, 5) for v in violations)
        
        return max(0.0, 100.0 - total_penalty)
    
    def _generate_healthcare_recommendations(self, violations: List[ComplianceViolation]) -> List[str]:
        """Generate healthcare compliance recommendations"""
        recommendations = []
        
        if violations:
            recommendations.extend([v.recommendation for v in violations])
        
        recommendations.extend([
            "Conduct annual HIPAA risk assessment",
            "Implement employee training program",
            "Establish incident response procedures"
        ])
        
        return list(set(recommendations))
    
    def _determine_status(self, score: float) -> str:
        """Determine overall compliance status"""
        if score >= 90:
            return "✅ COMPLIANT"
        elif score >= 70:
            return "⚠️ NEEDS REVIEW"
        else:
            return "❌ NON-COMPLIANT"

class CorporateGovernanceValidator:
    """Corporate governance compliance validator"""
    
    def __init__(self):
        self.asx_requirements = [
            "board_charter", "audit_committee", "risk_management",
            "remuneration_policy", "diversity_policy", "continuous_disclosure"
        ]
        
        self.governance_keywords = [
            "board", "director", "shareholder", "governance",
            "audit committee", "risk management", "compliance"
        ]
    
    async def validate_corporate_governance(self, 
                                          document_text: str,
                                          company_type: str = "public") -> IndustryComplianceResult:
        """Validate corporate governance compliance"""
        try:
            violations = []
            
            # Check ASX Corporate Governance Principles
            if company_type == "public":
                asx_violations = await self._check_asx_principles(document_text)
                violations.extend(asx_violations)
            
            # Check Corporations Act compliance
            corp_violations = await self._check_corporations_act(document_text)
            violations.extend(corp_violations)
            
            # Check director duties
            director_violations = await self._check_director_duties(document_text)
            violations.extend(director_violations)
            
            compliance_score = self._calculate_governance_score(violations)
            recommendations = self._generate_governance_recommendations(violations)
            
            return IndustryComplianceResult(
                industry="corporate_governance",
                overall_status=self._determine_status(compliance_score),
                compliance_score=compliance_score,
                violations=violations,
                recommendations=recommendations,
                regulatory_requirements=self.asx_requirements
            )
            
        except Exception as e:
            logger.error(f"Error in corporate governance validation: {str(e)}")
            raise
    
    async def _check_asx_principles(self, text: str) -> List[ComplianceViolation]:
        """Check ASX Corporate Governance Principles"""
        violations = []
        text_lower = text.lower()
        
        # Principle 1: Lay solid foundations for management and oversight
        if "board charter" not in text_lower and "board" in text_lower:
            violations.append(ComplianceViolation(
                violation_type="missing_board_charter",
                severity="high",
                description="Board mentioned without board charter reference",
                section="Foundation Principles",
                recommendation="Include board charter and role definition",
                legal_reference="ASX Principle 1.1"
            ))
        
        # Principle 4: Safeguard integrity in corporate reporting
        if "audit" in text_lower and "audit committee" not in text_lower:
            violations.append(ComplianceViolation(
                violation_type="missing_audit_committee",
                severity="high",
                description="Audit processes mentioned without audit committee structure",
                section="Corporate Reporting",
                recommendation="Establish audit committee charter and responsibilities",
                legal_reference="ASX Principle 4.1"
            ))
        
        return violations
    
    async def _check_corporations_act(self, text: str) -> List[ComplianceViolation]:
        """Check Corporations Act 2001 compliance"""
        violations = []
        
        # Check director duties
        if "director" in text.lower():
            if "duty of care" not in text.lower() and "fiduciary duty" not in text.lower():
                violations.append(ComplianceViolation(
                    violation_type="missing_director_duties",
                    severity="critical",
                    description="Director roles mentioned without duty obligations",
                    section="Director Duties",
                    recommendation="Include director duty of care and fiduciary obligations",
                    legal_reference="Corporations Act s180-184"
                ))
        
        return violations
    
    async def _check_director_duties(self, text: str) -> List[ComplianceViolation]:
        """Check director duties compliance"""
        violations = []
        
        required_duties = [
            "duty of care", "duty of diligence", "good faith",
            "proper purpose", "avoid conflicts"
        ]
        
        if "director" in text.lower():
            missing_duties = [duty for duty in required_duties if duty not in text.lower()]
            
            for duty in missing_duties:
                violations.append(ComplianceViolation(
                    violation_type=f"missing_{duty.replace(' ', '_')}",
                    severity="high",
                    description=f"Missing director {duty} reference",
                    section="Director Obligations",
                    recommendation=f"Include {duty} provisions",
                    legal_reference="Corporations Act s180-183"
                ))
        
        return violations
    
    def _calculate_governance_score(self, violations: List[ComplianceViolation]) -> float:
        """Calculate governance compliance score"""
        if not violations:
            return 100.0
        
        penalty_map = {"critical": 30, "high": 20, "medium": 10, "low": 5}
        total_penalty = sum(penalty_map.get(v.severity, 5) for v in violations)
        
        return max(0.0, 100.0 - total_penalty)
    
    def _generate_governance_recommendations(self, violations: List[ComplianceViolation]) -> List[str]:
        """Generate governance recommendations"""
        recommendations = []
        
        if violations:
            recommendations.extend([v.recommendation for v in violations])
        
        recommendations.extend([
            "Implement board effectiveness review",
            "Establish risk management framework",
            "Develop ESG reporting strategy"
        ])
        
        return list(set(recommendations))
    
    def _determine_status(self, score: float) -> str:
        """Determine overall compliance status"""
        if score >= 90:
            return "✅ COMPLIANT"
        elif score >= 70:
            return "⚠️ NEEDS REVIEW"
        else:
            return "❌ NON-COMPLIANT"

class RealEstateComplianceValidator:
    """Real estate contracts compliance validator"""
    
    def __init__(self):
        self.contract_requirements = [
            "property_description", "purchase_price", "settlement_terms",
            "cooling_off_period", "title_details", "disclosure_statement"
        ]
    
    async def validate_real_estate_contract(self, 
                                          document_text: str,
                                          property_type: str = "residential") -> IndustryComplianceResult:
        """Validate real estate contract compliance"""
        try:
            violations = []
            
            # Check mandatory contract terms
            mandatory_violations = await self._check_mandatory_terms(document_text)
            violations.extend(mandatory_violations)
            
            # Check cooling-off provisions
            cooling_off_violations = await self._check_cooling_off_provisions(document_text)
            violations.extend(cooling_off_violations)
            
            # Check disclosure requirements
            disclosure_violations = await self._check_disclosure_requirements(document_text)
            violations.extend(disclosure_violations)
            
            compliance_score = self._calculate_real_estate_score(violations)
            recommendations = self._generate_real_estate_recommendations(violations)
            
            return IndustryComplianceResult(
                industry="real_estate",
                overall_status=self._determine_status(compliance_score),
                compliance_score=compliance_score,
                violations=violations,
                recommendations=recommendations,
                regulatory_requirements=self.contract_requirements
            )
            
        except Exception as e:
            logger.error(f"Error in real estate compliance validation: {str(e)}")
            raise
    
    async def _check_mandatory_terms(self, text: str) -> List[ComplianceViolation]:
        """Check mandatory contract terms"""
        violations = []
        text_lower = text.lower()
        
        # Check property description
        if "property" in text_lower and "description" not in text_lower:
            violations.append(ComplianceViolation(
                violation_type="missing_property_description",
                severity="critical",
                description="Property mentioned without adequate description",
                section="Property Details",
                recommendation="Include detailed property description with address and title details",
                legal_reference="Sale of Land Act"
            ))
        
        # Check purchase price
        price_patterns = [r'\$[\d,]+', r'price.*\$', r'purchase.*price']
        if not any(re.search(pattern, text_lower) for pattern in price_patterns):
            violations.append(ComplianceViolation(
                violation_type="missing_purchase_price",
                severity="critical",
                description="Purchase price not clearly specified",
                section="Financial Terms",
                recommendation="Clearly state purchase price and payment terms",
                legal_reference="Sale of Land Act s9"
            ))
        
        return violations
    
    async def _check_cooling_off_provisions(self, text: str) -> List[ComplianceViolation]:
        """Check cooling-off period provisions"""
        violations = []
        
        if "cooling" not in text.lower() and "residential" in text.lower():
            violations.append(ComplianceViolation(
                violation_type="missing_cooling_off",
                severity="high",
                description="Residential contract missing cooling-off period provision",
                section="Consumer Protection",
                recommendation="Include cooling-off period clause (typically 5 business days)",
                legal_reference="Sale of Land Act s27"
            ))
        
        return violations
    
    async def _check_disclosure_requirements(self, text: str) -> List[ComplianceViolation]:
        """Check disclosure requirements"""
        violations = []
        
        disclosure_items = ["vendor statement", "title search", "building inspection"]
        
        for item in disclosure_items:
            if item not in text.lower():
                violations.append(ComplianceViolation(
                    violation_type=f"missing_{item.replace(' ', '_')}",
                    severity="medium",
                    description=f"Missing {item} disclosure",
                    section="Disclosure Requirements",
                    recommendation=f"Include {item} provision",
                    legal_reference="Sale of Land Act s32"
                ))
        
        return violations
    
    def _calculate_real_estate_score(self, violations: List[ComplianceViolation]) -> float:
        """Calculate real estate compliance score"""
        if not violations:
            return 100.0
        
        penalty_map = {"critical": 25, "high": 15, "medium": 8, "low": 3}
        total_penalty = sum(penalty_map.get(v.severity, 3) for v in violations)
        
        return max(0.0, 100.0 - total_penalty)
    
    def _generate_real_estate_recommendations(self, violations: List[ComplianceViolation]) -> List[str]:
        """Generate real estate recommendations"""
        recommendations = []
        
        if violations:
            recommendations.extend([v.recommendation for v in violations])
        
        recommendations.extend([
            "Review state-specific sale of land legislation",
            "Include standard contract conditions",
            "Verify all mandatory disclosures"
        ])
        
        return list(set(recommendations))
    
    def _determine_status(self, score: float) -> str:
        """Determine overall compliance status"""
        if score >= 90:
            return "✅ COMPLIANT"
        elif score >= 70:
            return "⚠️ NEEDS REVIEW"
        else:
            return "❌ NON-COMPLIANT"

class IndustryComplianceService:
    """Main industry compliance service"""
    
    def __init__(self):
        self.healthcare_validator = HealthcareComplianceValidator()
        self.governance_validator = CorporateGovernanceValidator()
        self.real_estate_validator = RealEstateComplianceValidator()
    
    async def validate_industry_compliance(self,
                                         document_text: str,
                                         industry: str,
                                         document_type: str = "general",
                                         **kwargs) -> IndustryComplianceResult:
        """Validate compliance for specific industry"""
        try:
            if industry == "healthcare":
                return await self.healthcare_validator.validate_healthcare_compliance(
                    document_text, document_type
                )
            elif industry == "corporate_governance":
                company_type = kwargs.get("company_type", "public")
                return await self.governance_validator.validate_corporate_governance(
                    document_text, company_type
                )
            elif industry == "real_estate":
                property_type = kwargs.get("property_type", "residential")
                return await self.real_estate_validator.validate_real_estate_contract(
                    document_text, property_type
                )
            else:
                raise ValueError(f"Unsupported industry: {industry}")
                
        except Exception as e:
            logger.error(f"Error in industry compliance validation: {str(e)}")
            raise
    
    # Direct methods for healthcare compliance (expected by tests) - SYNC version
    def validate_healthcare_compliance(self, 
                                     document_text: str,
                                     document_type: str = "general") -> Dict[str, Any]:
        """Direct healthcare compliance validation (sync for test compatibility)"""
        import asyncio
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                self.healthcare_validator.validate_healthcare_compliance(document_text, document_type)
            )
            return self._convert_to_dict(result)
        except Exception as e:
            logger.error(f"Healthcare compliance validation error: {str(e)}")
            return self._empty_result("healthcare")
    
    # Async version for internal use
    async def validate_healthcare_compliance_async(self, 
                                                 document_text: str,
                                                 document_type: str = "general") -> IndustryComplianceResult:
        """Async healthcare compliance validation"""
        return await self.healthcare_validator.validate_healthcare_compliance(document_text, document_type)
    
    # Direct methods for corporate governance compliance (expected by tests)
    async def validate_corporate_governance(self, 
                                          document_text: str,
                                          company_type: str = "public") -> IndustryComplianceResult:
        """Direct corporate governance compliance validation"""
        return await self.governance_validator.validate_corporate_governance(document_text, company_type)
    
    # Direct methods for real estate compliance (expected by tests)
    async def validate_real_estate_contract(self, 
                                          document_text: str,
                                          property_type: str = "residential") -> IndustryComplianceResult:
        """Direct real estate contract compliance validation"""
        return await self.real_estate_validator.validate_real_estate_contract(document_text, property_type)
    
    # Additional method aliases expected by tests
    def validate_corporate_compliance(self, document_text: str, company_type: str = "public") -> Dict[str, Any]:
        """Sync wrapper for corporate governance compliance (for test compatibility)"""
        import asyncio
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                self.governance_validator.validate_corporate_governance(document_text, company_type)
            )
            return self._convert_to_dict(result)
        except Exception as e:
            logger.error(f"Corporate compliance validation error: {str(e)}")
            return self._empty_result("corporate_governance")
    
    def validate_real_estate_compliance(self, document_text: str, property_type: str = "residential") -> Dict[str, Any]:
        """Sync wrapper for real estate compliance (for test compatibility)"""
        import asyncio
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                self.real_estate_validator.validate_real_estate_contract(document_text, property_type)
            )
            return self._convert_to_dict(result)
        except Exception as e:
            logger.error(f"Real estate compliance validation error: {str(e)}")
            return self._empty_result("real_estate")
    
    def _convert_to_dict(self, result: IndustryComplianceResult) -> Dict[str, Any]:
        """Convert IndustryComplianceResult to dictionary for test compatibility"""
        violations_dict = []
        for violation in result.violations:
            violations_dict.append({
                "violation_type": violation.violation_type,
                "severity": violation.severity,
                "description": violation.description,
                "section": violation.section,
                "recommendation": violation.recommendation,
                "legal_reference": violation.legal_reference
            })
        
        return {
            "industry": result.industry,
            "overall_status": result.overall_status,
            "compliance_score": result.compliance_score,
            "violations": violations_dict,
            "recommendations": result.recommendations,
            "regulatory_requirements": result.regulatory_requirements
        }
    
    def _empty_result(self, industry: str) -> Dict[str, Any]:
        """Return empty result for error cases"""
        return {
            "industry": industry,
            "overall_status": "❌ ERROR",
            "compliance_score": 0.0,
            "violations": [],
            "recommendations": [],
            "regulatory_requirements": []
        }

    # Additional utility methods expected by tests
    async def detect_pii_entities(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII entities in text"""
        pii_patterns = {
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        }
        
        entities = []
        for entity_type, pattern in pii_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "entity_type": entity_type,
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9  # High confidence for regex matches
                })
        
        return entities

# Initialize industry compliance service
industry_compliance_service = IndustryComplianceService() 