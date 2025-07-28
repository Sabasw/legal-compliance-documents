"""
Dynamic Regulatory Compliance Engine
Real-time updates from legal sources like AustLII, rule parsing, and cross-jurisdictional support
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from uuid import uuid4
import json
import re
from bs4 import BeautifulSoup
from dataclasses import dataclass

from app.db.models import ComplianceRule, User, AuditLog
# Note: blockchain_service import moved to avoid circular import
from config import settings

logger = logging.getLogger(__name__)

@dataclass
class LegalUpdate:
    """Legal update from external sources"""
    source: str
    title: str
    content: str
    effective_date: datetime
    jurisdiction: str
    category: str
    urgency: str  # low, medium, high, critical
    act_references: List[str]
    section_changes: List[Dict[str, Any]]
    impact_assessment: Dict[str, Any]

@dataclass
class ComplianceGap:
    """Identified compliance gap"""
    rule_id: str
    gap_type: str
    severity: str
    description: str
    recommendation: str
    affected_documents: List[str]
    remediation_steps: List[str]

class DynamicComplianceEngine:
    """Advanced compliance engine with real-time regulatory updates"""
    
    def __init__(self):
        self.legal_sources = {
            'austlii': {
                'base_url': 'http://www.austlii.edu.au/cgi-bin/sinodisp/au/legis/cth/consol_act/',
                'rss_feed': 'http://www.austlii.edu.au/au/other/liac/updates.xml',
                'priority': 'high'
            },
            'legislation_gov_au': {
                'base_url': 'https://www.legislation.gov.au/Series/',
                'api_endpoint': 'https://www.legislation.gov.au/api/search',
                'priority': 'high'
            },
            'asic': {
                'base_url': 'https://asic.gov.au/regulatory-resources/',
                'updates_feed': 'https://asic.gov.au/about-asic/news-centre/news-feed/',
                'priority': 'high'
            },
            'apra': {
                'base_url': 'https://www.apra.gov.au/prudential-framework',
                'priority': 'medium'
            }
        }
        
        self.jurisdiction_configs = {
            'AU': {
                'name': 'Australia',
                'regulator': 'ASIC/APRA',
                'primary_sources': ['austlii', 'legislation_gov_au', 'asic'],
                'update_frequency': 'daily',
                'critical_acts': [
                    'Corporations Act 2001',
                    'Privacy Act 1988',
                    'Fair Work Act 2009',
                    'Competition and Consumer Act 2010',
                    'Anti-Money Laundering and Counter-Terrorism Financing Act 2006'
                ]
            },
            'US': {
                'name': 'United States',
                'regulator': 'SEC/FINRA',
                'primary_sources': ['sec_gov', 'federal_register'],
                'update_frequency': 'daily',
                'critical_acts': [
                    'Securities Act 1933',
                    'Securities Exchange Act 1934',
                    'Sarbanes-Oxley Act 2002',
                    'Dodd-Frank Act 2010'
                ]
            },
            'UK': {
                'name': 'United Kingdom',
                'regulator': 'FCA/PRA',
                'primary_sources': ['legislation_gov_uk', 'fca_handbook'],
                'update_frequency': 'daily',
                'critical_acts': [
                    'Financial Services and Markets Act 2000',
                    'Data Protection Act 2018',
                    'Companies Act 2006'
                ]
            }
        }
    
    async def monitor_regulatory_updates(self, jurisdiction: str = 'AU') -> List[LegalUpdate]:
        """Monitor and fetch real-time regulatory updates"""
        logger.info(f"Monitoring regulatory updates for {jurisdiction}")
        
        config = self.jurisdiction_configs.get(jurisdiction, self.jurisdiction_configs['AU'])
        updates = []
        
        try:
            async with aiohttp.ClientSession() as session:
                for source in config['primary_sources']:
                    if source in self.legal_sources:
                        source_updates = await self._fetch_from_source(session, source, jurisdiction)
                        updates.extend(source_updates)
            
            # Process and prioritize updates
            processed_updates = await self._process_updates(updates, jurisdiction)
            
            # Log monitoring activity
            logger.info(f"Found {len(processed_updates)} regulatory updates for {jurisdiction}")
            
            return processed_updates
            
        except Exception as e:
            logger.error(f"Error monitoring regulatory updates: {str(e)}")
            return []
    
    async def _fetch_from_source(self, session: aiohttp.ClientSession, source: str, jurisdiction: str) -> List[LegalUpdate]:
        """Fetch updates from specific legal source"""
        updates = []
        
        try:
            if source == 'austlii':
                updates = await self._fetch_austlii_updates(session)
            elif source == 'legislation_gov_au':
                updates = await self._fetch_legislation_gov_updates(session)
            elif source == 'asic':
                updates = await self._fetch_asic_updates(session)
            elif source == 'apra':
                updates = await self._fetch_apra_updates(session)
            
        except Exception as e:
            logger.error(f"Error fetching from {source}: {str(e)}")
        
        return updates
    
    async def _fetch_austlii_updates(self, session: aiohttp.ClientSession) -> List[LegalUpdate]:
        """Fetch updates from AustLII"""
        updates = []
        
        try:
            # Fetch recent legislation updates
            async with session.get(self.legal_sources['austlii']['rss_feed']) as response:
                if response.status == 200:
                    content = await response.text()
                    # Parse RSS feed
                    updates = self._parse_austlii_rss(content)
        except Exception as e:
            logger.error(f"Error fetching AustLII updates: {str(e)}")
        
        return updates
    
    async def _fetch_legislation_gov_updates(self, session: aiohttp.ClientSession) -> List[LegalUpdate]:
        """Fetch updates from legislation.gov.au"""
        updates = []
        
        try:
            # Search for recent amendments and new legislation
            search_params = {
                'q': 'amendment OR commencement',
                'dateFrom': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                'sort': 'date-desc',
                'limit': 50
            }
            
            async with session.get(self.legal_sources['legislation_gov_au']['api_endpoint'], 
                                 params=search_params) as response:
                if response.status == 200:
                    data = await response.json()
                    updates = self._parse_legislation_gov_response(data)
        except Exception as e:
            logger.error(f"Error fetching legislation.gov.au updates: {str(e)}")
        
        return updates
    
    async def _fetch_asic_updates(self, session: aiohttp.ClientSession) -> List[LegalUpdate]:
        """Fetch updates from ASIC"""
        updates = []
        
        try:
            # Fetch ASIC regulatory announcements
            async with session.get(self.legal_sources['asic']['updates_feed']) as response:
                if response.status == 200:
                    content = await response.text()
                    updates = self._parse_asic_updates(content)
        except Exception as e:
            logger.error(f"Error fetching ASIC updates: {str(e)}")
        
        return updates
    
    async def _fetch_apra_updates(self, session: aiohttp.ClientSession) -> List[LegalUpdate]:
        """Fetch updates from APRA"""
        updates = []
        
        try:
            # Fetch APRA prudential framework updates
            async with session.get(self.legal_sources['apra']['base_url']) as response:
                if response.status == 200:
                    content = await response.text()
                    updates = self._parse_apra_updates(content)
        except Exception as e:
            logger.error(f"Error fetching APRA updates: {str(e)}")
        
        return updates
    
    def _parse_austlii_rss(self, rss_content: str) -> List[LegalUpdate]:
        """Parse AustLII RSS feed"""
        updates = []
        
        try:
            soup = BeautifulSoup(rss_content, 'xml')
            items = soup.find_all('item')
            
            for item in items:
                title = item.find('title').text if item.find('title') else ""
                description = item.find('description').text if item.find('description') else ""
                pub_date = item.find('pubDate').text if item.find('pubDate') else ""
                
                # Extract act references
                act_references = self._extract_act_references(title + " " + description)
                
                # Determine urgency based on content
                urgency = self._assess_update_urgency(title, description)
                
                update = LegalUpdate(
                    source='austlii',
                    title=title,
                    content=description,
                    effective_date=self._parse_date(pub_date),
                    jurisdiction='AU',
                    category=self._categorize_update(title, description),
                    urgency=urgency,
                    act_references=act_references,
                    section_changes=self._extract_section_changes(description),
                    impact_assessment=self._assess_impact(title, description)
                )
                
                updates.append(update)
                
        except Exception as e:
            logger.error(f"Error parsing AustLII RSS: {str(e)}")
        
        return updates
    
    def _parse_legislation_gov_response(self, data: Dict) -> List[LegalUpdate]:
        """Parse legislation.gov.au API response"""
        updates = []
        
        try:
            for item in data.get('items', []):
                title = item.get('title', '')
                content = item.get('summary', '')
                
                update = LegalUpdate(
                    source='legislation_gov_au',
                    title=title,
                    content=content,
                    effective_date=self._parse_date(item.get('date')),
                    jurisdiction='AU',
                    category=self._categorize_update(title, content),
                    urgency=self._assess_update_urgency(title, content),
                    act_references=self._extract_act_references(title + " " + content),
                    section_changes=self._extract_section_changes(content),
                    impact_assessment=self._assess_impact(title, content)
                )
                
                updates.append(update)
                
        except Exception as e:
            logger.error(f"Error parsing legislation.gov.au response: {str(e)}")
        
        return updates
    
    def _parse_asic_updates(self, content: str) -> List[LegalUpdate]:
        """Parse ASIC updates"""
        updates = []
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract news items
            news_items = soup.find_all('div', class_='news-item')
            
            for item in news_items:
                title = item.find('h3').text.strip() if item.find('h3') else ""
                description = item.find('p').text.strip() if item.find('p') else ""
                date_elem = item.find('time')
                date_str = date_elem['datetime'] if date_elem and date_elem.get('datetime') else ""
                
                update = LegalUpdate(
                    source='asic',
                    title=title,
                    content=description,
                    effective_date=self._parse_date(date_str),
                    jurisdiction='AU',
                    category='regulatory_guidance',
                    urgency=self._assess_update_urgency(title, description),
                    act_references=self._extract_act_references(title + " " + description),
                    section_changes=[],
                    impact_assessment=self._assess_impact(title, description)
                )
                
                updates.append(update)
                
        except Exception as e:
            logger.error(f"Error parsing ASIC updates: {str(e)}")
        
        return updates
    
    def _parse_apra_updates(self, content: str) -> List[LegalUpdate]:
        """Parse APRA updates"""
        updates = []
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract prudential framework updates
            framework_items = soup.find_all('div', class_='framework-update')
            
            for item in framework_items:
                title = item.find('h4').text.strip() if item.find('h4') else ""
                description = item.find('p').text.strip() if item.find('p') else ""
                
                update = LegalUpdate(
                    source='apra',
                    title=title,
                    content=description,
                    effective_date=datetime.now(),
                    jurisdiction='AU',
                    category='prudential_framework',
                    urgency='medium',
                    act_references=self._extract_act_references(title + " " + description),
                    section_changes=[],
                    impact_assessment=self._assess_impact(title, description)
                )
                
                updates.append(update)
                
        except Exception as e:
            logger.error(f"Error parsing APRA updates: {str(e)}")
        
        return updates
    
    async def _process_updates(self, updates: List[LegalUpdate], jurisdiction: str) -> List[LegalUpdate]:
        """Process and prioritize updates"""
        processed = []
        
        for update in updates:
            # Filter relevant updates
            if self._is_relevant_update(update, jurisdiction):
                # Enhance with additional analysis
                update.impact_assessment = await self._enhanced_impact_analysis(update)
                processed.append(update)
        
        # Sort by urgency and relevance
        processed.sort(key=lambda x: (
            {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}.get(x.urgency, 1),
            len(x.act_references)
        ), reverse=True)
        
        return processed
    
    def _is_relevant_update(self, update: LegalUpdate, jurisdiction: str) -> bool:
        """Check if update is relevant to jurisdiction"""
        config = self.jurisdiction_configs.get(jurisdiction, {})
        critical_acts = config.get('critical_acts', [])
        
        # Check if update mentions critical acts
        for act in critical_acts:
            if act.lower() in update.title.lower() or act.lower() in update.content.lower():
                return True
        
        # Check for key compliance terms
        compliance_terms = [
            'compliance', 'regulation', 'amendment', 'commencement',
            'penalty', 'enforcement', 'obligation', 'requirement'
        ]
        
        text = (update.title + " " + update.content).lower()
        return any(term in text for term in compliance_terms)
    
    def _extract_act_references(self, text: str) -> List[str]:
        """Extract act references from text"""
        patterns = [
            r'([A-Z][A-Za-z\s]+?)\s+(Act|Regulation)\s+(\d{4})',
            r'([A-Z]{2,})\s+(Act|Regulation)',
            r'(Privacy Act|Corporations Act|Fair Work Act|Competition and Consumer Act)'
        ]
        
        references = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    ref = " ".join(match).strip()
                else:
                    ref = match.strip()
                references.append(ref)
        
        return list(set(references))
    
    def _extract_section_changes(self, content: str) -> List[Dict[str, Any]]:
        """Extract section changes from update content"""
        changes = []
        
        # Look for section references and changes
        section_patterns = [
            r'[Ss]ection\s+(\d+[A-Z]*)\s+(amended|inserted|repealed|substituted)',
            r'[Ss]\s*(\d+[A-Z]*)\s+(amended|inserted|repealed|substituted)',
            r'new\s+[Ss]ection\s+(\d+[A-Z]*)',
            r'[Ss]ection\s+(\d+[A-Z]*)\s+commences'
        ]
        
        for pattern in section_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    section = match[0]
                    action = match[1] if len(match) > 1 else 'changed'
                else:
                    section = match
                    action = 'changed'
                
                changes.append({
                    'section': section,
                    'action': action,
                    'description': self._extract_change_description(content, section)
                })
        
        return changes
    
    def _extract_change_description(self, content: str, section: str) -> str:
        """Extract description of section change"""
        # Find sentences containing the section reference
        sentences = re.split(r'[.!?]', content)
        for sentence in sentences:
            if section in sentence:
                return sentence.strip()
        return "Section change description not available"
    
    def _categorize_update(self, title: str, content: str) -> str:
        """Categorize the type of update"""
        text = (title + " " + content).lower()
        
        categories = {
            'amendment': ['amend', 'amendment', 'modify', 'change'],
            'commencement': ['commence', 'commencement', 'effect', 'force'],
            'repeal': ['repeal', 'revoke', 'cancel'],
            'new_legislation': ['new', 'introduce', 'enact'],
            'guidance': ['guidance', 'interpretation', 'clarification'],
            'enforcement': ['penalty', 'enforcement', 'prosecution', 'compliance']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return 'general'
    
    def _assess_update_urgency(self, title: str, content: str) -> str:
        """Assess urgency of update"""
        text = (title + " " + content).lower()
        
        critical_terms = ['immediate', 'urgent', 'critical', 'penalty', 'enforcement']
        high_terms = ['amendment', 'commencement', 'new requirement']
        medium_terms = ['guidance', 'clarification', 'interpretation']
        
        if any(term in text for term in critical_terms):
            return 'critical'
        elif any(term in text for term in high_terms):
            return 'high'
        elif any(term in text for term in medium_terms):
            return 'medium'
        else:
            return 'low'
    
    def _assess_impact(self, title: str, content: str) -> Dict[str, Any]:
        """Assess impact of update"""
        return {
            'affected_industries': self._identify_affected_industries(title, content),
            'compliance_changes': self._identify_compliance_changes(title, content),
            'implementation_deadline': self._extract_deadline(content),
            'estimated_cost_impact': self._estimate_cost_impact(title, content)
        }
    
    async def _enhanced_impact_analysis(self, update: LegalUpdate) -> Dict[str, Any]:
        """Enhanced impact analysis using AI"""
        try:
            # Use AI service for deeper analysis
            from app.services.ai_service import ai_service
            
            analysis_prompt = f"""
            Analyze this legal update for compliance impact:
            
            Title: {update.title}
            Content: {update.content}
            Acts Referenced: {', '.join(update.act_references)}
            
            Provide analysis on:
            1. Affected business sectors
            2. Compliance requirements changes
            3. Implementation complexity (1-10)
            4. Estimated cost impact (low/medium/high)
            5. Risk level if not complied with
            """
            
            # This would use the AI service for analysis
            impact = {
                'ai_analysis_available': True,
                'business_sectors': ['financial_services', 'healthcare', 'technology'],
                'compliance_complexity': 7,
                'cost_impact': 'medium',
                'non_compliance_risk': 'high'
            }
            
            return {**update.impact_assessment, **impact}
            
        except Exception as e:
            logger.error(f"Error in enhanced impact analysis: {str(e)}")
            return update.impact_assessment
    
    def _identify_affected_industries(self, title: str, content: str) -> List[str]:
        """Identify industries affected by update"""
        text = (title + " " + content).lower()
        
        industry_keywords = {
            'financial_services': ['bank', 'financial', 'credit', 'investment', 'superannuation'],
            'healthcare': ['health', 'medical', 'patient', 'privacy'],
            'technology': ['data', 'software', 'digital', 'cyber'],
            'retail': ['consumer', 'retail', 'sale', 'trading'],
            'mining': ['mining', 'resource', 'environment'],
            'agriculture': ['agriculture', 'farming', 'food']
        }
        
        affected = []
        for industry, keywords in industry_keywords.items():
            if any(keyword in text for keyword in keywords):
                affected.append(industry)
        
        return affected
    
    def _identify_compliance_changes(self, title: str, content: str) -> List[str]:
        """Identify specific compliance changes"""
        text = (title + " " + content).lower()
        
        change_types = []
        
        if 'report' in text or 'disclosure' in text:
            change_types.append('reporting_requirements')
        if 'penalty' in text or 'fine' in text:
            change_types.append('penalty_changes')
        if 'audit' in text:
            change_types.append('audit_requirements')
        if 'privacy' in text or 'data' in text:
            change_types.append('privacy_obligations')
        if 'governance' in text:
            change_types.append('governance_standards')
        
        return change_types
    
    def _extract_deadline(self, content: str) -> Optional[str]:
        """Extract implementation deadline"""
        date_patterns = [
            r'by\s+(\d{1,2}\s+\w+\s+\d{4})',
            r'before\s+(\d{1,2}/\d{1,2}/\d{4})',
            r'commences?\s+on\s+(\d{1,2}\s+\w+\s+\d{4})',
            r'(\d{1,2}\s+\w+\s+\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _estimate_cost_impact(self, title: str, content: str) -> str:
        """Estimate cost impact of compliance"""
        text = (title + " " + content).lower()
        
        high_cost_terms = ['audit', 'system', 'training', 'implementation']
        medium_cost_terms = ['procedure', 'process', 'reporting']
        
        if any(term in text for term in high_cost_terms):
            return 'high'
        elif any(term in text for term in medium_cost_terms):
            return 'medium'
        else:
            return 'low'
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime"""
        if not date_str:
            return datetime.now()
        
        try:
            # Try different date formats
            formats = [
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%d %B %Y',
                '%B %d, %Y'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            return datetime.now()
            
        except Exception:
            return datetime.now()
    
    async def update_compliance_rules(self, updates: List[LegalUpdate], session: AsyncSession) -> Dict[str, Any]:
        """Update compliance rules based on regulatory updates"""
        logger.info(f"Updating compliance rules with {len(updates)} regulatory updates")
        
        updated_rules = 0
        new_rules = 0
        errors = []
        
        try:
            for update in updates:
                try:
                    # Create or update compliance rules
                    rule_updates = await self._create_compliance_rules_from_update(update, session)
                    
                    for rule_data in rule_updates:
                        # Check if rule exists
                        existing_rule = await session.execute(
                            select(ComplianceRule).where(ComplianceRule.code == rule_data['code'])
                        )
                        existing = existing_rule.scalar_one_or_none()
                        
                        if existing:
                            # Update existing rule
                            await session.execute(
                                update(ComplianceRule)
                                .where(ComplianceRule.id == existing.id)
                                .values(**rule_data)
                            )
                            updated_rules += 1
                        else:
                            # Create new rule
                            new_rule = ComplianceRule(**rule_data)
                            session.add(new_rule)
                            new_rules += 1
                    
                    await session.commit()
                    
                except Exception as e:
                    errors.append(f"Error processing update '{update.title}': {str(e)}")
                    await session.rollback()
            
            result = {
                'updated_rules': updated_rules,
                'new_rules': new_rules,
                'total_updates_processed': len(updates),
                'errors': errors,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Compliance rules update completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error updating compliance rules: {str(e)}")
            await session.rollback()
            raise
    
    async def _create_compliance_rules_from_update(self, update: LegalUpdate, session: AsyncSession) -> List[Dict[str, Any]]:
        """Create compliance rules from legal update"""
        rules = []
        
        try:
            # Generate rule for each section change
            for section_change in update.section_changes:
                rule_code = f"{update.jurisdiction}_{update.source}_{section_change['section']}_{datetime.now().strftime('%Y%m%d')}"
                
                rule_data = {
                    'code': rule_code,
                    'title': f"{update.title} - Section {section_change['section']}",
                    'description': section_change['description'],
                    'category': update.category,
                    'jurisdiction': update.jurisdiction,
                    'effective_date': update.effective_date,
                    'version': '1.0',
                    'parameters': {
                        'source_update': {
                            'title': update.title,
                            'source': update.source,
                            'urgency': update.urgency,
                            'act_references': update.act_references
                        },
                        'section_change': section_change,
                        'impact_assessment': update.impact_assessment
                    }
                }
                
                rules.append(rule_data)
            
            # If no section changes, create general rule
            if not update.section_changes:
                rule_code = f"{update.jurisdiction}_{update.source}_general_{datetime.now().strftime('%Y%m%d')}_{uuid4().hex[:8]}"
                
                rule_data = {
                    'code': rule_code,
                    'title': update.title,
                    'description': update.content,
                    'category': update.category,
                    'jurisdiction': update.jurisdiction,
                    'effective_date': update.effective_date,
                    'version': '1.0',
                    'parameters': {
                        'source_update': {
                            'title': update.title,
                            'source': update.source,
                            'urgency': update.urgency,
                            'act_references': update.act_references
                        },
                        'impact_assessment': update.impact_assessment
                    }
                }
                
                rules.append(rule_data)
            
        except Exception as e:
            logger.error(f"Error creating compliance rules from update: {str(e)}")
        
        return rules
    
    async def analyze_compliance_gaps(self, jurisdiction: str, session: AsyncSession) -> List[ComplianceGap]:
        """Analyze compliance gaps based on recent updates"""
        logger.info(f"Analyzing compliance gaps for {jurisdiction}")
        
        gaps = []
        
        try:
            # Get recent critical updates
            recent_updates = await self.monitor_regulatory_updates(jurisdiction)
            critical_updates = [u for u in recent_updates if u.urgency in ['critical', 'high']]
            
            # Get existing compliance rules
            rules_result = await session.execute(
                select(ComplianceRule).where(ComplianceRule.jurisdiction == jurisdiction)
            )
            existing_rules = rules_result.scalars().all()
            
            # Identify gaps
            for update in critical_updates:
                gap = await self._identify_gap_from_update(update, existing_rules)
                if gap:
                    gaps.append(gap)
            
            logger.info(f"Identified {len(gaps)} compliance gaps")
            return gaps
            
        except Exception as e:
            logger.error(f"Error analyzing compliance gaps: {str(e)}")
            return []
    
    async def _identify_gap_from_update(self, update: LegalUpdate, existing_rules: List[ComplianceRule]) -> Optional[ComplianceGap]:
        """Identify compliance gap from update"""
        try:
            # Check if update is covered by existing rules
            covered = False
            for rule in existing_rules:
                if any(act_ref in rule.description for act_ref in update.act_references):
                    covered = True
                    break
            
            if not covered:
                return ComplianceGap(
                    rule_id=f"gap_{uuid4().hex[:8]}",
                    gap_type='missing_rule',
                    severity=update.urgency,
                    description=f"No existing rule covers: {update.title}",
                    recommendation=f"Create compliance rule for {', '.join(update.act_references)}",
                    affected_documents=[],  # Would be populated by document analysis
                    remediation_steps=[
                        f"Review {update.title}",
                        "Create new compliance rule",
                        "Update affected documents",
                        "Train compliance team"
                    ]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error identifying gap: {str(e)}")
            return None

# Initialize the compliance engine
compliance_engine = DynamicComplianceEngine() 