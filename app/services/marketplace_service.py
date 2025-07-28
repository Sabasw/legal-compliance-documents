"""
Legal Add-on Marketplace Service
Manages legal compliance add-ons, modules, and marketplace functionality
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from uuid import uuid4
import json

from app.services.billing_service import billing_service
from app.services.encryption_service import encryption_service

logger = logging.getLogger(__name__)

@dataclass
class LegalAddOn:
    """Legal add-on product"""
    addon_id: str
    name: str
    description: str
    category: str
    provider: str
    version: str
    price: float
    pricing_model: str  # one_time, monthly, usage_based
    features: List[str]
    compatibility: List[str]
    rating: float
    downloads: int
    created_at: datetime
    updated_at: datetime
    is_active: bool
    metadata: Dict[str, Any]

@dataclass
class MarketplaceCategory:
    """Marketplace category"""
    category_id: str
    name: str
    description: str
    icon: str
    addon_count: int
    featured_addons: List[str]

@dataclass
class UserSubscription:
    """User's add-on subscription"""
    subscription_id: str
    user_id: str
    addon_id: str
    status: str
    started_at: datetime
    expires_at: Optional[datetime]
    auto_renew: bool
    usage_data: Dict[str, Any]

class MarketplaceService:
    """Legal add-on marketplace service"""
    
    def __init__(self):
        self.categories = {
            "compliance_templates": {
                "name": "Compliance Templates",
                "description": "Pre-built compliance document templates",
                "icon": "📋"
            },
            "industry_modules": {
                "name": "Industry Modules",
                "description": "Specialized compliance modules for specific industries",
                "icon": "🏢"
            },
            "ai_enhancements": {
                "name": "AI Enhancements",
                "description": "Advanced AI features and models",
                "icon": "🤖"
            },
            "integration_connectors": {
                "name": "Integration Connectors",
                "description": "Third-party system integrations",
                "icon": "🔗"
            },
            "reporting_tools": {
                "name": "Reporting Tools",
                "description": "Advanced reporting and analytics tools",
                "icon": "📊"
            },
            "security_addons": {
                "name": "Security Add-ons",
                "description": "Enhanced security and audit features",
                "icon": "🔒"
            }
        }
        
        # Initialize marketplace with sample add-ons
        self.sample_addons = self._create_sample_addons()
    
    def _create_sample_addons(self) -> List[LegalAddOn]:
        """Create sample marketplace add-ons"""
        
        addons = [
            LegalAddOn(
                addon_id="addon_001",
                name="GDPR Compliance Suite",
                description="Complete GDPR compliance toolkit with templates, checklist, and automated assessments",
                category="compliance_templates",
                provider="ComplianceFirst",
                version="2.1.0",
                price=299.99,
                pricing_model="one_time",
                features=[
                    "GDPR assessment templates",
                    "Data mapping tools",
                    "Breach notification templates",
                    "Cookie consent manager",
                    "DPO workflow automation"
                ],
                compatibility=["EU", "UK"],
                rating=4.8,
                downloads=1247,
                created_at=datetime.now() - timedelta(days=180),
                updated_at=datetime.now() - timedelta(days=30),
                is_active=True,
                metadata={
                    "languages": ["English", "German", "French"],
                    "certification": "ISO 27001 Compliant",
                    "support_level": "Premium"
                }
            ),
            
            LegalAddOn(
                addon_id="addon_002",
                name="Healthcare HIPAA Module",
                description="Comprehensive HIPAA compliance module for healthcare organizations",
                category="industry_modules",
                provider="HealthTech Solutions",
                version="1.5.2",
                price=49.99,
                pricing_model="monthly",
                features=[
                    "HIPAA risk assessment",
                    "PHI handling protocols",
                    "Business associate agreements",
                    "Incident response procedures",
                    "Employee training materials"
                ],
                compatibility=["US", "Healthcare"],
                rating=4.6,
                downloads=892,
                created_at=datetime.now() - timedelta(days=120),
                updated_at=datetime.now() - timedelta(days=15),
                is_active=True,
                metadata={
                    "industry": "Healthcare",
                    "regulation": "HIPAA",
                    "support_level": "Standard"
                }
            ),
            
            LegalAddOn(
                addon_id="addon_003",
                name="AI Contract Analyzer Pro",
                description="Advanced AI-powered contract analysis with clause extraction and risk scoring",
                category="ai_enhancements",
                provider="LegalAI Corp",
                version="3.0.1",
                price=0.10,
                pricing_model="usage_based",
                features=[
                    "Advanced clause extraction",
                    "Contract comparison",
                    "Risk prediction modeling",
                    "Multi-language support",
                    "Custom model training"
                ],
                compatibility=["Global"],
                rating=4.9,
                downloads=2156,
                created_at=datetime.now() - timedelta(days=90),
                updated_at=datetime.now() - timedelta(days=7),
                is_active=True,
                metadata={
                    "pricing_unit": "per document",
                    "ai_model": "GPT-4 Enhanced",
                    "support_level": "Premium"
                }
            ),
            
            LegalAddOn(
                addon_id="addon_004",
                name="Salesforce Legal Connector",
                description="Seamless integration with Salesforce for legal document management",
                category="integration_connectors",
                provider="IntegrateNow",
                version="2.3.0",
                price=199.99,
                pricing_model="monthly",
                features=[
                    "Bi-directional sync",
                    "Automated contract creation",
                    "Opportunity linking",
                    "Approval workflows",
                    "Custom field mapping"
                ],
                compatibility=["Salesforce"],
                rating=4.4,
                downloads=567,
                created_at=datetime.now() - timedelta(days=200),
                updated_at=datetime.now() - timedelta(days=45),
                is_active=True,
                metadata={
                    "integration_type": "OAuth 2.0",
                    "api_version": "v54.0",
                    "support_level": "Standard"
                }
            ),
            
            LegalAddOn(
                addon_id="addon_005",
                name="Executive Dashboard Suite",
                description="Advanced reporting and analytics for executive-level compliance insights",
                category="reporting_tools",
                provider="ReportPro",
                version="1.8.0",
                price=399.99,
                pricing_model="monthly",
                features=[
                    "Executive summaries",
                    "Compliance KPI tracking",
                    "Risk trend analysis",
                    "Regulatory change alerts",
                    "Board reporting templates"
                ],
                compatibility=["Enterprise"],
                rating=4.7,
                downloads=334,
                created_at=datetime.now() - timedelta(days=150),
                updated_at=datetime.now() - timedelta(days=20),
                is_active=True,
                metadata={
                    "user_tier": "Enterprise",
                    "customizable": True,
                    "support_level": "Premium"
                }
            ),
            
            LegalAddOn(
                addon_id="addon_006",
                name="Advanced Encryption Module",
                description="Enhanced encryption and security features for sensitive legal documents",
                category="security_addons",
                provider="SecureLegal",
                version="4.2.1",
                price=149.99,
                pricing_model="one_time",
                features=[
                    "AES-256 encryption",
                    "Digital signatures",
                    "Secure sharing",
                    "Access controls",
                    "Audit trails"
                ],
                compatibility=["Global"],
                rating=4.5,
                downloads=723,
                created_at=datetime.now() - timedelta(days=300),
                updated_at=datetime.now() - timedelta(days=10),
                is_active=True,
                metadata={
                    "encryption_level": "Military Grade",
                    "compliance": ["SOC 2", "ISO 27001"],
                    "support_level": "Premium"
                }
            )
        ]
        
        return addons
    
    async def get_marketplace_catalog(self, 
                                    category: Optional[str] = None,
                                    search_query: Optional[str] = None,
                                    price_range: Optional[tuple] = None,
                                    sort_by: str = "popularity") -> Dict[str, Any]:
        """Get marketplace catalog with filtering and sorting"""
        
        try:
            addons = self.sample_addons.copy()
            
            # Apply filters
            if category:
                addons = [addon for addon in addons if addon.category == category]
            
            if search_query:
                search_lower = search_query.lower()
                addons = [
                    addon for addon in addons 
                    if search_lower in addon.name.lower() 
                    or search_lower in addon.description.lower()
                    or any(search_lower in feature.lower() for feature in addon.features)
                ]
            
            if price_range:
                min_price, max_price = price_range
                addons = [
                    addon for addon in addons 
                    if min_price <= addon.price <= max_price
                ]
            
            # Apply sorting
            if sort_by == "popularity":
                addons.sort(key=lambda x: x.downloads, reverse=True)
            elif sort_by == "rating":
                addons.sort(key=lambda x: x.rating, reverse=True)
            elif sort_by == "price_low":
                addons.sort(key=lambda x: x.price)
            elif sort_by == "price_high":
                addons.sort(key=lambda x: x.price, reverse=True)
            elif sort_by == "newest":
                addons.sort(key=lambda x: x.created_at, reverse=True)
            elif sort_by == "updated":
                addons.sort(key=lambda x: x.updated_at, reverse=True)
            
            # Get categories with counts
            categories = await self._get_categories_with_counts(addons)
            
            return {
                "addons": [asdict(addon) for addon in addons],
                "categories": categories,
                "total_count": len(addons),
                "applied_filters": {
                    "category": category,
                    "search_query": search_query,
                    "price_range": price_range,
                    "sort_by": sort_by
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting marketplace catalog: {str(e)}")
            raise
    
    async def get_addon_details(self, addon_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific add-on"""
        
        try:
            addon = next((addon for addon in self.sample_addons if addon.addon_id == addon_id), None)
            
            if not addon:
                return None
            
            # Get additional details
            details = asdict(addon)
            details.update({
                "reviews": await self._get_addon_reviews(addon_id),
                "changelog": await self._get_addon_changelog(addon_id),
                "compatibility_details": await self._get_compatibility_details(addon),
                "pricing_tiers": await self._get_pricing_tiers(addon),
                "support_info": await self._get_support_info(addon),
                "similar_addons": await self._get_similar_addons(addon)
            })
            
            return details
            
        except Exception as e:
            logger.error(f"Error getting addon details: {str(e)}")
            return None
    
    async def purchase_addon(self, 
                           user_id: str,
                           addon_id: str,
                           session: AsyncSession,
                           pricing_tier: str = "standard") -> Dict[str, Any]:
        """Purchase an add-on"""
        
        try:
            addon = next((addon for addon in self.sample_addons if addon.addon_id == addon_id), None)
            
            if not addon:
                raise ValueError(f"Add-on {addon_id} not found")
            
            if not addon.is_active:
                raise ValueError(f"Add-on {addon_id} is not available")
            
            # Calculate price based on pricing model
            price_info = await self._calculate_addon_price(addon, pricing_tier)
            
            # Create billing record
            billing_result = await billing_service.create_subscription(
                user_id=user_id,
                plan_id=f"{addon_id}_{pricing_tier}",
                billing_cycle=self._get_billing_cycle(addon.pricing_model),
                session=session
            )
            
            # Create user subscription
            subscription = UserSubscription(
                subscription_id=str(uuid4()),
                user_id=user_id,
                addon_id=addon_id,
                status="active",
                started_at=datetime.now(),
                expires_at=self._calculate_expiry_date(addon.pricing_model),
                auto_renew=pricing_tier != "one_time",
                usage_data={}
            )
            
            # Store subscription (in production, this would go to database)
            await self._store_subscription(subscription, session)
            
            # Activate add-on for user
            await self._activate_addon_for_user(user_id, addon_id, session)
            
            return {
                "success": True,
                "subscription_id": subscription.subscription_id,
                "addon_id": addon_id,
                "addon_name": addon.name,
                "price_paid": price_info["amount"],
                "billing_cycle": self._get_billing_cycle(addon.pricing_model),
                "activation_date": subscription.started_at.isoformat(),
                "expires_at": subscription.expires_at.isoformat() if subscription.expires_at else None,
                "features_unlocked": addon.features
            }
            
        except Exception as e:
            logger.error(f"Error purchasing addon: {str(e)}")
            raise
    
    async def get_user_addons(self, user_id: str, session: AsyncSession) -> List[Dict[str, Any]]:
        """Get user's purchased add-ons"""
        
        try:
            # In production, this would query the database
            user_subscriptions = await self._get_user_subscriptions(user_id, session)
            
            user_addons = []
            for subscription in user_subscriptions:
                addon = next((addon for addon in self.sample_addons if addon.addon_id == subscription.addon_id), None)
                
                if addon:
                    user_addons.append({
                        "subscription_id": subscription.subscription_id,
                        "addon": asdict(addon),
                        "status": subscription.status,
                        "started_at": subscription.started_at.isoformat(),
                        "expires_at": subscription.expires_at.isoformat() if subscription.expires_at else None,
                        "auto_renew": subscription.auto_renew,
                        "usage_data": subscription.usage_data
                    })
            
            return user_addons
            
        except Exception as e:
            logger.error(f"Error getting user addons: {str(e)}")
            return []
    
    async def manage_subscription(self,
                                user_id: str,
                                subscription_id: str,
                                action: str,
                                session: AsyncSession) -> Dict[str, Any]:
        """Manage add-on subscription (cancel, renew, upgrade, etc.)"""
        
        try:
            subscription = await self._get_subscription(subscription_id, session)
            
            if not subscription or subscription.user_id != user_id:
                raise ValueError("Subscription not found")
            
            if action == "cancel":
                result = await self._cancel_subscription(subscription, session)
            elif action == "renew":
                result = await self._renew_subscription(subscription, session)
            elif action == "toggle_auto_renew":
                result = await self._toggle_auto_renew(subscription, session)
            elif action == "upgrade":
                result = await self._upgrade_subscription(subscription, session)
            else:
                raise ValueError(f"Invalid action: {action}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error managing subscription: {str(e)}")
            raise
    
    async def get_marketplace_analytics(self) -> Dict[str, Any]:
        """Get marketplace analytics and statistics"""
        
        try:
            total_addons = len(self.sample_addons)
            active_addons = len([addon for addon in self.sample_addons if addon.is_active])
            
            # Category distribution
            category_stats = {}
            for category_id, category_info in self.categories.items():
                count = len([addon for addon in self.sample_addons if addon.category == category_id])
                category_stats[category_id] = {
                    "name": category_info["name"],
                    "count": count,
                    "percentage": (count / total_addons * 100) if total_addons > 0 else 0
                }
            
            # Top rated add-ons
            top_rated = sorted(self.sample_addons, key=lambda x: x.rating, reverse=True)[:5]
            
            # Most popular add-ons
            most_popular = sorted(self.sample_addons, key=lambda x: x.downloads, reverse=True)[:5]
            
            # Pricing analysis
            prices = [addon.price for addon in self.sample_addons]
            avg_price = sum(prices) / len(prices) if prices else 0
            
            return {
                "total_addons": total_addons,
                "active_addons": active_addons,
                "category_distribution": category_stats,
                "top_rated_addons": [{"name": addon.name, "rating": addon.rating} for addon in top_rated],
                "most_popular_addons": [{"name": addon.name, "downloads": addon.downloads} for addon in most_popular],
                "pricing_stats": {
                    "average_price": round(avg_price, 2),
                    "min_price": min(prices) if prices else 0,
                    "max_price": max(prices) if prices else 0
                },
                "pricing_models": {
                    "one_time": len([a for a in self.sample_addons if a.pricing_model == "one_time"]),
                    "monthly": len([a for a in self.sample_addons if a.pricing_model == "monthly"]),
                    "usage_based": len([a for a in self.sample_addons if a.pricing_model == "usage_based"])
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting marketplace analytics: {str(e)}")
            return {}
    
    # Helper methods
    
    async def _get_categories_with_counts(self, addons: List[LegalAddOn]) -> List[Dict[str, Any]]:
        """Get categories with addon counts"""
        
        categories = []
        for category_id, category_info in self.categories.items():
            count = len([addon for addon in addons if addon.category == category_id])
            categories.append({
                "category_id": category_id,
                "name": category_info["name"],
                "description": category_info["description"],
                "icon": category_info["icon"],
                "addon_count": count
            })
        
        return sorted(categories, key=lambda x: x["addon_count"], reverse=True)
    
    async def _get_addon_reviews(self, addon_id: str) -> List[Dict[str, Any]]:
        """Get reviews for an add-on"""
        
        # Mock reviews data
        return [
            {
                "review_id": "rev_001",
                "user_name": "Legal Manager",
                "rating": 5,
                "title": "Excellent compliance tool",
                "comment": "This add-on has streamlined our compliance process significantly.",
                "date": "2024-01-15",
                "verified_purchase": True
            },
            {
                "review_id": "rev_002",
                "user_name": "Compliance Officer",
                "rating": 4,
                "title": "Good value for money",
                "comment": "Works well, could use more customization options.",
                "date": "2024-01-10",
                "verified_purchase": True
            }
        ]
    
    async def _get_addon_changelog(self, addon_id: str) -> List[Dict[str, Any]]:
        """Get changelog for an add-on"""
        
        return [
            {
                "version": "2.1.0",
                "date": "2024-01-01",
                "changes": [
                    "Added new GDPR assessment templates",
                    "Improved data mapping interface",
                    "Fixed minor bugs in reporting"
                ]
            },
            {
                "version": "2.0.0",
                "date": "2023-12-01",
                "changes": [
                    "Major UI overhaul",
                    "Added multi-language support",
                    "Enhanced security features"
                ]
            }
        ]
    
    async def _get_compatibility_details(self, addon: LegalAddOn) -> Dict[str, Any]:
        """Get detailed compatibility information"""
        
        return {
            "jurisdictions": addon.compatibility,
            "system_requirements": ["Python 3.8+", "PostgreSQL 12+", "Redis 6+"],
            "dependencies": ["fastapi>=0.104.1", "sqlalchemy>=2.0.23"],
            "integration_apis": ["REST API", "GraphQL", "Webhooks"]
        }
    
    async def _get_pricing_tiers(self, addon: LegalAddOn) -> List[Dict[str, Any]]:
        """Get pricing tiers for an add-on"""
        
        if addon.pricing_model == "usage_based":
            return [
                {"tier": "starter", "price": addon.price, "included": "Up to 100 documents/month"},
                {"tier": "professional", "price": addon.price * 0.8, "included": "Up to 1000 documents/month"},
                {"tier": "enterprise", "price": addon.price * 0.6, "included": "Unlimited documents"}
            ]
        else:
            return [
                {"tier": "standard", "price": addon.price, "included": "All features"}
            ]
    
    async def _get_support_info(self, addon: LegalAddOn) -> Dict[str, Any]:
        """Get support information for an add-on"""
        
        support_level = addon.metadata.get("support_level", "Standard")
        
        return {
            "support_level": support_level,
            "response_time": "24 hours" if support_level == "Premium" else "48 hours",
            "channels": ["Email", "Documentation", "Community Forum"],
            "premium_features": ["Phone support", "Dedicated account manager"] if support_level == "Premium" else []
        }
    
    async def _get_similar_addons(self, addon: LegalAddOn) -> List[Dict[str, Any]]:
        """Get similar add-ons"""
        
        similar = [
            a for a in self.sample_addons 
            if a.addon_id != addon.addon_id and a.category == addon.category
        ][:3]
        
        return [{"addon_id": a.addon_id, "name": a.name, "rating": a.rating} for a in similar]
    
    async def _calculate_addon_price(self, addon: LegalAddOn, pricing_tier: str) -> Dict[str, Any]:
        """Calculate add-on price"""
        
        tiers = await self._get_pricing_tiers(addon)
        tier_info = next((tier for tier in tiers if tier["tier"] == pricing_tier), tiers[0])
        
        return {
            "base_price": addon.price,
            "tier": pricing_tier,
            "amount": tier_info["price"],
            "currency": "USD",
            "tax": tier_info["price"] * 0.1,  # 10% tax
            "total": tier_info["price"] * 1.1
        }
    
    def _get_billing_cycle(self, pricing_model: str) -> str:
        """Get billing cycle based on pricing model"""
        
        mapping = {
            "one_time": "one_time",
            "monthly": "monthly",
            "usage_based": "monthly"
        }
        return mapping.get(pricing_model, "monthly")
    
    def _calculate_expiry_date(self, pricing_model: str) -> Optional[datetime]:
        """Calculate expiry date based on pricing model"""
        
        if pricing_model == "one_time":
            return None  # No expiry for one-time purchases
        elif pricing_model == "monthly":
            return datetime.now() + timedelta(days=30)
        else:
            return datetime.now() + timedelta(days=30)
    
    async def _store_subscription(self, subscription: UserSubscription, session: AsyncSession):
        """Store subscription in database"""
        # In production, this would store in the database
        pass
    
    async def _activate_addon_for_user(self, user_id: str, addon_id: str, session: AsyncSession):
        """Activate add-on features for user"""
        # In production, this would update user permissions/features
        pass
    
    async def _get_user_subscriptions(self, user_id: str, session: AsyncSession) -> List[UserSubscription]:
        """Get user subscriptions from database"""
        # Mock data for demonstration
        return [
            UserSubscription(
                subscription_id="sub_001",
                user_id=user_id,
                addon_id="addon_001",
                status="active",
                started_at=datetime.now() - timedelta(days=30),
                expires_at=None,
                auto_renew=False,
                usage_data={}
            )
        ]
    
    async def _get_subscription(self, subscription_id: str, session: AsyncSession) -> Optional[UserSubscription]:
        """Get subscription by ID"""
        # Mock implementation
        return UserSubscription(
            subscription_id=subscription_id,
            user_id="user_123",
            addon_id="addon_001",
            status="active",
            started_at=datetime.now() - timedelta(days=30),
            expires_at=datetime.now() + timedelta(days=30),
            auto_renew=True,
            usage_data={}
        )
    
    async def _cancel_subscription(self, subscription: UserSubscription, session: AsyncSession) -> Dict[str, Any]:
        """Cancel subscription"""
        subscription.status = "cancelled"
        subscription.auto_renew = False
        return {"status": "cancelled", "message": "Subscription cancelled successfully"}
    
    async def _renew_subscription(self, subscription: UserSubscription, session: AsyncSession) -> Dict[str, Any]:
        """Renew subscription"""
        subscription.expires_at = datetime.now() + timedelta(days=30)
        return {"status": "renewed", "expires_at": subscription.expires_at.isoformat()}
    
    async def _toggle_auto_renew(self, subscription: UserSubscription, session: AsyncSession) -> Dict[str, Any]:
        """Toggle auto-renew setting"""
        subscription.auto_renew = not subscription.auto_renew
        return {"auto_renew": subscription.auto_renew}
    
    async def _upgrade_subscription(self, subscription: UserSubscription, session: AsyncSession) -> Dict[str, Any]:
        """Upgrade subscription"""
        return {"status": "upgraded", "message": "Subscription upgraded successfully"}

# Initialize marketplace service
marketplace_service = MarketplaceService() 