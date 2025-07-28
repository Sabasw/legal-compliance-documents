"""
Usage-Based Billing Service
Integrates with Stripe for payment processing and usage tracking
"""

import os
import logging
import stripe
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc
from app.database.models.models import User, Document, UsageRecord, BillingPlan, Invoice
from app.config import settings
from app.services.blockchain_service import blockchain_service

logger = logging.getLogger(__name__)

# Initialize Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY

class BillingService:
    def __init__(self):
        self.pricing_tiers = {
            'basic': {
                'monthly_fee': Decimal('29.99'),
                'included_documents': 50,
                'price_per_document': Decimal('0.50'),
                'included_ai_analysis': 10,
                'price_per_ai_analysis': Decimal('2.99'),
                'included_predictions': 5,
                'price_per_prediction': Decimal('9.99'),
                'included_storage_gb': 5,
                'price_per_gb': Decimal('1.99')
            },
            'professional': {
                'monthly_fee': Decimal('99.99'),
                'included_documents': 200,
                'price_per_document': Decimal('0.40'),
                'included_ai_analysis': 50,
                'price_per_ai_analysis': Decimal('2.49'),
                'included_predictions': 25,
                'price_per_prediction': Decimal('7.99'),
                'included_storage_gb': 25,
                'price_per_gb': Decimal('1.49')
            },
            'enterprise': {
                'monthly_fee': Decimal('299.99'),
                'included_documents': 1000,
                'price_per_document': Decimal('0.30'),
                'included_ai_analysis': 200,
                'price_per_ai_analysis': Decimal('1.99'),
                'included_predictions': 100,
                'price_per_prediction': Decimal('5.99'),
                'included_storage_gb': 100,
                'price_per_gb': Decimal('0.99')
            }
        }
    
    async def create_stripe_customer(
        self, 
        user_id: str, 
        email: str, 
        name: str = None,
        session: AsyncSession = None
    ) -> Dict[str, Any]:
        """Create a Stripe customer for the user"""
        try:
            # Create customer in Stripe
            customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata={
                    'user_id': user_id,
                    'created_at': datetime.utcnow().isoformat()
                }
            )
            
            # Update user with Stripe customer ID
            if session:
                user_query = select(User).where(User.id == user_id)
                result = await session.execute(user_query)
                user = result.scalars().first()
                
                if user:
                    user.stripe_customer_id = customer.id
                    await session.commit()
            
            logger.info(f"Created Stripe customer {customer.id} for user {user_id}")
            
            return {
                "customer_id": customer.id,
                "email": customer.email,
                "created": customer.created
            }
            
        except Exception as e:
            logger.error(f"Failed to create Stripe customer: {str(e)}")
            raise
    
    async def create_subscription(
        self,
        user_id: str,
        plan_name: str,
        payment_method_id: str = None,
        session: AsyncSession = None
    ) -> Dict[str, Any]:
        """Create a subscription for the user"""
        try:
            if plan_name not in self.pricing_tiers:
                raise ValueError(f"Invalid plan: {plan_name}")
            
            # Get user's Stripe customer ID
            user_query = select(User).where(User.id == user_id)
            result = await session.execute(user_query)
            user = result.scalars().first()
            
            if not user or not user.stripe_customer_id:
                raise ValueError("User not found or no Stripe customer ID")
            
            # Create subscription in Stripe
            subscription_data = {
                'customer': user.stripe_customer_id,
                'items': [{
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': f'SolveLex {plan_name.capitalize()} Plan',
                        },
                        'unit_amount': int(self.pricing_tiers[plan_name]['monthly_fee'] * 100),
                        'recurring': {
                            'interval': 'month',
                        },
                    },
                }],
                'metadata': {
                    'user_id': user_id,
                    'plan_name': plan_name
                }
            }
            
            if payment_method_id:
                subscription_data['default_payment_method'] = payment_method_id
            
            subscription = stripe.Subscription.create(**subscription_data)
            
            # Create billing plan record
            billing_plan = BillingPlan(
                user_id=user_id,
                plan_name=plan_name,
                stripe_subscription_id=subscription.id,
                monthly_fee=self.pricing_tiers[plan_name]['monthly_fee'],
                status='active',
                current_period_start=datetime.fromtimestamp(subscription.current_period_start),
                current_period_end=datetime.fromtimestamp(subscription.current_period_end),
                created_at=datetime.utcnow()
            )
            
            session.add(billing_plan)
            await session.commit()
            
            # Record audit trail
            await blockchain_service.record_audit(
                document_id=f"subscription_{subscription.id}",
                user_id=user_id,
                action="subscription_created",
                additional_data={
                    "plan_name": plan_name,
                    "subscription_id": subscription.id,
                    "monthly_fee": str(self.pricing_tiers[plan_name]['monthly_fee'])
                },
                session=session
            )
            
            logger.info(f"Created subscription {subscription.id} for user {user_id}")
            
            return {
                "subscription_id": subscription.id,
                "plan_name": plan_name,
                "status": subscription.status,
                "current_period_start": subscription.current_period_start,
                "current_period_end": subscription.current_period_end
            }
            
        except Exception as e:
            logger.error(f"Failed to create subscription: {str(e)}")
            raise
    
    async def record_usage(
        self,
        user_id: str,
        usage_type: str,  # 'document_upload', 'ai_analysis', 'prediction', 'storage'
        quantity: int = 1,
        metadata: Dict[str, Any] = None,
        session: AsyncSession = None
    ) -> Dict[str, Any]:
        """Record usage for billing purposes"""
        try:
            usage_record = UsageRecord(
                user_id=user_id,
                usage_type=usage_type,
                quantity=quantity,
                timestamp=datetime.utcnow(),
                metadata=metadata or {},
                billing_period=self._get_current_billing_period()
            )
            
            session.add(usage_record)
            await session.commit()
            
            # Record audit trail
            await blockchain_service.record_audit(
                document_id=f"usage_{usage_record.id}",
                user_id=user_id,
                action="usage_recorded",
                additional_data={
                    "usage_type": usage_type,
                    "quantity": quantity,
                    "billing_period": usage_record.billing_period
                },
                session=session
            )
            
            logger.info(f"Recorded usage: {usage_type} x{quantity} for user {user_id}")
            
            return {
                "usage_id": usage_record.id,
                "usage_type": usage_type,
                "quantity": quantity,
                "timestamp": usage_record.timestamp,
                "billing_period": usage_record.billing_period
            }
            
        except Exception as e:
            logger.error(f"Failed to record usage: {str(e)}")
            raise
    
    async def get_usage_summary(
        self,
        user_id: str,
        billing_period: str = None,
        session: AsyncSession = None
    ) -> Dict[str, Any]:
        """Get usage summary for a user"""
        try:
            if not billing_period:
                billing_period = self._get_current_billing_period()
            
            # Get usage records for the billing period
            usage_query = select(
                UsageRecord.usage_type,
                func.sum(UsageRecord.quantity).label('total_quantity')
            ).where(
                and_(
                    UsageRecord.user_id == user_id,
                    UsageRecord.billing_period == billing_period
                )
            ).group_by(UsageRecord.usage_type)
            
            result = await session.execute(usage_query)
            usage_data = result.fetchall()
            
            # Get user's billing plan
            plan_query = select(BillingPlan).where(
                and_(
                    BillingPlan.user_id == user_id,
                    BillingPlan.status == 'active'
                )
            ).order_by(desc(BillingPlan.created_at))
            
            plan_result = await session.execute(plan_query)
            billing_plan = plan_result.scalars().first()
            
            plan_name = billing_plan.plan_name if billing_plan else 'basic'
            plan_limits = self.pricing_tiers.get(plan_name, self.pricing_tiers['basic'])
            
            # Calculate usage and costs
            usage_summary = {
                'user_id': user_id,
                'billing_period': billing_period,
                'plan_name': plan_name,
                'monthly_fee': float(plan_limits['monthly_fee']),
                'usage_details': {},
                'overage_charges': {},
                'total_overage': 0.0
            }
            
            for usage_type, total_quantity in usage_data:
                # Map usage types to plan limits
                limit_key = {
                    'document_upload': 'included_documents',
                    'ai_analysis': 'included_ai_analysis', 
                    'prediction': 'included_predictions',
                    'storage': 'included_storage_gb'
                }.get(usage_type)
                
                price_key = {
                    'document_upload': 'price_per_document',
                    'ai_analysis': 'price_per_ai_analysis',
                    'prediction': 'price_per_prediction',
                    'storage': 'price_per_gb'
                }.get(usage_type)
                
                included_quantity = plan_limits.get(limit_key, 0)
                price_per_unit = plan_limits.get(price_key, Decimal('0.00'))
                
                overage_quantity = max(0, total_quantity - included_quantity)
                overage_cost = float(overage_quantity * price_per_unit)
                
                usage_summary['usage_details'][usage_type] = {
                    'total_used': total_quantity,
                    'included_in_plan': included_quantity,
                    'overage_quantity': overage_quantity,
                    'price_per_unit': float(price_per_unit),
                    'overage_cost': overage_cost
                }
                
                usage_summary['overage_charges'][usage_type] = overage_cost
                usage_summary['total_overage'] += overage_cost
            
            usage_summary['total_amount'] = usage_summary['monthly_fee'] + usage_summary['total_overage']
            
            return usage_summary
            
        except Exception as e:
            logger.error(f"Failed to get usage summary: {str(e)}")
            raise
    
    async def generate_invoice(
        self,
        user_id: str,
        billing_period: str = None,
        session: AsyncSession = None
    ) -> Dict[str, Any]:
        """Generate invoice for a billing period"""
        try:
            if not billing_period:
                billing_period = self._get_current_billing_period()
            
            # Get usage summary
            usage_summary = await self.get_usage_summary(user_id, billing_period, session)
            
            # Check if invoice already exists
            existing_invoice_query = select(Invoice).where(
                and_(
                    Invoice.user_id == user_id,
                    Invoice.billing_period == billing_period
                )
            )
            
            result = await session.execute(existing_invoice_query)
            existing_invoice = result.scalars().first()
            
            if existing_invoice:
                return {
                    "invoice_id": existing_invoice.id,
                    "status": "already_exists",
                    "total_amount": float(existing_invoice.total_amount)
                }
            
            # Create invoice
            invoice = Invoice(
                user_id=user_id,
                billing_period=billing_period,
                plan_name=usage_summary['plan_name'],
                monthly_fee=Decimal(str(usage_summary['monthly_fee'])),
                overage_charges=Decimal(str(usage_summary['total_overage'])),
                total_amount=Decimal(str(usage_summary['total_amount'])),
                usage_details=usage_summary['usage_details'],
                status='generated',
                generated_at=datetime.utcnow()
            )
            
            session.add(invoice)
            await session.commit()
            
            # Create Stripe invoice if overage charges exist
            stripe_invoice_id = None
            if usage_summary['total_overage'] > 0:
                stripe_invoice_id = await self._create_stripe_overage_invoice(
                    user_id, usage_summary, session
                )
                
                invoice.stripe_invoice_id = stripe_invoice_id
                await session.commit()
            
            # Record audit trail
            await blockchain_service.record_audit(
                document_id=f"invoice_{invoice.id}",
                user_id=user_id,
                action="invoice_generated",
                additional_data={
                    "billing_period": billing_period,
                    "total_amount": str(usage_summary['total_amount']),
                    "stripe_invoice_id": stripe_invoice_id
                },
                session=session
            )
            
            logger.info(f"Generated invoice {invoice.id} for user {user_id}")
            
            return {
                "invoice_id": invoice.id,
                "billing_period": billing_period,
                "total_amount": float(invoice.total_amount),
                "monthly_fee": float(invoice.monthly_fee),
                "overage_charges": float(invoice.overage_charges),
                "stripe_invoice_id": stripe_invoice_id,
                "usage_details": usage_summary['usage_details']
            }
            
        except Exception as e:
            logger.error(f"Failed to generate invoice: {str(e)}")
            raise
    
    async def _create_stripe_overage_invoice(
        self,
        user_id: str,
        usage_summary: Dict[str, Any],
        session: AsyncSession = None
    ) -> Optional[str]:
        """Create Stripe invoice for overage charges"""
        try:
            # Get user's Stripe customer ID
            user_query = select(User).where(User.id == user_id)
            result = await session.execute(user_query)
            user = result.scalars().first()
            
            if not user or not user.stripe_customer_id:
                logger.warning(f"No Stripe customer ID for user {user_id}")
                return None
            
            # Create invoice in Stripe
            invoice = stripe.Invoice.create(
                customer=user.stripe_customer_id,
                auto_advance=True,
                metadata={
                    'user_id': user_id,
                    'billing_period': usage_summary['billing_period'],
                    'type': 'overage_charges'
                }
            )
            
            # Add line items for overage charges
            for usage_type, details in usage_summary['usage_details'].items():
                if details['overage_cost'] > 0:
                    stripe.InvoiceItem.create(
                        customer=user.stripe_customer_id,
                        invoice=invoice.id,
                        amount=int(details['overage_cost'] * 100),  # Convert to cents
                        currency='usd',
                        description=f"{usage_type.replace('_', ' ').title()} overage: {details['overage_quantity']} units"
                    )
            
            # Finalize invoice
            invoice.finalize()
            
            return invoice.id
            
        except Exception as e:
            logger.error(f"Failed to create Stripe overage invoice: {str(e)}")
            return None
    
    async def get_billing_history(
        self,
        user_id: str,
        limit: int = 12,
        session: AsyncSession = None
    ) -> List[Dict[str, Any]]:
        """Get billing history for a user"""
        try:
            query = select(Invoice).where(
                Invoice.user_id == user_id
            ).order_by(desc(Invoice.generated_at)).limit(limit)
            
            result = await session.execute(query)
            invoices = result.scalars().all()
            
            return [
                {
                    "invoice_id": invoice.id,
                    "billing_period": invoice.billing_period,
                    "plan_name": invoice.plan_name,
                    "monthly_fee": float(invoice.monthly_fee),
                    "overage_charges": float(invoice.overage_charges),
                    "total_amount": float(invoice.total_amount),
                    "status": invoice.status,
                    "generated_at": invoice.generated_at,
                    "paid_at": invoice.paid_at,
                    "stripe_invoice_id": invoice.stripe_invoice_id
                }
                for invoice in invoices
            ]
            
        except Exception as e:
            logger.error(f"Failed to get billing history: {str(e)}")
            return []
    
    async def cancel_subscription(
        self,
        user_id: str,
        session: AsyncSession = None
    ) -> Dict[str, Any]:
        """Cancel user's subscription"""
        try:
            # Get active billing plan
            plan_query = select(BillingPlan).where(
                and_(
                    BillingPlan.user_id == user_id,
                    BillingPlan.status == 'active'
                )
            )
            
            result = await session.execute(plan_query)
            billing_plan = result.scalars().first()
            
            if not billing_plan:
                raise ValueError("No active subscription found")
            
            # Cancel subscription in Stripe
            subscription = stripe.Subscription.modify(
                billing_plan.stripe_subscription_id,
                cancel_at_period_end=True
            )
            
            # Update billing plan status
            billing_plan.status = 'cancelled'
            billing_plan.cancelled_at = datetime.utcnow()
            await session.commit()
            
            # Record audit trail
            await blockchain_service.record_audit(
                document_id=f"subscription_{billing_plan.stripe_subscription_id}",
                user_id=user_id,
                action="subscription_cancelled",
                additional_data={
                    "subscription_id": billing_plan.stripe_subscription_id,
                    "plan_name": billing_plan.plan_name
                },
                session=session
            )
            
            logger.info(f"Cancelled subscription for user {user_id}")
            
            return {
                "status": "cancelled",
                "cancel_at_period_end": subscription.cancel_at_period_end,
                "current_period_end": subscription.current_period_end
            }
            
        except Exception as e:
            logger.error(f"Failed to cancel subscription: {str(e)}")
            raise
    
    def _get_current_billing_period(self) -> str:
        """Get current billing period (YYYY-MM format)"""
        now = datetime.utcnow()
        return f"{now.year}-{now.month:02d}"
    
    async def get_current_plan(
        self,
        user_id: str,
        session: AsyncSession = None
    ) -> Optional[Dict[str, Any]]:
        """Get user's current billing plan"""
        try:
            query = select(BillingPlan).where(
                and_(
                    BillingPlan.user_id == user_id,
                    BillingPlan.status == 'active'
                )
            ).order_by(desc(BillingPlan.created_at))
            
            result = await session.execute(query)
            billing_plan = result.scalars().first()
            
            if not billing_plan:
                return None
            
            plan_details = self.pricing_tiers.get(billing_plan.plan_name, {})
            
            return {
                "plan_name": billing_plan.plan_name,
                "monthly_fee": float(billing_plan.monthly_fee),
                "status": billing_plan.status,
                "current_period_start": billing_plan.current_period_start,
                "current_period_end": billing_plan.current_period_end,
                "limits": plan_details
            }
            
        except Exception as e:
            logger.error(f"Failed to get current plan: {str(e)}")
            return None

# Global billing service instance
billing_service = BillingService() 