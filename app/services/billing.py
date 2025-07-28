import logging
import redis
from datetime import datetime
from uuid import uuid4
from typing import Dict, Any

from config import settings

logger = logging.getLogger(__name__)

# Currency converter class placeholder - replace with actual implementation
class CurrencyConverter:
    def convert(self, amount, from_currency, to_currency):
        # Placeholder implementation
        return amount  # In real implementation, this would perform actual conversion


class BillingSystem:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            db=settings.REDIS_DB
        )
        self.currency_converter = CurrencyConverter()
        logger.info("Billing system initialized")

    def track_usage(self, user_id: str, doc_type: str, jurisdiction: str) -> Dict[str, Any]:
        """Track document analysis usage with tiered pricing and limits"""
        # Get user tier and check daily limit
        tier = self._get_user_tier(user_id)
        daily_count = self._get_daily_usage(user_id)

        if daily_count >= settings.MONETIZATION_DAILY_LIMITS[tier]:
            raise ValueError(f"Daily limit reached for tier {tier}")

        # Calculate price with jurisdiction conversion
        base_price = settings.MONETIZATION_PRICING_TIERS[tier]['per_doc']
        target_currency = settings.JURISDICTIONS.get(jurisdiction, {}).get('currency', settings.MONETIZATION_CURRENCY)

        if target_currency != settings.MONETIZATION_CURRENCY:
            try:
                base_price = self.currency_converter.convert(
                    base_price,
                    settings.MONETIZATION_CURRENCY,
                    target_currency
                )
            except Exception as e:
                logger.warning(f"Currency conversion failed: {str(e)}")
                base_price = max(base_price, settings.MONETIZATION_MINIMUM_CHARGE)

        # Ensure minimum charge
        final_price = max(base_price, settings.MONETIZATION_MINIMUM_CHARGE)

        # Generate transaction
        transaction_id = str(uuid4())
        transaction_data = {
            'user_id': user_id,
            'transaction_id': transaction_id,
            'amount': final_price,
            'currency': target_currency,
            'doc_type': doc_type,
            'jurisdiction': jurisdiction,
            'tier': tier,
            'timestamp': datetime.now().isoformat()
        }

        # Record transaction in Redis
        with self.redis_client.pipeline() as pipe:
            pipe.hset(f"transactions:{transaction_id}", mapping=transaction_data)
            pipe.lpush(f"user:{user_id}:transactions", transaction_id)
            pipe.incr(f"user:{user_id}:daily_count")
            pipe.expire(f"user:{user_id}:daily_count", 86400)  # 24h TTL
            pipe.execute()

        logger.info(f"Usage tracked: {transaction_id}")
        return {
            "transaction_id": transaction_id,
            "amount": final_price,
            "currency": target_currency,
            "daily_usage": daily_count + 1,
            "daily_limit": settings.MONETIZATION_DAILY_LIMITS[tier]
        }

    def _get_user_tier(self, user_id: str) -> str:
        """Get user's pricing tier with cache"""
        tier = self.redis_client.hget(f"user:{user_id}", "tier")
        if not tier:
            # Default to professional tier if not set
            self.redis_client.hset(f"user:{user_id}", "tier", "professional")
            return "professional"
        return tier.decode('utf-8')

    def _get_daily_usage(self, user_id: str) -> int:
        """Get today's document count for user"""
        count = self.redis_client.get(f"user:{user_id}:daily_count")
        return int(count) if count else 0

    def get_user_balance(self, user_id: str) -> Dict[str, Any]:
        """Get user's current balance and usage statistics"""
        transaction_ids = self.redis_client.lrange(f"user:{user_id}:transactions", 0, -1)
        transactions = []
        total_spent = 0.0

        for tx_id in transaction_ids:
            tx_data = self.redis_client.hgetall(f"transactions:{tx_id.decode('utf-8')}")
            if tx_data:
                tx = {k.decode('utf-8'): v.decode('utf-8') for k, v in tx_data.items()}
                transactions.append(tx)
                try:
                    total_spent += float(tx.get('amount', 0))
                except (ValueError, TypeError):
                    pass

        tier = self._get_user_tier(user_id)
        daily_usage = self._get_daily_usage(user_id)

        return {
            "tier": tier,
            "total_transactions": len(transactions),
            "total_spent": total_spent,
            "daily_usage": daily_usage,
            "daily_limit": settings.MONETIZATION_DAILY_LIMITS[tier],
            "monthly_cost": settings.MONETIZATION_PRICING_TIERS[tier]['monthly']
        } 