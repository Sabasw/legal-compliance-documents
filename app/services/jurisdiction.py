import logging
import pytz
from datetime import datetime
from typing import Dict, Any, List, Optional
from geoip2 import database

from app.config import settings

logger = logging.getLogger(__name__)

class JurisdictionHandler:
    def __init__(self):
        try:
            self.geoip_reader = database.Reader('GeoLite2-Country.mmdb')
            logger.info("Jurisdiction handler initialized with GeoIP")
        except Exception as e:
            logger.error(f"Failed to initialize GeoIP database: {str(e)}")
            self.geoip_reader = None

    def detect_jurisdiction(self, ip_address: str = None, text: str = None) -> str:
        """Detect jurisdiction from IP or document content with fallback"""
        # Try IP-based detection first
        if ip_address and self.geoip_reader:
            try:
                response = self.geoip_reader.country(ip_address)
                country_code = response.country.iso_code
                if country_code in settings.JURISDICTIONS:
                    return country_code

                # Handle EU member states
                if country_code in ['DE', 'FR', 'IT', 'ES']:  # Example EU countries
                    return "EU"
            except Exception as e:
                logger.warning(f"IP-based jurisdiction detection failed: {str(e)}")

        # Fall back to text analysis
        if text:
            text_lower = text.lower()
            for code, data in settings.JURISDICTIONS.items():
                country_name = data['name'].lower()
                if country_name in text_lower:
                    return code
                if f"({code})" in text:
                    return code

                # Check for jurisdiction-specific legislation
                for _, act_name in data['compliance_rules'].items():
                    if act_name.lower() in text_lower:
                        return code

        # Default to Australia if no detection
        return "AU"

    def get_jurisdiction_config(self, jurisdiction_code: str) -> Dict[str, Any]:
        """Get complete configuration for a specific jurisdiction"""
        config = settings.JURISDICTIONS.get(jurisdiction_code, settings.JURISDICTIONS['AU'])

        # Add timezone-aware current time
        tz = pytz.timezone(config['timezone'])
        config['current_time'] = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S %Z')

        return config

    def get_relevant_legislation(self, jurisdiction_code: str, doc_type: str) -> List[str]:
        """Get relevant legislation for document type in jurisdiction"""
        base_legislation = []
        config = self.get_jurisdiction_config(jurisdiction_code)

        # Common legislation for all document types
        base_legislation.extend(config['compliance_rules'].values())

        # Type-specific legislation
        if doc_type == "contract":
            if jurisdiction_code == "AU":
                base_legislation.extend(["Corporations Act 2001", "Electronic Transactions Act 1999"])
            elif jurisdiction_code == "UK":
                base_legislation.append("Contracts (Rights of Third Parties) Act 1999")
        elif doc_type == "policy":
            if jurisdiction_code in ["AU", "UK"]:
                base_legislation.append("Data Protection Act")

        return sorted(list(set(base_legislation)))

# Create service instance
jurisdiction_service = JurisdictionHandler() 