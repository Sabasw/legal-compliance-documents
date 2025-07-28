from sqlalchemy import create_engine
from app.database.models.blockchain_models import Base
from app.config import settings
import logging

logger = logging.getLogger(__name__)

def create_blockchain_tables():
    """Create blockchain-related database tables"""
    try:
        # Create engine
        engine = create_engine(settings.DATABASE_URL)
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("Successfully created blockchain tables")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create blockchain tables: {str(e)}")
        return False

def main():
    """Main migration function"""
    success = create_blockchain_tables()
    if success:
        print("✅ Blockchain tables created successfully")
    else:
        print("❌ Failed to create blockchain tables")

if __name__ == "__main__":
    main() 