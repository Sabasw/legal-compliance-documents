# app/database/db/db_connection.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

# Create SQLite engine and session
DATABASE_URL = (
    f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    f"?sslmode=disable"
)

engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20, pool_timeout=60)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# DON'T CREATE TABLES HERE - Remove this line!
# Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency to get the database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_engine():
    """Get the database engine"""
    return engine

def create_tables():
    """Create all database tables - call this after importing models"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")

def drop_tables():
    """Drop all database tables - BE CAREFUL!"""
    Base.metadata.drop_all(bind=engine)
    print("🗑️ All tables dropped!")