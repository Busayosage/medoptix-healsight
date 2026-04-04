import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# =========================
# Load environment variables
# =========================

load_dotenv()

# =========================
# Get database path from .env
# =========================

DB_PATH = os.getenv("DATABASE_PATH", "medoptix.db")

# =========================
# Create engine
# =========================

def get_engine():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    return engine

# =========================
# Test connection (optional)
# =========================

if __name__ == "__main__":
    engine = get_engine()
    print(f"Database connected successfully → {DB_PATH}")