import sys
from pathlib import Path
import pytest

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.database import init_db

@pytest.fixture(autouse=True)
def setup_database():
    """Ensure database tables exist before any tests run."""
    init_db()
