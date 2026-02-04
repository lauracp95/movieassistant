import sys
from pathlib import Path

# Add the api/ directory to sys.path so "import app" works reliably
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))