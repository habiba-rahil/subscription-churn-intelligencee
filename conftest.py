"""
conftest.py — pytest configuration
Ensures the src directory is on the Python path for all tests.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
