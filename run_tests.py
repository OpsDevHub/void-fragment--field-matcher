"""
Run all tests.

Usage:
    python run_tests.py
"""

import sys
import pytest

if __name__ == "__main__":
    # Run pytest with verbose output, exit with appropriate code
    sys.exit(pytest.main(["-v", "."]))
