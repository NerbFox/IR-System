"""Configuration module for pytest.

This module sets up the Python path to ensure proper module imports
during test execution by adding the project root directory to sys.path.
"""

import os
import sys

# Get the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
