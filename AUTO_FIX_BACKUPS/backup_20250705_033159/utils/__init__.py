"""
Core utilities for Phoenix trading system
"""

import sys
import os
from pathlib import Path

# Add current package to path
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Add project root to path  
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Available modules:
# from .config import *
# from .logger import *
# from .portfolio import *
# from .data import *

__version__ = "2.0.0"
__package_name__ = "utils"
