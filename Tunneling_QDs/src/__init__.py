"""
Tunneling_QDs Engine Package

This package contains the core logic for quantum dot simulation,
data generation, and model training.
"""

__version__ = "0.1.0"

from .config import default_params_5
from .physics import *
from .generator import generate_and_save_dqd_dataset
from .dataloader import *
from .model import *

__all__ = [
    'default_params_5',
    'generate_and_save_dqd_dataset',
]
