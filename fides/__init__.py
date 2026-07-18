"""
FIDES: Fidelity-Informed Deployment for Exact Segmentation
Core library for the Training-Inference Consistency Guideline (TICP).
"""

from fides.pipeline import FIDESPipeline
from fides.configs import PipelineConfig, get_config, CONFIG_NAMES

__version__ = "1.0.0"
__author__ = "Hanyu Bao"
__email__ = "bhy_tale@bupt.edu.cn"

__all__ = [
    "FIDESPipeline",
    "PipelineConfig",
    "get_config",
    "CONFIG_NAMES",
]
