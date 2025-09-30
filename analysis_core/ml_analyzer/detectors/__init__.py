"""
ML Anti-Pattern Detectors Package

This package contains specialized detectors for different types of ML anti-patterns.
Each detector is focused on a specific domain of ML issues.
"""

from .base_detector import BaseMLDetector
from .test_contamination_detector import TestSetContaminationDetector

# TODO: Import other detectors as they are refactored
# from .data_leakage_detector import DataLeakageDetector
# from .gpu_memory_detector import GPUMemoryLeakDetector
# from .magic_numbers_detector import MagicNumberExtractor
# from .reproducibility_detector import ReproducibilityChecker
# from .hardcoded_thresholds_detector import HardcodedThresholdsDetector
# from .inefficient_loading_detector import InefficientDataLoadingDetector

__all__ = [
    'BaseMLDetector',
    'TestSetContaminationDetector',
    # TODO: Add other detectors as they are refactored
    # 'DataLeakageDetector', 
    # 'GPUMemoryLeakDetector',
    # 'MagicNumberExtractor',
    # 'ReproducibilityChecker',
    # 'HardcodedThresholdsDetector',
    # 'InefficientDataLoadingDetector'
]