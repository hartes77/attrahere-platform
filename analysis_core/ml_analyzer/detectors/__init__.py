"""
ML Anti-Pattern Detectors Package

This package contains specialized detectors for different types of ML anti-patterns.
Each detector is focused on a specific domain of ML issues.
"""

from .base_detector import BaseMLDetector
from .test_contamination_detector import TestSetContaminationDetector
from .data_flow_contamination_detector import DataFlowContaminationDetector
from .temporal_leakage_detector import TemporalLeakageDetector
from .magic_numbers_detector import MagicNumberExtractor
from .preprocessing_leakage_detector import PreprocessingLeakageDetector

# TODO: Import other detectors as they are refactored
# from .data_leakage_detector import DataLeakageDetector
# from .gpu_memory_detector import GPUMemoryLeakDetector
# from .reproducibility_detector import ReproducibilityChecker
# from .hardcoded_thresholds_detector import HardcodedThresholdsDetector
# from .inefficient_loading_detector import InefficientDataLoadingDetector

__all__ = [
    'BaseMLDetector',
    'TestSetContaminationDetector',
    'DataFlowContaminationDetector',
    'TemporalLeakageDetector',
    'MagicNumberExtractor',
    'PreprocessingLeakageDetector',
    # TODO: Add other detectors as they are refactored
    # 'DataLeakageDetector', 
    # 'GPUMemoryLeakDetector',
    # 'ReproducibilityChecker',
    # 'HardcodedThresholdsDetector',
    # 'InefficientDataLoadingDetector'
]