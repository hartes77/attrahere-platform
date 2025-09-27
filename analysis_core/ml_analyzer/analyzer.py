"""
High-level ML Code Analyzer - Main entry point for ML pattern detection

This module provides a simplified, high-level API for analyzing ML code
and detecting anti-patterns. It orchestrates the various specialized
analyzers and provides a unified interface.
"""

from pathlib import Path
from typing import Dict, List, Any, Union

from .ast_engine import MLSemanticAnalyzer
from .ml_patterns import MLPatternDetector


class MLCodeAnalyzer:
    """
    High-level analyzer for ML code anti-pattern detection.

    This is the main entry point for users who want to analyze
    ML code for anti-patterns and get refactoring suggestions.
    """

    def __init__(self):
        """Initialize the ML code analyzer."""
        self.semantic_analyzer = MLSemanticAnalyzer()
        self.pattern_detector = MLPatternDetector()

    def analyze_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze a single Python file for ML anti-patterns.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            Dictionary containing analysis results with patterns detected
        """
        try:
            # Perform AST analysis
            analysis_result = self.semantic_analyzer.analyze_file(file_path)

            # Detect ML patterns
            patterns = self.pattern_detector.detect_all_patterns(analysis_result)

            # Get pattern summary
            summary = self.pattern_detector.get_pattern_summary(patterns)

            return {
                'file_path': str(file_path),
                'patterns': [self._pattern_to_dict(p) for p in patterns],
                'summary': summary,
                'analysis_metadata': {
                    'functions_analyzed': len(analysis_result.functions),
                    'classes_analyzed': len(analysis_result.classes),
                    'ml_constructs_found': len(analysis_result.ml_constructs),
                }
            }

        except Exception as e:
            return {
                'file_path': str(file_path),
                'error': str(e),
                'patterns': [],
                'summary': {'total_patterns': 0}
            }

    def analyze_code_string(self, code: str, file_path: str = "code.py") -> Dict[str, Any]:
        """
        Analyze code from a string.

        Args:
            code: Python code to analyze
            file_path: Virtual file path for context

        Returns:
            Dictionary containing analysis results
        """
        import tempfile
        import os

        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
                tmp_file.write(code)
                tmp_path = tmp_file.name

            # Analyze the temporary file
            result = self.analyze_file(tmp_path)

            # Update the file_path in result to the virtual path
            result['file_path'] = file_path

            return result

        except Exception as e:
            return {
                'file_path': file_path,
                'error': str(e),
                'patterns': [],
                'summary': {'total_patterns': 0}
            }
        finally:
            # Clean up temporary file
            try:
                if 'tmp_path' in locals():
                    os.unlink(tmp_path)
            except:
                pass

    def _pattern_to_dict(self, pattern) -> Dict[str, Any]:
        """Convert a pattern object to dictionary for serialization."""
        return {
            'type': pattern.pattern_type,
            'severity': pattern.severity.value,
            'line': pattern.line_number,
            'column': pattern.column,
            'message': pattern.message,
            'explanation': pattern.explanation,
            'suggested_fix': pattern.suggested_fix,
            'confidence': pattern.confidence,
            'code_snippet': pattern.code_snippet,
            'fix_snippet': pattern.fix_snippet,
        }
