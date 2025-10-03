"""
Base ML Detector - Abstract base class for all ML anti-pattern detectors

This class provides common functionality shared across all detectors,
including code snippet extraction, AST utilities, and pattern creation helpers.
"""

import ast
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set, Union
from enum import Enum
from dataclasses import dataclass

from ..ast_engine import ASTAnalysisResult


class PatternSeverity(Enum):
    """Severity levels for ML anti-patterns"""
    CRITICAL = "critical"      # Data leakage, test contamination
    HIGH = "high"             # Memory leaks, reproducibility issues
    MEDIUM = "medium"         # Performance issues, magic numbers
    LOW = "low"               # Code quality issues


@dataclass
class MLAntiPattern:
    """Represents a detected ML anti-pattern"""
    pattern_type: str
    severity: PatternSeverity
    line_number: int
    column: int
    message: str
    explanation: str
    suggested_fix: str
    confidence: float
    code_snippet: str
    fix_snippet: str
    references: List[str] = None  # Links to documentation/papers


class BaseMLDetector(ABC):
    """
    Abstract base class for all ML anti-pattern detectors.
    
    Provides common functionality and enforces consistent interface
    across all specialized detectors.
    """
    
    def __init__(self):
        """Initialize detector with common configuration"""
        # Common ML libraries and patterns
        self.ml_libraries = {
            'sklearn', 'scikit-learn', 'torch', 'pytorch', 'tensorflow', 'tf',
            'pandas', 'numpy', 'scipy', 'xgboost', 'lightgbm', 'catboost'
        }
        
        # Common ML function patterns
        self.split_functions = {
            'train_test_split', 'TimeSeriesSplit', 'KFold', 'StratifiedKFold',
            'GroupKFold', 'LeaveOneOut', 'LeavePOut', 'StratifiedGroupKFold'
        }
        
        # Common preprocessing classes
        self.preprocessing_classes = {
            'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'PowerTransformer',
            'QuantileTransformer', 'Normalizer', 'LabelEncoder', 'OneHotEncoder',
            'OrdinalEncoder', 'TargetEncoder', 'WOEEncoder'
        }
    
    @abstractmethod
    def detect_patterns(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """
        Main entry point for pattern detection.
        
        Each detector must implement this method to perform its specific
        analysis and return a list of detected anti-patterns.
        
        Args:
            analysis: AST analysis result containing parsed code and metadata
            
        Returns:
            List of detected ML anti-patterns
        """
        pass
    
    def get_function_name(self, node: ast.Call) -> str:
        """Extract function name from call node"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""
    
    def get_full_function_name(self, node: ast.Call) -> str:
        """Extract full function name including module (e.g., 'pd.read_csv')"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
            return node.func.attr
        return ""
    
    def extract_code_snippet(self, analysis: ASTAnalysisResult, line_number: int, 
                           context_lines: int = 2) -> str:
        """
        Extract code snippet around the specified line number.
        
        Args:
            analysis: AST analysis result
            line_number: Target line number
            context_lines: Number of context lines before and after
            
        Returns:
            Formatted code snippet with line numbers
        """
        try:
            with open(analysis.file_path, 'r') as f:
                lines = f.readlines()
                
            start = max(0, line_number - 1 - context_lines)
            end = min(len(lines), line_number + context_lines)
            
            snippet_lines = []
            for i in range(start, end):
                prefix = ">>> " if i == line_number - 1 else "    "
                snippet_lines.append(f"{prefix}{lines[i].rstrip()}")
            
            return "\n".join(snippet_lines)
        except (IOError, IndexError):
            return f"Line {line_number}: [code snippet unavailable]"
    
    def is_ml_related_call(self, node: ast.Call) -> bool:
        """Check if a function call is related to ML libraries"""
        func_name = self.get_full_function_name(node).lower()
        
        # Check if any ML library is mentioned
        for lib in self.ml_libraries:
            if lib in func_name:
                return True
        
        # Check for common ML patterns
        ml_patterns = [
            'fit', 'transform', 'predict', 'score', 'cross_val',
            'train', 'test', 'split', 'scale', 'encode'
        ]
        
        return any(pattern in func_name for pattern in ml_patterns)
    
    def find_variable_assignments(self, analysis: ASTAnalysisResult, 
                                var_name: str) -> List[ast.Assign]:
        """Find all assignments to a specific variable"""
        assignments = []
        
        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        assignments.append(node)
        
        return assignments
    
    def get_node_line_number(self, node: ast.AST) -> int:
        """Get line number from AST node, with fallback"""
        return getattr(node, 'lineno', 0)
    
    def get_node_column(self, node: ast.AST) -> int:
        """Get column number from AST node, with fallback"""
        return getattr(node, 'col_offset', 0)
    
    def create_pattern(self, pattern_type: str, severity: PatternSeverity,
                      node: ast.AST, analysis: ASTAnalysisResult,
                      message: str, explanation: str, suggested_fix: str,
                      confidence: float, fix_snippet: str = "",
                      references: List[str] = None) -> MLAntiPattern:
        """
        Helper method to create ML anti-pattern with consistent structure.
        
        Args:
            pattern_type: Type identifier for the pattern
            severity: Severity level of the issue
            node: AST node where pattern was detected
            analysis: Analysis result for context
            message: Short description of the issue
            explanation: Detailed explanation of why this is problematic
            suggested_fix: How to fix the issue
            confidence: Confidence score (0.0 to 1.0)
            fix_snippet: Example of fixed code
            references: Links to documentation/papers
            
        Returns:
            Configured MLAntiPattern instance
        """
        return MLAntiPattern(
            pattern_type=pattern_type,
            severity=severity,
            line_number=self.get_node_line_number(node),
            column=self.get_node_column(node),
            message=message,
            explanation=explanation,
            suggested_fix=suggested_fix,
            confidence=confidence,
            code_snippet=self.extract_code_snippet(analysis, self.get_node_line_number(node)),
            fix_snippet=fix_snippet,
            references=references or []
        )
    
    def analyze_chronological_operations(self, analysis: ASTAnalysisResult) -> List[Dict]:
        """
        Analyze operations in chronological order for sequence-based detection.
        
        Returns:
            List of operations with metadata, sorted by line number
        """
        operations = []
        
        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Assign):
                op_info = self._analyze_assignment_operation(node)
                if op_info:
                    operations.append(op_info)
            elif isinstance(node, ast.Call):
                op_info = self._analyze_call_operation(node)
                if op_info:
                    operations.append(op_info)
        
        # Sort by line number for chronological analysis
        return sorted(operations, key=lambda x: x.get('line', 0))
    
    def _analyze_assignment_operation(self, node: ast.Assign) -> Optional[Dict]:
        """Analyze an assignment for ML-relevant operations"""
        if not isinstance(node.value, ast.Call):
            return None
        
        func_name = self.get_function_name(node.value)
        full_func_name = self.get_full_function_name(node.value)
        
        operation = {
            'type': 'assignment',
            'node': node,
            'line': self.get_node_line_number(node),
            'function': func_name,
            'full_function': full_func_name
        }
        
        # Classify operation type
        if func_name in self.split_functions:
            operation['type'] = 'data_split'
        elif func_name in self.preprocessing_classes:
            operation['type'] = 'preprocessing_creation'
        elif 'fit' in func_name:
            operation['type'] = 'preprocessing_fit'
        
        return operation
    
    def _analyze_call_operation(self, node: ast.Call) -> Optional[Dict]:
        """Analyze a function call for ML-relevant operations"""
        func_name = self.get_function_name(node)
        full_func_name = self.get_full_function_name(node)
        
        if not self.is_ml_related_call(node):
            return None
        
        return {
            'type': 'function_call',
            'node': node,
            'line': self.get_node_line_number(node),
            'function': func_name,
            'full_function': full_func_name
        }