"""
ML Anti-Pattern Detection - The core differentiator of our platform

This module contains detectors for ML-specific anti-patterns that cause:
- Data leakage (critical research validity issue)
- GPU memory waste (expensive resource issue)
- Non-reproducible results (science validity issue)
- Performance bottlenecks (efficiency issue)
"""

import ast
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Union
from pathlib import Path

from .ast_engine import MLSemanticAnalyzer, ASTAnalysisResult


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


class DataLeakageDetector:
    """
    Detects data leakage patterns - the most critical ML anti-pattern.

    Data leakage occurs when information from the future/test set
    inadvertently influences the training process, leading to
    overly optimistic but invalid results.
    """

    def __init__(self):
        self.preprocessor_classes = {
            'StandardScaler', 'MinMaxScaler', 'RobustScaler',
            'Normalizer', 'QuantileTransformer', 'PowerTransformer',
            'LabelEncoder', 'OneHotEncoder', 'OrdinalEncoder'
        }

        self.fit_methods = {'fit', 'fit_transform'}
        self.split_functions = {'train_test_split', 'KFold', 'StratifiedKFold'}

    def detect_patterns(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Main entry point for data leakage detection"""
        patterns = []

        patterns.extend(self._detect_preprocessing_before_split(analysis))
        patterns.extend(self._detect_test_set_contamination(analysis))
        patterns.extend(self._detect_target_leakage(analysis))
        patterns.extend(self._detect_temporal_leakage(analysis))

        return patterns

    def _detect_preprocessing_before_split(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """
        Detect preprocessing applied to entire dataset before train/test split.

        This is the most common and critical data leakage pattern.
        """
        patterns = []

        # Track the order of operations
        operations = []
        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Assign):
                operation_info = self._analyze_assignment_for_leakage(node)
                if operation_info:
                    operations.append(operation_info)

        # Find problematic sequences
        for i, op in enumerate(operations):
            if op['type'] == 'preprocessing' and op['method'] in self.fit_methods:
                # Look for subsequent split operations
                for j in range(i + 1, len(operations)):
                    if operations[j]['type'] == 'data_split':
                        # Found preprocessing before split - DATA LEAKAGE!
                        patterns.append(MLAntiPattern(
                            pattern_type="data_leakage_preprocessing",
                            severity=PatternSeverity.CRITICAL,
                            line_number=op['line'],
                            column=op.get('column', 0),
                            message=f"{op['preprocessor']}.{op['method']}() applied before train_test_split",
                            explanation=(
                                "Preprocessing fitted on entire dataset before splitting causes data leakage. "
                                "The preprocessor learns statistics from the test set, leading to overly optimistic performance estimates."
                            ),
                            suggested_fix=(
                                f"Move {op['preprocessor']}.fit() after train_test_split. "
                                "Fit only on training data, then transform both train and test sets."
                            ),
                            confidence=0.95,
                            code_snippet=self._extract_code_snippet(analysis, op['line']),
                            fix_snippet=self._generate_leakage_fix(op),
                            references=[
                                "https://scikit-learn.org/stable/common_pitfalls.html#data-leakage",
                                "https://machinelearningmastery.com/data-leakage-machine-learning/"
                            ]
                        ))
                        break

        return patterns

    def _detect_test_set_contamination(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """
        Detect when test set is used for any decision making.

        Test set should ONLY be used for final evaluation.
        """
        patterns = []

        # Look for test set being used in conditions or calculations
        test_vars = self._identify_test_variables(analysis)

        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.If):
                # Check if test variables are used in if conditions
                test_usage = self._find_test_variable_usage(node.test, test_vars)
                if test_usage:
                    patterns.append(MLAntiPattern(
                        pattern_type="test_set_contamination",
                        severity=PatternSeverity.CRITICAL,
                        line_number=node.lineno,
                        column=node.col_offset,
                        message=f"Test set variable '{test_usage}' used in decision making",
                        explanation=(
                            "Using test set for any decision making (model selection, hyperparameter tuning) "
                            "invalidates the test set as an unbiased performance estimate."
                        ),
                        suggested_fix="Use validation set or cross-validation for model decisions. Reserve test set only for final evaluation.",
                        confidence=0.90,
                        code_snippet=self._extract_code_snippet(analysis, node.lineno),
                        fix_snippet="# Use validation set instead:\nif validation_metric > threshold:",
                    ))

        return patterns

    def _detect_target_leakage(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """
        Detect when future information leaks into features.

        This is subtle but critical - using information that wouldn't
        be available at prediction time.
        """
        patterns = []

        # Look for feature engineering that might use future information
        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if self._is_suspicious_feature_engineering(node):
                    patterns.append(MLAntiPattern(
                        pattern_type="target_leakage",
                        severity=PatternSeverity.HIGH,
                        line_number=node.lineno,
                        column=node.col_offset,
                        message="Potential target leakage in feature engineering",
                        explanation=(
                            "Feature engineering that might incorporate future information "
                            "or information derived from the target variable."
                        ),
                        suggested_fix="Ensure features only use information available at prediction time.",
                        confidence=0.75,
                        code_snippet=self._extract_code_snippet(analysis, node.lineno),
                        fix_snippet="# Review feature engineering logic for temporal consistency"
                    ))

        return patterns

    def _detect_temporal_leakage(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """
        Detect temporal data leakage in time series data.

        When working with time series, future data must not leak into past predictions.
        """
        patterns = []

        # Look for time-based operations that might cause leakage
        time_related_calls = ['sort_values', 'shift', 'rolling', 'expanding']

        for node in ast.walk(analysis.ast_tree):
            if (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Attribute) and
                node.func.attr in time_related_calls):

                # Analyze if this could cause temporal leakage
                if self._is_temporal_leakage_risk(node):
                    patterns.append(MLAntiPattern(
                        pattern_type="temporal_leakage",
                        severity=PatternSeverity.HIGH,
                        line_number=node.lineno,
                        column=node.col_offset,
                        message=f"Potential temporal leakage with {node.func.attr}()",
                        explanation=(
                            "Time-based operations might be leaking future information into past predictions. "
                            "Ensure proper temporal ordering and avoid look-ahead bias."
                        ),
                        suggested_fix="Use time-aware cross-validation and ensure no future data leaks into features.",
                        confidence=0.70,
                        code_snippet=self._extract_code_snippet(analysis, node.lineno),
                        fix_snippet="# Use time-based splits and avoid look-ahead bias"
                    ))

        return patterns

    def _analyze_assignment_for_leakage(self, node: ast.Assign) -> Optional[Dict[str, Any]]:
        """Analyze an assignment to see if it's a leakage-relevant operation"""
        if not isinstance(node.value, ast.Call):
            return None

        # Check for preprocessing operations
        if isinstance(node.value.func, ast.Attribute):
            obj_name = None
            if isinstance(node.value.func.value, ast.Name):
                obj_name = node.value.func.value.id

            method_name = node.value.func.attr

            if method_name in self.fit_methods:
                return {
                    'type': 'preprocessing',
                    'preprocessor': obj_name or 'unknown',
                    'method': method_name,
                    'line': node.lineno,
                    'column': node.col_offset
                }

        # Check for data splitting operations
        elif isinstance(node.value.func, ast.Name):
            func_name = node.value.func.id
            if func_name in self.split_functions:
                return {
                    'type': 'data_split',
                    'function': func_name,
                    'line': node.lineno,
                    'column': node.col_offset
                }

        return None

    def _identify_test_variables(self, analysis: ASTAnalysisResult) -> Set[str]:
        """Identify variable names that likely contain test data"""
        test_vars = set()

        # Look in regular variables
        for var_name in analysis.variables.keys():
            if any(keyword in var_name.lower() for keyword in ['test', 'val', 'holdout']):
                test_vars.add(var_name)

        # Also look for variables created by train_test_split tuple unpacking
        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Assign):
                # Check if RHS is train_test_split call
                if (isinstance(node.value, ast.Call) and
                    isinstance(node.value.func, ast.Name) and
                    node.value.func.id == 'train_test_split'):

                    # Check if LHS is tuple unpacking
                    for target in node.targets:
                        if isinstance(target, (ast.Tuple, ast.List)):
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    var_name = elt.id
                                    if any(keyword in var_name.lower() for keyword in ['test', 'val', 'holdout']):
                                        test_vars.add(var_name)

        return test_vars

    def _find_test_variable_usage(self, node: ast.AST, test_vars: Set[str]) -> Optional[str]:
        """Find if any test variables are used in the given AST node"""
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in test_vars:
                return child.id
        return None

    def _is_suspicious_feature_engineering(self, node: ast.Assign) -> bool:
        """Check if feature engineering might involve target leakage"""
        # Look for operations that might use target information
        suspicious_patterns = ['mean', 'std', 'max', 'min', 'quantile']

        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute):
                if node.value.func.attr in suspicious_patterns:
                    return True

        return False

    def _is_temporal_leakage_risk(self, node: ast.Call) -> bool:
        """Assess if a time-based operation risks temporal leakage"""
        # This would need more sophisticated analysis in practice
        return True  # Conservative - flag for human review

    def _extract_code_snippet(self, analysis: ASTAnalysisResult, line_number: int) -> str:
        """Extract relevant code snippet for the pattern"""
        try:
            with open(analysis.file_path, 'r') as f:
                lines = f.readlines()
                if 0 <= line_number - 1 < len(lines):
                    return lines[line_number - 1].strip()
        except:
            pass
        return "Unable to extract code snippet"

    def _generate_leakage_fix(self, operation_info: Dict[str, Any]) -> str:
        """Generate a fix for data leakage"""
        preprocessor = operation_info['preprocessor']
        method = operation_info['method']

        return f"""# Fixed version - no data leakage:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
{preprocessor} = StandardScaler()  # or appropriate preprocessor
X_train_scaled = {preprocessor}.fit_transform(X_train)  # Fit only on training data
X_test_scaled = {preprocessor}.transform(X_test)  # Only transform test data"""


class GPUMemoryLeakDetector:
    """
    Detects GPU memory leak patterns in PyTorch/TensorFlow code.

    GPU memory is expensive and leaks can crash training or waste resources.
    """

    def __init__(self):
        self.tensor_accumulation_patterns = ['append', 'extend', '+=']
        self.gpu_operations = ['cuda', 'to_device', 'gpu']
        self.torch_functions = ['torch', 'nn', 'optim']
        self.tensor_types = ['Tensor', 'tensor', 'Variable']

    def detect_patterns(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Detect GPU memory leak patterns"""
        patterns = []

        patterns.extend(self._detect_tensor_accumulation(analysis))
        patterns.extend(self._detect_missing_detach(analysis))
        patterns.extend(self._detect_uncleaned_cache(analysis))

        return patterns

    def _detect_tensor_accumulation(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Detect accumulation of tensors with gradients in loops"""
        patterns = []

        # Look for loops that accumulate tensors
        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, (ast.For, ast.While)):
                # Track variables that might be tensors
                tensor_vars = self._identify_tensor_variables_in_scope(node)

                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                        # Check for list.append(tensor) patterns
                        if child.func.attr == 'append' and len(child.args) > 0:
                            arg = child.args[0]

                            # Check if we're appending a potential tensor
                            if self._is_likely_tensor_expression(arg, tensor_vars):
                                # Check if the tensor has .detach() called
                                if not self._has_detach_call(arg):
                                    patterns.append(MLAntiPattern(
                                        pattern_type="gpu_memory_leak",
                                        severity=PatternSeverity.HIGH,
                                        line_number=child.lineno,
                                        column=child.col_offset,
                                        message="Tensor accumulation in loop may cause GPU memory leak",
                                        explanation=(
                                            "Accumulating tensors with gradients in a loop prevents garbage collection "
                                            "and causes GPU memory to grow indefinitely."
                                        ),
                                        suggested_fix="Use .detach() or .item() to break gradient computation chain.",
                                        confidence=0.85,
                                        code_snippet=self._extract_code_snippet(analysis, child.lineno),
                                        fix_snippet="metrics.append(predictions.detach())  # Break gradient chain"
                                    ))

                        # Check for += operations with tensors
                        elif child.func.attr in ['__iadd__', 'add_'] and len(child.args) > 0:
                            if self._is_likely_tensor_expression(child.args[0], tensor_vars):
                                patterns.append(MLAntiPattern(
                                    pattern_type="gpu_memory_leak",
                                    severity=PatternSeverity.HIGH,
                                    line_number=child.lineno,
                                    column=child.col_offset,
                                    message="Tensor accumulation with += may cause memory leak",
                                    explanation="Accumulating tensors in-place can cause memory issues.",
                                    suggested_fix="Consider using detached values for accumulation.",
                                    confidence=0.80,
                                    code_snippet=self._extract_code_snippet(analysis, child.lineno),
                                    fix_snippet="total += loss.detach()  # Detach for accumulation"
                                ))

        return patterns

    def _identify_tensor_variables_in_scope(self, loop_node: ast.AST) -> Set[str]:
        """Identify variables that are likely to be tensors within a loop scope"""
        tensor_vars = set()

        # Look for assignments that create tensors
        for node in ast.walk(loop_node):
            if isinstance(node, ast.Assign):
                # Check if RHS looks like tensor creation/operation
                if self._is_tensor_creating_expression(node.value):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            tensor_vars.add(target.id)

            # Look for function calls that return tensors
            elif isinstance(node, ast.Call):
                if self._is_tensor_returning_call(node):
                    # If this is assigned to a variable, track it
                    parent = getattr(node, 'parent', None)
                    if isinstance(parent, ast.Assign):
                        for target in parent.targets:
                            if isinstance(target, ast.Name):
                                tensor_vars.add(target.id)

        return tensor_vars

    def _is_likely_tensor_expression(self, expr: ast.AST, tensor_vars: Set[str]) -> bool:
        """Check if an expression is likely to produce a tensor"""
        if isinstance(expr, ast.Name):
            # Check if it's a known tensor variable
            if expr.id in tensor_vars:
                return True
            
            var_name = expr.id.lower()
            
            # Direct tensor indicators
            tensor_keywords = ['tensor', 'output', 'prediction', 'logits', 'features']
            if any(keyword in var_name for keyword in tensor_keywords):
                return True
                
            # Loss variables are tensors UNLESS they have scalar prefixes
            if 'loss' in var_name:
                # Exclude scalar computations
                scalar_prefixes = ['avg', 'mean', 'final', 'total', 'sum']
                if not any(prefix in var_name for prefix in scalar_prefixes):
                    return True
                    
            return False

        elif isinstance(expr, ast.Call):
            # Check if it's a tensor-returning function call
            return self._is_tensor_returning_call(expr)

        elif isinstance(expr, ast.Attribute):
            # Check for model(input) type calls
            return any(keyword in expr.attr.lower() for keyword in ['forward', 'predict', 'loss'])

        return False

    def _is_tensor_creating_expression(self, expr: ast.AST) -> bool:
        """Check if expression creates a tensor"""
        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Name):
                # Direct torch functions
                return expr.func.id in ['torch', 'tensor'] or expr.func.id.startswith('torch.')
            elif isinstance(expr.func, ast.Attribute):
                # Method calls that create tensors
                if hasattr(expr.func, 'value') and isinstance(expr.func.value, ast.Name):
                    base = expr.func.value.id
                    return (base in self.torch_functions or
                           expr.func.attr in ['tensor', 'zeros', 'ones', 'randn', 'rand'])

        return False

    def _is_tensor_returning_call(self, call: ast.Call) -> bool:
        """Check if a function call is likely to return a tensor"""
        if isinstance(call.func, ast.Name):
            # Model call: model(input)
            return call.func.id.endswith('model') or call.func.id in ['model', 'net']
        elif isinstance(call.func, ast.Attribute):
            # Method calls on models or tensor operations
            method_name = call.func.attr.lower()
            return (method_name in ['forward', '__call__', 'predict'] or
                   method_name.endswith('loss') or
                   method_name in ['matmul', 'mm', 'bmm', 'conv2d', 'relu', 'softmax'])

        return False

    def _has_detach_call(self, expr: ast.AST) -> bool:
        """Check if expression has .detach() or .item() called on it"""
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            return expr.func.attr in ['detach', 'item', 'cpu', 'numpy']
        return False

    def _detect_missing_detach(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Detect missing .detach() calls on tensors"""
        patterns = []

        # Look for tensor operations without detach in evaluation contexts
        for node in ast.walk(analysis.ast_tree):
            if self._is_evaluation_context(node):
                for child in ast.walk(node):
                    if (isinstance(child, ast.Call) and
                        self._is_tensor_operation(child) and
                        not self._has_detach_call(child)):

                        patterns.append(MLAntiPattern(
                            pattern_type="missing_detach",
                            severity=PatternSeverity.MEDIUM,
                            line_number=child.lineno,
                            column=child.col_offset,
                            message="Missing .detach() on tensor in evaluation context",
                            explanation=(
                                "Tensors used for metrics or storage should be detached "
                                "to prevent unnecessary gradient computation."
                            ),
                            suggested_fix="Add .detach() to break gradient chain.",
                            confidence=0.80,
                            code_snippet=self._extract_code_snippet(analysis, child.lineno),
                            fix_snippet="result = model(input).detach()"
                        ))

        return patterns

    def _detect_uncleaned_cache(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Detect missing cache cleanup in training loops"""
        patterns = []

        # Look for training loops without cache cleanup
        training_loops = self._find_training_loops(analysis.ast_tree)

        for loop in training_loops:
            if not self._has_cache_cleanup(loop):
                patterns.append(MLAntiPattern(
                    pattern_type="uncleaned_gpu_cache",
                    severity=PatternSeverity.MEDIUM,
                    line_number=loop.lineno,
                    column=loop.col_offset,
                    message="Training loop missing GPU cache cleanup",
                    explanation=(
                        "Long training loops should periodically clear GPU cache "
                        "to prevent memory fragmentation."
                    ),
                    suggested_fix="Add torch.cuda.empty_cache() periodically in the loop.",
                    confidence=0.75,
                    code_snippet=self._extract_code_snippet(analysis, loop.lineno),
                    fix_snippet="""if batch_idx % 100 == 0:
    torch.cuda.empty_cache()  # Clear GPU cache periodically"""
                ))

        return patterns

    def _extract_code_snippet(self, analysis: ASTAnalysisResult, line_number: int) -> str:
        """Extract relevant code snippet for the pattern"""
        try:
            with open(analysis.file_path, 'r') as f:
                lines = f.readlines()
                if 0 <= line_number - 1 < len(lines):
                    return lines[line_number - 1].strip()
        except:
            pass
        return f"Line {line_number} (unable to extract code snippet)"

    def _is_evaluation_context(self, node: ast.AST) -> bool:
        """Check if this is an evaluation/inference context"""
        # Look for with torch.no_grad() or model.eval() patterns
        return False  # Simplified for now

    def _is_tensor_operation(self, node: ast.Call) -> bool:
        """Check if this is a tensor operation"""
        return False  # Simplified for now

    def _has_detach_call(self, node: ast.Call) -> bool:
        """Check if tensor has .detach() call"""
        return False  # Simplified for now

    def _find_training_loops(self, tree: ast.AST) -> List[ast.For]:
        """Find loops that look like training loops"""
        return []  # Simplified for now

    def _has_cache_cleanup(self, loop: ast.For) -> bool:
        """Check if loop has GPU cache cleanup"""
        return False  # Simplified for now


class MagicNumberExtractor:
    """
    Detects magic numbers and suggests extracting them to configuration files.

    Magic numbers make ML experiments non-reproducible and hard to tune.
    """

    def __init__(self):
        self.ml_contexts = ['learning_rate', 'lr', 'batch_size', 'epochs', 'hidden_size']
        
        # Standard ML values that are commonly accepted (not magic numbers)
        self.standard_ml_values = {
            0, 1, -1,      # Basic values
            0.2, 0.3, 0.8, # Common test/validation split ratios
            42, 123, 0,    # Common random seeds
            0.01, 0.001,   # Common learning rates
            32, 64, 128,   # Common batch sizes
            100, 1000,     # Common epoch/iteration counts
        }
        
        # Parameter contexts where certain values are expected and acceptable
        self.parameter_contexts = {
            'test_size': {0.1, 0.15, 0.2, 0.25, 0.3, 0.33},  # Standard test split ratios
            'val_size': {0.1, 0.15, 0.2, 0.25},              # Standard validation split ratios
            'random_state': {0, 42, 123, 1337},              # Common random seeds
            'random_seed': {0, 42, 123, 1337},               # Alternative random seed param
            'n_jobs': {-1, 1, 2, 4, 8},                      # Common parallel job counts
            'batch_size': {1, 8, 16, 32, 64, 128, 256, 512}, # Common batch sizes
        }
        
        # Common neural network dimensions (not magic numbers)
        self.standard_nn_dimensions = {
            # Input/output common sizes
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 28, 32, 64, 128, 256, 512, 784, 1024,
            # Common hidden layer sizes
            16, 32, 64, 128, 256, 512, 1024, 2048,
            # Common embedding dimensions
            50, 100, 128, 200, 256, 300, 512,
        }

    def detect_patterns(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Detect magic numbers in ML contexts"""
        patterns = []

        patterns.extend(self._detect_hardcoded_hyperparameters(analysis))
        patterns.extend(self._detect_magic_dimensions(analysis))

        return patterns

    def _detect_hardcoded_hyperparameters(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Detect hardcoded hyperparameters"""
        patterns = []

        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Call):
                # Look for ML constructor calls with numeric arguments
                if self._is_ml_constructor(node):
                    # Check positional arguments
                    for i, arg in enumerate(node.args):
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
                            # For positional args, we don't have parameter context, so be more lenient
                            if self._is_magic_number_in_constructor(arg.value):  # Use constructor-specific detection
                                patterns.append(MLAntiPattern(
                                    pattern_type="magic_hyperparameter",
                                    severity=PatternSeverity.MEDIUM,
                                    line_number=arg.lineno,
                                    column=arg.col_offset,
                                    message=f"Magic number {arg.value} in ML constructor",
                                    explanation=(
                                        "Hardcoded hyperparameters make experiments difficult to reproduce "
                                        "and hyperparameter tuning cumbersome."
                                    ),
                                    suggested_fix="Extract to configuration file or use named constants.",
                                    confidence=0.80,
                                    code_snippet=self._extract_code_snippet(analysis, arg.lineno),
                                    fix_snippet=self._generate_config_fix(arg.value)
                                ))

                    # Check keyword arguments
                    for keyword in node.keywords:
                        if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, (int, float)):
                            if self._is_magic_number(keyword.value.value, keyword.arg):  # Use context-aware detection
                                patterns.append(MLAntiPattern(
                                    pattern_type="magic_hyperparameter",
                                    severity=PatternSeverity.MEDIUM,
                                    line_number=keyword.value.lineno,
                                    column=keyword.value.col_offset,
                                    message=f"Magic number {keyword.value.value} in {keyword.arg} parameter",
                                    explanation=(
                                        "Hardcoded hyperparameters make experiments difficult to reproduce "
                                        "and hyperparameter tuning cumbersome."
                                    ),
                                    suggested_fix="Extract to configuration file or use named constants.",
                                    confidence=0.80,
                                    code_snippet=self._extract_code_snippet(analysis, keyword.value.lineno),
                                    fix_snippet=self._generate_config_fix(keyword.value.value)
                                ))

        return patterns

    def _detect_magic_dimensions(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Detect hardcoded tensor/array dimensions"""
        patterns = []

        # Look for reshape, view, linear layer definitions with magic numbers
        dimension_functions = ['reshape', 'view', 'Linear', 'Conv2d']

        for node in ast.walk(analysis.ast_tree):
            if (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Attribute) and
                node.func.attr in dimension_functions):

                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                        # Enhanced magic dimension detection with context awareness
                        if self._is_magic_dimension(arg.value, node.func.attr):  # Use context-aware detection
                            patterns.append(MLAntiPattern(
                                pattern_type="magic_dimension",
                                severity=PatternSeverity.LOW,
                                line_number=arg.lineno,
                                column=arg.col_offset,
                                message=f"Magic dimension {arg.value}",
                                explanation="Hardcoded dimensions make model architecture changes difficult.",
                                suggested_fix="Use named constants or configuration for dimensions.",
                                confidence=0.70,
                                code_snippet=self._extract_code_snippet(analysis, arg.lineno),
                                fix_snippet=f"HIDDEN_SIZE = {arg.value}  # Define as constant"
                            ))

        return patterns

    def _is_ml_constructor(self, node: ast.Call) -> bool:
        """Check if this is an ML model/optimizer constructor"""
        ml_constructors = {
            # Sklearn models
            'LinearRegression', 'RandomForestClassifier', 'SVC', 'LogisticRegression',
            'DecisionTreeClassifier', 'GradientBoostingClassifier', 'KNeighborsClassifier',
            # Sklearn utilities
            'train_test_split', 'cross_val_score', 'GridSearchCV', 'RandomizedSearchCV',
            # Neural network optimizers
            'Adam', 'SGD', 'AdamW', 'RMSprop',
            # PyTorch layers
            'Linear', 'Conv2d', 'LSTM', 'BatchNorm2d', 'Dropout',
            # TensorFlow/Keras
            'Dense', 'Conv2D', 'LSTM', 'Dropout', 'BatchNormalization'
        }

        if isinstance(node.func, ast.Name):
            return node.func.id in ml_constructors
        elif isinstance(node.func, ast.Attribute):
            # Check direct attribute (e.g., torch.Linear)
            if node.func.attr in ml_constructors:
                return True
            # Check for nn.* patterns (e.g., nn.Linear, nn.Conv2d)
            if (isinstance(node.func.value, ast.Name) and 
                node.func.value.id in ['nn', 'torch.nn'] and
                node.func.attr in ml_constructors):
                return True

        return False

    def _is_magic_number(self, value: Union[int, float], parameter_name: Optional[str] = None) -> bool:
        """Enhanced magic number detection with context awareness"""
        # Check if it's a standard ML value
        if value in self.standard_ml_values:
            return False
            
        # Check parameter-specific contexts
        if parameter_name and parameter_name in self.parameter_contexts:
            if value in self.parameter_contexts[parameter_name]:
                return False
                
        # Additional contextual rules
        if parameter_name:
            # Learning rates are typically small decimals
            if 'lr' in parameter_name.lower() or 'learning_rate' in parameter_name.lower():
                if 0.0001 <= value <= 0.1:
                    return False
                    
            # Batch sizes are typically powers of 2 or common values
            if 'batch' in parameter_name.lower():
                if value in {1, 8, 16, 32, 64, 128, 256, 512, 1024}:
                    return False
                    
            # Epochs/iterations are typically round numbers
            if any(keyword in parameter_name.lower() for keyword in ['epoch', 'iter', 'step']):
                if isinstance(value, int) and 1 <= value <= 10000:
                    return False
                    
            # Neural network layer dimensions
            if any(keyword in parameter_name.lower() for keyword in ['hidden', 'embed', 'dim', 'size']):
                if value in self.standard_nn_dimensions:
                    return False
                    
        # If none of the above exceptions apply, it's likely a magic number
        return True
    
    def _is_magic_dimension(self, value: int, context: str) -> bool:
        """Enhanced magic dimension detection with neural network context awareness"""
        # Common standard dimensions are never magic
        if value in self.standard_nn_dimensions:
            return False
            
        # Context-specific rules
        if context:
            # Linear layers: input/output dimensions
            if context == 'Linear':
                # Very common sizes in tutorials and standard architectures
                if value in {784, 10, 1000}:  # MNIST (28*28), CIFAR-10 classes, ImageNet classes
                    return False
                    
            # Convolutional layers: typically powers of 2 or small numbers
            if context in ['Conv2d', 'Conv1d']:
                if value in {1, 3, 5, 7, 11}:  # Common kernel sizes
                    return False
                    
        # Small dimensions (< 10) are typically not magic
        if value <= 10:
            return False
            
        # If it's a large dimension not in our whitelist, it's likely magic
        return True
    
    def _is_magic_number_in_constructor(self, value: Union[int, float]) -> bool:
        """Special magic number detection for constructor arguments (more lenient)"""
        # Standard ML values are never magic
        if value in self.standard_ml_values:
            return False
            
        # Neural network dimensions are never magic
        if isinstance(value, int) and value in self.standard_nn_dimensions:
            return False
            
        # Small integers (1-20) are typically legitimate in constructors
        if isinstance(value, int) and 1 <= value <= 20:
            return False
            
        # Common decimal values in ML
        if isinstance(value, float) and value in {0.01, 0.001, 0.1, 0.5, 0.2, 0.8}:
            return False
            
        # If none of the above exceptions apply, it's likely a magic number
        return True
    
    def _generate_config_fix(self, value: Union[int, float]) -> str:
        """Generate configuration file suggestion"""
        return f"""# config.yaml
model:
  hyperparameter: {value}

# In code:
config = load_config()
model = Model(config.model.hyperparameter)"""

    def _extract_code_snippet(self, analysis: ASTAnalysisResult, line_number: int) -> str:
        """Extract relevant code snippet for the pattern"""
        try:
            with open(analysis.file_path, 'r') as f:
                lines = f.readlines()
                if 0 <= line_number - 1 < len(lines):
                    return lines[line_number - 1].strip()
                return "Unable to extract code"
        except (FileNotFoundError, IndexError):
            return "Unable to extract code"


class ReproducibilityChecker:
    """
    Detects patterns that break reproducibility in ML experiments.

    Reproducibility is crucial for scientific validity and debugging.
    """

    def detect_patterns(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Detect reproducibility issues"""
        patterns = []

        patterns.extend(self._detect_missing_random_seeds(analysis))
        patterns.extend(self._detect_non_deterministic_operations(analysis))

        return patterns

    def _detect_missing_random_seeds(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Detect random operations without fixed seeds"""
        patterns = []

        random_functions = [
            'train_test_split', 'shuffle', 'sample',
            'RandomForestClassifier', 'random', 'rand', 'randn',
            # PyTorch random functions
            'randn', 'rand', 'randint', 'normal', 'uniform'
        ]

        # Check if there's a global seed set
        has_global_seed = self._has_global_random_seed(analysis)

        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in random_functions:
                    # Check if random_state parameter is provided
                    has_random_state = any(
                        isinstance(kw, ast.keyword) and
                        kw.arg in ['random_state', 'seed']
                        for kw in node.keywords
                    )

                    # For random functions that are covered by global seed, check if global seed is set
                    is_covered_by_global_seed = (
                        # numpy.random functions (np.random.*)
                        (isinstance(node.func, ast.Attribute) and
                         isinstance(node.func.value, ast.Attribute) and
                         node.func.value.attr == 'random') or
                        # torch functions (torch.randn, torch.rand, etc.)
                        (isinstance(node.func, ast.Attribute) and
                         isinstance(node.func.value, ast.Name) and
                         node.func.value.id == 'torch' and
                         func_name in ['randn', 'rand', 'randint', 'normal', 'uniform']) or
                        # sklearn functions that respect numpy global seed
                        func_name in ['shuffle', 'sample'] or
                        # direct numpy random calls after 'from numpy import random'
                        (isinstance(node.func, ast.Name) and 
                         func_name in ['rand', 'randn', 'choice'])
                    )

                    # Skip functions covered by global seed when global seed is set
                    if is_covered_by_global_seed and has_global_seed:
                        continue

                    if not has_random_state:
                        patterns.append(MLAntiPattern(
                            pattern_type="missing_random_seed",
                            severity=PatternSeverity.HIGH,
                            line_number=node.lineno,
                            column=node.col_offset,
                            message=f"{func_name}() missing random_state parameter",
                            explanation=(
                                "Random operations without fixed seeds make experiments non-reproducible. "
                                "This makes debugging and result validation impossible."
                            ),
                            suggested_fix=f"Add random_state=42 parameter to {func_name}()",
                            confidence=0.90,
                            code_snippet=self._extract_code_snippet(analysis, node.lineno),
                            fix_snippet=f"{func_name}(..., random_state=42)"
                        ))

        return patterns

    def _detect_non_deterministic_operations(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Detect operations that introduce non-determinism"""
        patterns = []

        for node in ast.walk(analysis.ast_tree):
            # Look for ThreadPoolExecutor, multiprocessing.Pool, etc.
            if isinstance(node, ast.withitem):
                if isinstance(node.context_expr, ast.Call):
                    if self._is_non_deterministic_context(node.context_expr):
                        patterns.append(MLAntiPattern(
                            pattern_type="non_deterministic_operation",
                            severity=PatternSeverity.MEDIUM,
                            line_number=node.context_expr.lineno,
                            column=node.context_expr.col_offset,
                            message="Potentially non-deterministic parallel operation",
                            explanation=(
                                "Parallel operations can introduce non-determinism unless "
                                "properly configured for reproducibility."
                            ),
                            suggested_fix="Ensure deterministic execution or document non-deterministic behavior.",
                            confidence=0.70,
                            code_snippet=self._extract_code_snippet(analysis, node.context_expr.lineno),
                            fix_snippet="# Add deterministic=True or equivalent parameter"
                        ))

            # Look for multiprocessing.Pool() calls
            elif isinstance(node, ast.Call):
                if self._is_non_deterministic_call(node):
                    patterns.append(MLAntiPattern(
                        pattern_type="non_deterministic_operation",
                        severity=PatternSeverity.MEDIUM,
                        line_number=node.lineno,
                        column=node.col_offset,
                        message="Potentially non-deterministic parallel operation",
                        explanation=(
                            "Parallel operations can introduce non-determinism unless "
                            "properly configured for reproducibility."
                        ),
                        suggested_fix="Ensure deterministic execution or document non-deterministic behavior.",
                        confidence=0.70,
                        code_snippet=self._extract_code_snippet(analysis, node.lineno),
                        fix_snippet="# Add deterministic=True or equivalent parameter"
                    ))

        return patterns

    def _has_global_random_seed(self, analysis: ASTAnalysisResult) -> bool:
        """Check if a global random seed is set"""
        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Call):
                # Check for np.random.seed(), random.seed(), torch.manual_seed()
                if isinstance(node.func, ast.Attribute):
                    # np.random.seed() pattern
                    if (node.func.attr == 'seed' and
                        isinstance(node.func.value, ast.Attribute) and
                        node.func.value.attr == 'random'):
                        return True
                    # torch.manual_seed() pattern
                    elif (node.func.attr == 'manual_seed' and
                          isinstance(node.func.value, ast.Name) and
                          node.func.value.id == 'torch'):
                        return True
                    # random.seed() pattern
                    elif (node.func.attr == 'seed' and
                          isinstance(node.func.value, ast.Name) and
                          node.func.value.id == 'random'):
                        return True
        return False

    def _is_non_deterministic_context(self, node: ast.Call) -> bool:
        """Check if this is a non-deterministic context manager"""
        if isinstance(node.func, ast.Name):
            return node.func.id in ['ThreadPoolExecutor', 'ProcessPoolExecutor']
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr in ['ThreadPoolExecutor', 'ProcessPoolExecutor']
        return False

    def _is_non_deterministic_call(self, node: ast.Call) -> bool:
        """Check if this is a non-deterministic function call"""
        if isinstance(node.func, ast.Attribute):
            # multiprocessing.Pool()
            if (node.func.attr == 'Pool' and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'multiprocessing'):
                return True
        return False

    def _extract_code_snippet(self, analysis: ASTAnalysisResult, line_number: int) -> str:
        """Extract code snippet for display"""
        try:
            with open(analysis.file_path, 'r') as f:
                lines = f.readlines()
                if 0 <= line_number - 1 < len(lines):
                    return lines[line_number - 1].strip()
        except:
            pass
        return "Unable to extract code snippet"


class TestSetContaminationDetector:
    """
    Detects test set contamination patterns - the core Sprint 3 detector.
    
    Test set contamination is one of the most serious issues in ML research,
    leading to overly optimistic results that don't generalize. This detector
    uses sophisticated semantic analysis to identify subtle contamination patterns.
    
    Detection capabilities:
    1. Exact duplicate detection between train/test sets
    2. Feature leakage (using future information)
    3. Temporal leakage in time series data
    4. Preprocessing leakage (fit before split)
    5. Cross-validation contamination
    """
    
    def __init__(self):
        # Track variable assignments and data transformations
        self.variable_tracker = VariableSemanticTracker()
        
        # Split function patterns
        self.split_functions = {
            'train_test_split', 'TimeSeriesSplit', 'KFold', 'StratifiedKFold',
            'GroupKFold', 'LeaveOneOut', 'LeavePOut', 'StratifiedGroupKFold'
        }
        
        # Preprocessing classes that can cause leakage
        self.preprocessing_classes = {
            'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'PowerTransformer',
            'QuantileTransformer', 'Normalizer', 'LabelEncoder', 'OneHotEncoder',
            'OrdinalEncoder', 'TargetEncoder', 'WOEEncoder'
        }
        
        # Methods that fit on data (potential leakage sources)
        self.fit_methods = {'fit', 'fit_transform'}
        
        # Suspicious feature patterns that might indicate leakage
        self.leakage_indicators = {
            'target', 'label', 'outcome', 'result', 'future', 'next',
            'tomorrow', 'later', 'after', 'post', 'subsequent'
        }
        
        # Temporal operations that can cause leakage
        self.temporal_operations = {
            'shift', 'rolling', 'expanding', 'resample', 'lag', 'lead',
            'diff', 'pct_change', 'cumsum', 'cummax', 'cummin'
        }
    
    def detect_patterns(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Main entry point for test set contamination detection"""
        patterns = []
        
        # Reset variable tracker for each file
        self.variable_tracker.reset()
        
        # Build semantic model of the code
        self._build_semantic_model(analysis)
        
        # Apply detection algorithms in order of complexity
        patterns.extend(self._detect_exact_duplicates(analysis))
        patterns.extend(self._detect_preprocessing_leakage(analysis))
        patterns.extend(self._detect_feature_leakage(analysis))
        patterns.extend(self._detect_temporal_leakage(analysis))
        patterns.extend(self._detect_cv_contamination(analysis))
        
        return patterns
    
    def _build_semantic_model(self, analysis: ASTAnalysisResult):
        """Build a semantic model of variable flows and transformations"""
        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Assign):
                self._track_assignment(node)
            elif isinstance(node, ast.Call):
                self._track_function_call(node)
    
    def _track_assignment(self, node: ast.Assign):
        """Track variable assignments for semantic analysis"""
        for target in node.targets:
            if isinstance(target, (ast.Tuple, ast.List)):
                # Handle tuple unpacking (e.g., X_train, X_test, y_train, y_test = train_test_split(...))
                self._track_tuple_unpacking(target, node.value)
            elif isinstance(target, ast.Name):
                # Simple assignment
                self.variable_tracker.assign_variable(target.id, node.value, node.lineno)
    
    def _track_tuple_unpacking(self, target: Union[ast.Tuple, ast.List], value: ast.AST):
        """Track tuple unpacking assignments like train_test_split returns"""
        if isinstance(value, ast.Call) and self._is_split_function(value):
            # This is likely a train_test_split call
            elements = target.elts
            if len(elements) >= 4:  # X_train, X_test, y_train, y_test
                for i, elt in enumerate(elements):
                    if isinstance(elt, ast.Name):
                        var_name = elt.id
                        var_type = self._infer_split_variable_type(var_name, i)
                        self.variable_tracker.assign_split_variable(
                            var_name, var_type, value, elt.lineno if hasattr(elt, 'lineno') else 0
                        )
    
    def _track_function_call(self, node: ast.Call):
        """Track function calls that might be relevant for contamination"""
        func_name = self._get_function_name(node)
        
        if func_name in self.preprocessing_classes:
            # Track preprocessing object creation
            self.variable_tracker.track_preprocessor_creation(func_name, node)
        elif self._is_fit_method_call(node):
            # Track fit/fit_transform calls
            self.variable_tracker.track_fit_operation(node)
    
    def _detect_exact_duplicates(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """
        Detect exact duplicates between train and test sets.
        
        This is the simplest but most critical contamination pattern.
        """
        patterns = []
        
        # Look for set intersection operations or duplicate checking
        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Call):
                # Check for set operations on train/test data
                if self._is_set_intersection_on_splits(node):
                    patterns.append(self._create_duplicate_detection_pattern(node, analysis))
                
                # Check for pandas merge/join operations between train and test
                elif self._is_cross_split_merge(node):
                    patterns.append(self._create_cross_merge_pattern(node, analysis))
        
        # Check for missing duplicate detection
        split_vars = self.variable_tracker.get_split_variables()
        if split_vars and not self._has_duplicate_check(analysis):
            # Find the split operation line
            split_line = self._find_split_operation_line(analysis)
            if split_line:
                patterns.append(self._create_missing_duplicate_check_pattern(
                    analysis, split_line
                ))
        
        return patterns
    
    def _detect_preprocessing_leakage(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """
        Detect preprocessing applied before train/test split.
        
        This extends the existing data leakage detector with more sophisticated
        variable tracking and contamination detection.
        """
        patterns = []
        
        # Get chronological order of operations
        operations = self.variable_tracker.get_chronological_operations()
        
        # Find preprocessing operations that happen before splits
        for i, op in enumerate(operations):
            if op['type'] == 'preprocessing_fit' and op['method'] in self.fit_methods:
                # Look for subsequent split operations
                for j in range(i + 1, len(operations)):
                    if operations[j]['type'] == 'data_split':
                        # Check if the same data is involved
                        if self._affects_same_data(op, operations[j]):
                            patterns.append(self._create_preprocessing_leakage_pattern(
                                op, operations[j], analysis
                            ))
                        break
        
        return patterns
    
    def _detect_feature_leakage(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """
        Detect features that contain information from the future or target.
        
        This is the most sophisticated detection requiring semantic understanding.
        """
        patterns = []
        
        # Analyze feature engineering operations
        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Assign):
                # Check if this creates a feature with potential leakage
                if self._creates_leaky_feature(node):
                    patterns.append(self._create_feature_leakage_pattern(node, analysis))
        
        # Check for target-derived features
        target_vars = self.variable_tracker.get_target_variables()
        for var_name, var_info in self.variable_tracker.get_all_variables().items():
            if self._is_derived_from_target(var_info, target_vars):
                patterns.append(self._create_target_leakage_pattern(var_info, analysis))
        
        return patterns
    
    def _detect_temporal_leakage(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """
        Detect temporal leakage in time series data.
        
        Uses semantic analysis to detect look-ahead bias and improper temporal splits.
        """
        patterns = []
        
        # Check for temporal operations that might cause leakage
        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node)
                if any(temp_op in func_name for temp_op in self.temporal_operations):
                    # Check if this operation might cause future leakage
                    if self._causes_temporal_leakage(node):
                        patterns.append(self._create_temporal_leakage_pattern(node, analysis))
        
        # Check for incorrect temporal splits (random instead of chronological)
        split_operations = self.variable_tracker.get_split_operations()
        for split_op in split_operations:
            if self._is_temporal_data(split_op) and not self._uses_temporal_split(split_op):
                patterns.append(self._create_incorrect_temporal_split_pattern(split_op, analysis))
        
        return patterns
    
    def _detect_cv_contamination(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """
        Detect contamination in cross-validation setups.
        
        Identifies preprocessing outside of CV folds and other CV contamination.
        """
        patterns = []
        
        # Look for cross-validation patterns
        cv_operations = self._find_cv_operations(analysis)
        
        for cv_op in cv_operations:
            # Check if preprocessing happens outside CV folds
            if self._has_preprocessing_outside_cv(cv_op, analysis):
                patterns.append(self._create_cv_preprocessing_leakage_pattern(cv_op, analysis))
            
            # Check for target leakage in feature selection within CV
            if self._has_feature_selection_leakage_in_cv(cv_op, analysis):
                patterns.append(self._create_cv_feature_selection_leakage_pattern(cv_op, analysis))
        
        return patterns
    
    def _is_split_function(self, node: ast.Call) -> bool:
        """Check if a function call is a data splitting function"""
        func_name = self._get_function_name(node)
        return func_name in self.split_functions
    
    def _get_function_name(self, node: ast.Call) -> str:
        """Extract function name from call node"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""
    
    def _infer_split_variable_type(self, var_name: str, position: int) -> str:
        """Infer the type of variable from train_test_split unpacking"""
        var_lower = var_name.lower()
        
        if 'train' in var_lower:
            return 'train'
        elif 'test' in var_lower or 'val' in var_lower:
            return 'test'
        elif position in [0, 2]:  # Typically X_train, y_train positions
            return 'train'
        elif position in [1, 3]:  # Typically X_test, y_test positions
            return 'test'
        else:
            return 'unknown'
    
    def _is_set_intersection_on_splits(self, node: ast.Call) -> bool:
        """Check if this is a set intersection operation on train/test splits"""
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'intersection':
            # Check if operands are train/test variables
            return self._involves_split_variables(node)
        return False
    
    def _is_cross_split_merge(self, node: ast.Call) -> bool:
        """Check if this merges train and test data inappropriately"""
        func_name = self._get_function_name(node)
        if func_name in ['merge', 'join', 'concat']:
            return self._involves_multiple_splits(node)
        return False
    
    def _involves_split_variables(self, node: ast.Call) -> bool:
        """Check if a function call involves train/test split variables"""
        split_vars = self.variable_tracker.get_split_variables()
        
        for arg in ast.walk(node):
            if isinstance(arg, ast.Name) and arg.id in split_vars:
                return True
        return False
    
    def _involves_multiple_splits(self, node: ast.Call) -> bool:
        """Check if operation involves both train and test data"""
        train_vars = self.variable_tracker.get_variables_by_type('train')
        test_vars = self.variable_tracker.get_variables_by_type('test')
        
        has_train = False
        has_test = False
        
        for arg in ast.walk(node):
            if isinstance(arg, ast.Name):
                if arg.id in train_vars:
                    has_train = True
                elif arg.id in test_vars:
                    has_test = True
        
        return has_train and has_test
    
    def _has_duplicate_check(self, analysis: ASTAnalysisResult) -> bool:
        """Check if code contains duplicate detection between train/test"""
        # Look for set operations, duplicated() calls, or similar
        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node)
                if func_name in ['intersection', 'duplicated', 'drop_duplicates']:
                    if self._involves_split_variables(node):
                        return True
        return False
    
    def _find_split_operation_line(self, analysis: ASTAnalysisResult) -> Optional[int]:
        """Find the line number where train/test split occurs"""
        split_ops = self.variable_tracker.get_split_operations()
        if split_ops:
            return split_ops[0].get('line_number')
        return None
    
    def _affects_same_data(self, preprocessing_op: Dict, split_op: Dict) -> bool:
        """Check if preprocessing and split operations affect the same data"""
        # This would need more sophisticated analysis in practice
        return True  # Conservative approach
    
    def _creates_leaky_feature(self, node: ast.Assign) -> bool:
        """Check if assignment creates a potentially leaky feature"""
        # Check variable names for leakage indicators
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id.lower()
                if any(indicator in var_name for indicator in self.leakage_indicators):
                    return True
        
        # Check if the assignment uses suspicious operations
        if isinstance(node.value, ast.Call):
            func_name = self._get_function_name(node.value)
            if func_name in ['shift', 'rolling'] and self._has_positive_shift(node.value):
                return True
        
        return False
    
    def _has_positive_shift(self, node: ast.Call) -> bool:
        """Check if shift operation has positive periods (future data)"""
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                if arg.value > 0:  # Positive shift = future data
                    return True
        return False
    
    def _is_derived_from_target(self, var_info: Dict, target_vars: Set[str]) -> bool:
        """Check if variable is derived from target variables"""
        # Simplified check - would need more sophisticated dependency analysis
        return False
    
    def _causes_temporal_leakage(self, node: ast.Call) -> bool:
        """Check if temporal operation causes future leakage"""
        # Check for rolling operations without proper window constraints
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'rolling':
            # Check if window parameter is provided and reasonable
            return not self._has_safe_rolling_window(node)
        return False
    
    def _has_safe_rolling_window(self, node: ast.Call) -> bool:
        """Check if rolling operation has safe window parameters"""
        # Look for window parameter
        for keyword in node.keywords:
            if keyword.arg == 'window':
                return True
        return False
    
    def _is_temporal_data(self, split_op: Dict) -> bool:
        """Check if split operation involves temporal data"""
        # Look for datetime columns or temporal indicators
        return False  # Simplified for now
    
    def _uses_temporal_split(self, split_op: Dict) -> bool:
        """Check if split uses temporal-aware splitting"""
        func_name = split_op.get('function', '')
        return 'TimeSeries' in func_name or 'temporal' in func_name.lower()
    
    def _find_cv_operations(self, analysis: ASTAnalysisResult) -> List[Dict]:
        """Find cross-validation operations in the code"""
        cv_ops = []
        cv_functions = ['cross_val_score', 'cross_validate', 'GridSearchCV', 'RandomizedSearchCV']
        
        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node)
                if func_name in cv_functions:
                    cv_ops.append({
                        'function': func_name,
                        'node': node,
                        'line': node.lineno
                    })
        
        return cv_ops
    
    def _has_preprocessing_outside_cv(self, cv_op: Dict, analysis: ASTAnalysisResult) -> bool:
        """Check if preprocessing happens outside CV folds"""
        # Look for preprocessing operations before the CV operation
        cv_line = cv_op['line']
        operations = self.variable_tracker.get_chronological_operations()
        
        for op in operations:
            if op['line'] < cv_line and op['type'] == 'preprocessing_fit':
                return True
        return False
    
    def _has_feature_selection_leakage_in_cv(self, cv_op: Dict, analysis: ASTAnalysisResult) -> bool:
        """Check for feature selection leakage within CV"""
        # This would need sophisticated analysis of CV pipeline
        return False
    
    def _is_fit_method_call(self, node: ast.Call) -> bool:
        """Check if this is a fit method call on a preprocessor"""
        if isinstance(node.func, ast.Attribute):
            return node.func.attr in self.fit_methods
        return False
    
    # Pattern creation methods
    def _create_duplicate_detection_pattern(self, node: ast.Call, analysis: ASTAnalysisResult) -> MLAntiPattern:
        """Create pattern for duplicate detection between train/test"""
        return MLAntiPattern(
            pattern_type="test_set_exact_duplicates",
            severity=PatternSeverity.CRITICAL,
            line_number=node.lineno,
            column=node.col_offset,
            message="Exact duplicates detected between train and test sets",
            explanation=(
                "Identical samples in both training and test sets cause severe overfitting. "
                "The model has already seen the test data during training, making evaluation invalid."
            ),
            suggested_fix="Remove duplicates before splitting or use temporal/group-based splits",
            confidence=0.95,
            code_snippet=self._extract_code_snippet(analysis, node.lineno),
            fix_snippet="""# Remove duplicates before splitting
X_clean = X.drop_duplicates()
y_clean = y[X_clean.index]
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, random_state=42)""",
            references=[
                "https://scikit-learn.org/stable/common_pitfalls.html#data-leakage",
                "https://machinelearningmastery.com/data-leakage-machine-learning/"
            ]
        )
    
    def _create_missing_duplicate_check_pattern(self, analysis: ASTAnalysisResult, split_line: int) -> MLAntiPattern:
        """Create pattern for missing duplicate check"""
        return MLAntiPattern(
            pattern_type="missing_duplicate_check",
            severity=PatternSeverity.HIGH,
            line_number=split_line,
            column=0,
            message="No duplicate detection between train and test sets",
            explanation=(
                "Without checking for duplicates between train and test sets, "
                "you risk contamination that leads to overly optimistic results."
            ),
            suggested_fix="Add duplicate detection before evaluation",
            confidence=0.80,
            code_snippet=self._extract_code_snippet(analysis, split_line),
            fix_snippet="""# Check for contamination
train_samples = set(X_train.apply(tuple, axis=1))
test_samples = set(X_test.apply(tuple, axis=1))
contamination = train_samples.intersection(test_samples)
print(f"Found {len(contamination)} duplicates between train and test")""",
            references=[
                "https://scikit-learn.org/stable/common_pitfalls.html",
                "https://developers.google.com/machine-learning/guides/rules-of-ml"
            ]
        )
    
    def _extract_code_snippet(self, analysis: ASTAnalysisResult, line_number: int) -> str:
        """Extract relevant code snippet for the pattern"""
        try:
            with open(analysis.file_path, 'r') as f:
                lines = f.readlines()
                if 0 <= line_number - 1 < len(lines):
                    return lines[line_number - 1].strip()
        except:
            pass
        return "Unable to extract code snippet"
    
    def _create_cross_merge_pattern(self, node: ast.Call, analysis: ASTAnalysisResult) -> MLAntiPattern:
        """Create pattern for inappropriate merging of train/test data"""
        return MLAntiPattern(
            pattern_type="cross_split_merge",
            severity=PatternSeverity.CRITICAL,
            line_number=node.lineno,
            column=node.col_offset,
            message="Inappropriate merge/join between train and test data",
            explanation=(
                "Merging or joining train and test data can cause contamination. "
                "Train and test sets should remain completely separate."
            ),
            suggested_fix="Keep train and test data separate. Use validation set for model selection.",
            confidence=0.90,
            code_snippet=self._extract_code_snippet(analysis, node.lineno),
            fix_snippet="# Keep data separate\n# Only merge within train or within test, never across",
            references=[
                "https://scikit-learn.org/stable/common_pitfalls.html#data-leakage"
            ]
        )
    
    def _create_preprocessing_leakage_pattern(self, preprocessing_op: Dict, split_op: Dict, analysis: ASTAnalysisResult) -> MLAntiPattern:
        """Create pattern for preprocessing applied before split"""
        return MLAntiPattern(
            pattern_type="preprocessing_before_split",
            severity=PatternSeverity.CRITICAL,
            line_number=preprocessing_op.get('line', 0),
            column=0,
            message=f"Preprocessing {preprocessing_op.get('method', 'operation')} applied before train/test split",
            explanation=(
                "Applying preprocessing to the entire dataset before splitting causes data leakage. "
                "The preprocessor learns statistics from the test set, leading to overly optimistic results."
            ),
            suggested_fix="Move preprocessing after train_test_split. Fit only on training data.",
            confidence=0.95,
            code_snippet=self._extract_code_snippet(analysis, preprocessing_op.get('line', 0)),
            fix_snippet="""# Correct approach:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training data
X_test_scaled = scaler.transform(X_test)  # Only transform test data""",
            references=[
                "https://scikit-learn.org/stable/common_pitfalls.html#data-leakage",
                "https://machinelearningmastery.com/data-leakage-machine-learning/"
            ]
        )
    
    def _create_feature_leakage_pattern(self, node: ast.Assign, analysis: ASTAnalysisResult) -> MLAntiPattern:
        """Create pattern for feature leakage"""
        return MLAntiPattern(
            pattern_type="feature_leakage",
            severity=PatternSeverity.HIGH,
            line_number=node.lineno,
            column=node.col_offset,
            message="Potential feature leakage detected",
            explanation=(
                "Feature appears to use information that wouldn't be available at prediction time. "
                "This can lead to overly optimistic model performance."
            ),
            suggested_fix="Ensure features only use information available at prediction time.",
            confidence=0.80,
            code_snippet=self._extract_code_snippet(analysis, node.lineno),
            fix_snippet="# Use only historical data for features\n# Avoid future information or target-derived features",
            references=[
                "https://machinelearningmastery.com/data-leakage-machine-learning/"
            ]
        )
    
    def _create_target_leakage_pattern(self, var_info: Dict, analysis: ASTAnalysisResult) -> MLAntiPattern:
        """Create pattern for target leakage"""
        return MLAntiPattern(
            pattern_type="target_leakage",
            severity=PatternSeverity.CRITICAL,
            line_number=var_info.get('line', 0),
            column=0,
            message="Feature derived from target variable",
            explanation=(
                "Features derived from the target variable cause severe data leakage. "
                "The model learns to cheat by using the answer to predict itself."
            ),
            suggested_fix="Remove target-derived features from the feature set.",
            confidence=0.90,
            code_snippet=self._extract_code_snippet(analysis, var_info.get('line', 0)),
            fix_snippet="# Remove features that use target information",
            references=[
                "https://machinelearningmastery.com/data-leakage-machine-learning/"
            ]
        )
    
    def _create_temporal_leakage_pattern(self, node: ast.Call, analysis: ASTAnalysisResult) -> MLAntiPattern:
        """Create pattern for temporal leakage"""
        return MLAntiPattern(
            pattern_type="temporal_leakage",
            severity=PatternSeverity.HIGH,
            line_number=node.lineno,
            column=node.col_offset,
            message="Potential temporal leakage in time series operation",
            explanation=(
                "Time series operation might be using future information. "
                "This creates look-ahead bias in time series models."
            ),
            suggested_fix="Use only past data for time series features. Check window parameters.",
            confidence=0.75,
            code_snippet=self._extract_code_snippet(analysis, node.lineno),
            fix_snippet="""# Use negative shifts for historical data:
df['lag_1'] = df['value'].shift(1)  # Previous value (safe)
# Avoid positive shifts: df['future'] = df['value'].shift(-1)  # Future value (dangerous!)""",
            references=[
                "https://otexts.com/fpp3/",
                "https://machinelearningmastery.com/time-series-forecasting-supervised-learning/"
            ]
        )
    
    def _create_incorrect_temporal_split_pattern(self, split_op: Dict, analysis: ASTAnalysisResult) -> MLAntiPattern:
        """Create pattern for incorrect temporal splitting"""
        return MLAntiPattern(
            pattern_type="incorrect_temporal_split",
            severity=PatternSeverity.HIGH,
            line_number=split_op.get('line', 0),
            column=0,
            message="Random split used on temporal data",
            explanation=(
                "Using random train/test split on time series data causes temporal leakage. "
                "Future data ends up in training set, allowing model to 'see the future'."
            ),
            suggested_fix="Use temporal split: train on past data, test on future data.",
            confidence=0.85,
            code_snippet=self._extract_code_snippet(analysis, split_op.get('line', 0)),
            fix_snippet="""# Temporal split for time series:
split_point = int(0.8 * len(data))
train_data = data[:split_point]  # Past data
test_data = data[split_point:]   # Future data""",
            references=[
                "https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split"
            ]
        )
    
    def _create_cv_preprocessing_leakage_pattern(self, cv_op: Dict, analysis: ASTAnalysisResult) -> MLAntiPattern:
        """Create pattern for preprocessing leakage in cross-validation"""
        return MLAntiPattern(
            pattern_type="cv_preprocessing_leakage",
            severity=PatternSeverity.HIGH,
            line_number=cv_op.get('line', 0),
            column=0,
            message="Preprocessing applied outside cross-validation folds",
            explanation=(
                "Preprocessing before cross-validation causes data leakage. "
                "Each CV fold sees preprocessed data from other folds."
            ),
            suggested_fix="Use Pipeline to include preprocessing within CV folds.",
            confidence=0.90,
            code_snippet=self._extract_code_snippet(analysis, cv_op.get('line', 0)),
            fix_snippet="""# Correct CV with Pipeline:
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
cross_val_score(pipeline, X, y, cv=5)  # Preprocessing happens within each fold""",
            references=[
                "https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-pipelines"
            ]
        )
    
    def _create_cv_feature_selection_leakage_pattern(self, cv_op: Dict, analysis: ASTAnalysisResult) -> MLAntiPattern:
        """Create pattern for feature selection leakage in CV"""
        return MLAntiPattern(
            pattern_type="cv_feature_selection_leakage",
            severity=PatternSeverity.MEDIUM,
            line_number=cv_op.get('line', 0),
            column=0,
            message="Feature selection may be leaking information in cross-validation",
            explanation=(
                "Feature selection before CV can cause subtle leakage. "
                "Features are selected using the entire dataset including test folds."
            ),
            suggested_fix="Include feature selection within CV pipeline.",
            confidence=0.70,
            code_snippet=self._extract_code_snippet(analysis, cv_op.get('line', 0)),
            fix_snippet="""# Include feature selection in pipeline:
pipeline = Pipeline([
    ('feature_selection', SelectKBest(k=10)),
    ('classifier', RandomForestClassifier())
])""",
            references=[
                "https://scikit-learn.org/stable/modules/cross_validation.html"
            ]
        )


class VariableSemanticTracker:
    """
    Tracks variable assignments and transformations for semantic analysis.
    
    This class builds a semantic model of how data flows through the code,
    enabling sophisticated contamination detection.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset tracking state for new analysis"""
        self.variables = {}  # var_name -> variable info
        self.split_variables = {}  # var_name -> split type (train/test)
        self.operations = []  # chronological list of operations
        self.preprocessors = {}  # preprocessor instances
    
    def assign_variable(self, var_name: str, value: ast.AST, line_number: int):
        """Track a variable assignment"""
        self.variables[var_name] = {
            'value': value,
            'line': line_number,
            'type': self._infer_variable_type(value)
        }
    
    def assign_split_variable(self, var_name: str, split_type: str, split_call: ast.Call, line_number: int):
        """Track variables created by train/test split"""
        self.split_variables[var_name] = split_type
        self.variables[var_name] = {
            'value': split_call,
            'line': line_number,
            'type': f'split_{split_type}',
            'split_type': split_type
        }
        
        # Record the split operation
        self.operations.append({
            'type': 'data_split',
            'function': self._get_function_name(split_call),
            'line': line_number,
            'variables': [var_name]
        })
    
    def track_preprocessor_creation(self, class_name: str, node: ast.Call):
        """Track creation of preprocessor objects"""
        # This would track assignments to preprocessor instances
        pass
    
    def track_fit_operation(self, node: ast.Call):
        """Track fit/fit_transform operations"""
        self.operations.append({
            'type': 'preprocessing_fit',
            'method': self._get_method_name(node),
            'line': node.lineno,
            'node': node
        })
    
    def get_split_variables(self) -> Dict[str, str]:
        """Get all variables created by train/test splits"""
        return self.split_variables.copy()
    
    def get_variables_by_type(self, var_type: str) -> Set[str]:
        """Get variables of a specific type"""
        return {name for name, info in self.split_variables.items() if info == var_type}
    
    def get_target_variables(self) -> Set[str]:
        """Get variables that appear to be targets/labels"""
        target_vars = set()
        for var_name in self.variables:
            if any(indicator in var_name.lower() for indicator in ['target', 'label', 'y']):
                target_vars.add(var_name)
        return target_vars
    
    def get_chronological_operations(self) -> List[Dict]:
        """Get operations in chronological order"""
        return sorted(self.operations, key=lambda x: x.get('line', 0))
    
    def get_split_operations(self) -> List[Dict]:
        """Get all split operations"""
        return [op for op in self.operations if op['type'] == 'data_split']
    
    def get_all_variables(self) -> Dict[str, Dict]:
        """Get all tracked variables"""
        return self.variables.copy()
    
    def _infer_variable_type(self, value: ast.AST) -> str:
        """Infer the type of a variable from its assignment"""
        if isinstance(value, ast.Call):
            func_name = self._get_function_name(value)
            if 'read_' in func_name:
                return 'dataframe'
            elif func_name in ['array', 'zeros', 'ones']:
                return 'array'
        return 'unknown'
    
    def _get_function_name(self, node: ast.Call) -> str:
        """Get function name from call node"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""
    
    def _get_method_name(self, node: ast.Call) -> str:
        """Get method name from call node"""
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""


class MLPatternDetector:
    """
    Main orchestrator for all ML anti-pattern detection.

    This class coordinates all the specialized detectors and provides
    a unified interface for pattern detection.
    """

    def __init__(self):
        self.detectors = {
            'data_leakage': DataLeakageDetector(),
            'test_contamination': TestSetContaminationDetector(),  # Added Sprint 3 detector
            'gpu_memory': GPUMemoryLeakDetector(),
            'magic_numbers': MagicNumberExtractor(),
            'reproducibility': ReproducibilityChecker(),
            'hardcoded_thresholds': HardcodedThresholdsDetector(),
            'inefficient_data_loading': InefficientDataLoadingDetector(),
        }

    def detect_all_patterns(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Run all pattern detectors and return combined results"""
        all_patterns = []

        for detector_name, detector in self.detectors.items():
            try:
                patterns = detector.detect_patterns(analysis)
                all_patterns.extend(patterns)
            except Exception as e:
                # Log error but continue with other detectors
                print(f"Error in {detector_name} detector: {e}")

        # Sort by severity and confidence
        all_patterns.sort(
            key=lambda p: (p.severity.value, -p.confidence),
            reverse=True
        )

        return all_patterns

    def get_pattern_summary(self, patterns: List[MLAntiPattern]) -> Dict[str, Any]:
        """Get summary statistics of detected patterns"""
        summary = {
            'total_patterns': len(patterns),
            'by_severity': {},
            'by_type': {},
            'high_confidence_count': 0,
            'avg_confidence': 0.0
        }

        if not patterns:
            return summary

        # Count by severity
        for severity in PatternSeverity:
            summary['by_severity'][severity.value] = sum(
                1 for p in patterns if p.severity == severity
            )

        # Count by type
        for pattern in patterns:
            pattern_type = pattern.pattern_type
            summary['by_type'][pattern_type] = summary['by_type'].get(pattern_type, 0) + 1

        # High confidence patterns (>0.8)
        summary['high_confidence_count'] = sum(
            1 for p in patterns if p.confidence > 0.8
        )

        # Average confidence
        summary['avg_confidence'] = sum(p.confidence for p in patterns) / len(patterns)

        return summary


class HardcodedThresholdsDetector:
    """
    Detects hardcoded threshold values without proper justification.
    
    This detector identifies magic numbers used as thresholds in ML code
    and distinguishes between arbitrary values and well-documented constants.
    
    Key patterns detected:
    - Magic number thresholds (e.g., threshold = 0.73625)
    - Overly precise decimal values without context
    - Threshold assignments without business logic
    - Missing documentation for critical decision boundaries
    """
    
    def __init__(self):
        # Common ML threshold variable patterns
        self.threshold_patterns = {
            'threshold', 'cutoff', 'limit', 'boundary', 'barrier',
            'confidence', 'probability', 'score', 'minimum', 'maximum'
        }
        
        # Suspicious decimal patterns (too precise, arbitrary)
        self.suspicious_precision_regex = re.compile(r'0\.\d{4,}')  # 4+ decimal places
        
        # Well-known statistical/business thresholds (should be ignored)
        self.known_good_thresholds = {
            0.5, 0.05, 0.01, 0.001,  # Statistical significance levels
            0.8, 0.85, 0.9, 0.95,    # Common business performance targets
            0.1, 0.2, 0.3, 0.7,      # Round training/test split ratios
            1.0, 2.0, 3.0,           # Integer thresholds
        }
        
        # Business context indicators (good constants)
        self.business_context_keywords = {
            'business', 'requirement', 'target', 'goal', 'policy',
            'acceptable', 'minimum', 'maximum', 'standard', 'regulation',
            'precision_requirement', 'recall_target', 'accuracy_goal'
        }
    
    def detect_patterns(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Detect hardcoded threshold anti-patterns in the code"""
        patterns = []
        
        # Analyze all variable assignments
        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Assign):
                patterns.extend(self._analyze_assignment(node, analysis))
            elif isinstance(node, ast.Compare):
                patterns.extend(self._analyze_comparison(node, analysis))
        
        return patterns
    
    def _analyze_assignment(self, node: ast.Assign, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Analyze variable assignments for hardcoded thresholds"""
        patterns = []
        
        # Check if this looks like a threshold assignment
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                var_name_lower = var_name.lower()
                
                # Check if variable name suggests a threshold
                if any(pattern in var_name_lower for pattern in self.threshold_patterns):
                    # Check the assigned value
                    if isinstance(node.value, ast.Constant):
                        value = node.value.value
                        
                        if isinstance(value, (int, float)):
                            suspicion_level = self._assess_threshold_suspicion(
                                var_name_lower, value, node, analysis
                            )
                            
                            if suspicion_level > 0.7:  # High suspicion threshold
                                patterns.append(self._create_hardcoded_threshold_pattern(
                                    node, analysis, var_name, value, suspicion_level
                                ))
        
        return patterns
    
    def _analyze_comparison(self, node: ast.Compare, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Analyze direct comparisons with magic numbers"""
        patterns = []
        
        # Look for comparisons with suspicious numeric literals
        for comparator in node.comparators:
            if isinstance(comparator, ast.Constant) and isinstance(comparator.value, (int, float)):
                value = comparator.value
                
                # Check if this is a suspicious magic number in comparison
                if self._is_suspicious_magic_number(value):
                    # Check context - is this in an ML-related comparison?
                    if self._is_ml_context_comparison(node, analysis):
                        patterns.append(self._create_magic_comparison_pattern(
                            node, analysis, value
                        ))
        
        return patterns
    
    def _assess_threshold_suspicion(self, var_name: str, value: float, 
                                   node: ast.Assign, analysis: ASTAnalysisResult) -> float:
        """
        Assess how suspicious a threshold assignment is (0.0 = clean, 1.0 = very suspicious)
        """
        suspicion = 0.0
        
        # Factor 1: Check if value is a known good threshold
        if value in self.known_good_thresholds:
            return 0.0  # Not suspicious at all
        
        # Factor 2: Check precision level
        if self.suspicious_precision_regex.match(str(value)):
            suspicion += 0.5  # Very precise decimals are suspicious
        
        # Factor 3: Check for business context in variable name
        if any(keyword in var_name for keyword in self.business_context_keywords):
            suspicion -= 0.3  # Business context reduces suspicion
        
        # Factor 4: Check for documentation/comments near assignment
        if self._has_nearby_documentation(node, analysis):
            suspicion -= 0.3  # Documentation reduces suspicion
        
        # Factor 5: Check if value is calculated vs hardcoded
        if self._is_calculated_value(node):
            suspicion -= 0.4  # Calculated values are less suspicious
        
        # Factor 6: Range analysis - values between 0 and 1 are more likely to be thresholds
        if 0.0 < value < 1.0 and value not in {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}:
            suspicion += 0.4
        
        # Factor 7: Special case for any threshold assignment (even if not overly precise)
        if 'threshold' in var_name:
            suspicion += 0.3  # Boost suspicion for explicit threshold variables
        
        return max(0.0, min(1.0, suspicion))
    
    def _is_suspicious_magic_number(self, value: float) -> bool:
        """Check if a numeric value looks like a suspicious magic number"""
        # Skip known good values
        if value in self.known_good_thresholds:
            return False
        
        # Check for overly precise decimals
        if self.suspicious_precision_regex.match(str(value)):
            return True
        
        # Check for weird threshold ranges
        if 0.0 < value < 1.0:
            # Suspicious if it's not a round number
            rounded = round(value, 1)
            if abs(value - rounded) > 0.05:  # Not close to round number
                return True
        
        return False
    
    def _is_ml_context_comparison(self, node: ast.Compare, analysis: ASTAnalysisResult) -> bool:
        """Check if a comparison is in an ML context (model predictions, probabilities, etc.)"""
        # Check variable names in the comparison
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                var_name = child.id.lower()
                ml_indicators = ['prob', 'score', 'predict', 'confidence', 'accuracy', 'precision', 'recall']
                if any(indicator in var_name for indicator in ml_indicators):
                    return True
        
        return False
    
    def _has_nearby_documentation(self, node: ast.Assign, analysis: ASTAnalysisResult) -> bool:
        """Check if there's documentation (comments) near the assignment"""
        # This is a simplified check - in a real implementation,
        # we'd parse comments from the source code
        line_num = node.lineno
        
        # Check if variable name itself is self-documenting
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                # Self-documenting names
                if any(word in var_name.lower() for word in self.business_context_keywords):
                    return True
        
        return False
    
    def _is_calculated_value(self, node: ast.Assign) -> bool:
        """Check if the assigned value is calculated rather than a literal"""
        # If the value is anything other than a simple constant, it's calculated
        return not isinstance(node.value, ast.Constant)
    
    def _create_hardcoded_threshold_pattern(self, node: ast.Assign, analysis: ASTAnalysisResult,
                                          var_name: str, value: float, confidence: float) -> MLAntiPattern:
        """Create anti-pattern for hardcoded threshold assignment"""
        return MLAntiPattern(
            pattern_type="hardcoded_threshold",
            severity=PatternSeverity.MEDIUM,
            line_number=node.lineno,
            column=node.col_offset,
            message=f"Hardcoded threshold value {value} assigned to {var_name}",
            explanation=f"The threshold value {value} appears to be hardcoded without business justification. "
                       f"Consider using a named constant with documentation explaining the business logic.",
            suggested_fix=f"# Define threshold with business context\n"
                         f"BUSINESS_THRESHOLD = {value}  # Document why this value was chosen\n"
                         f"{var_name} = BUSINESS_THRESHOLD",
            confidence=confidence,
            code_snippet=self._extract_code_snippet(analysis, node.lineno),
            fix_snippet=f"# Business requirement: explain why {value}\n"
                       f"DECISION_THRESHOLD = {value}  # Based on cost analysis\n"
                       f"{var_name} = DECISION_THRESHOLD",
            references=[
                "https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-curve",
                "https://developers.google.com/machine-learning/crash-course/classification/thresholding"
            ]
        )
    
    def _create_magic_comparison_pattern(self, node: ast.Compare, analysis: ASTAnalysisResult,
                                       value: float) -> MLAntiPattern:
        """Create anti-pattern for magic number in comparison"""
        return MLAntiPattern(
            pattern_type="magic_number_comparison",
            severity=PatternSeverity.MEDIUM,
            line_number=node.lineno,
            column=node.col_offset,
            message=f"Magic number {value} used in comparison",
            explanation=f"The value {value} is used directly in a comparison without explanation. "
                       f"This makes the code hard to understand and maintain.",
            suggested_fix=f"Define a named constant: THRESHOLD = {value}  # Add explanation",
            confidence=0.8,
            code_snippet=self._extract_code_snippet(analysis, node.lineno),
            fix_snippet=f"# Define meaningful constant\n"
                       f"CONFIDENCE_THRESHOLD = {value}  # Business requirement\n"
                       f"if prediction >= CONFIDENCE_THRESHOLD:",
            references=[
                "https://en.wikipedia.org/wiki/Magic_number_(programming)",
                "https://www.python.org/dev/peps/pep-0008/#constants"
            ]
        )
    
    def _extract_code_snippet(self, analysis: ASTAnalysisResult, line_number: int) -> str:
        """Extract a code snippet around the given line number"""
        try:
            with open(analysis.file_path, 'r') as f:
                lines = f.readlines()
                
            start = max(0, line_number - 2)
            end = min(len(lines), line_number + 1)
            
            snippet_lines = []
            for i in range(start, end):
                prefix = ">>> " if i == line_number - 1 else "    "
                snippet_lines.append(f"{prefix}{lines[i].rstrip()}")
            
            return "\n".join(snippet_lines)
        except (IOError, IndexError):
            return f"Line {line_number}: [code snippet unavailable]"


class InefficientDataLoadingDetector:
    """
    Detects inefficient data loading patterns that cause performance bottlenecks.
    
    This detector identifies common data loading anti-patterns in ML code that
    can lead to memory issues, slow execution, and poor resource utilization.
    
    Key patterns detected:
    - Loading entire large datasets into memory without chunking
    - Redundant data loading (same file loaded multiple times)
    - Row-by-row iteration instead of vectorized operations
    - Loading all columns when only subset is needed
    - Missing data type specifications leading to memory waste
    - No caching for frequently accessed data
    """
    
    def __init__(self):
        # Data loading functions that should use chunking for large files
        self.data_loading_functions = {
            'pd.read_csv', 'pandas.read_csv', 'read_csv',
            'pd.read_excel', 'pandas.read_excel', 'read_excel',
            'pd.read_json', 'pandas.read_json', 'read_json',
            'pd.read_parquet', 'pandas.read_parquet', 'read_parquet'
        }
        
        # Inefficient iteration patterns
        self.inefficient_iteration_patterns = {
            'iterrows', 'itertuples', 'iloc', 'loc'
        }
        
        # Memory-efficient parameters that are often missing
        self.efficiency_parameters = {
            'chunksize', 'dtype', 'usecols', 'nrows'
        }
        
        # File variables that suggest large datasets
        self.large_dataset_indicators = {
            'huge', 'large', 'big', 'massive', 'full', 'complete', 'entire'
        }
    
    def detect_patterns(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Detect inefficient data loading anti-patterns"""
        patterns = []
        
        # Track loaded files to detect redundant loading
        loaded_files = {}
        
        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Call):
                patterns.extend(self._analyze_data_loading_call(node, analysis, loaded_files))
            elif isinstance(node, ast.For):
                patterns.extend(self._analyze_iteration_pattern(node, analysis))
        
        return patterns
    
    def _analyze_data_loading_call(self, node: ast.Call, analysis: ASTAnalysisResult, 
                                  loaded_files: Dict[str, List[int]]) -> List[MLAntiPattern]:
        """Analyze data loading function calls for inefficiencies"""
        patterns = []
        
        # Get function name
        func_name = self._get_function_name(node)
        
        if any(loading_func in func_name for loading_func in self.data_loading_functions):
            # Check for various inefficiency patterns
            
            # 1. Check for missing efficiency parameters
            patterns.extend(self._check_missing_efficiency_params(node, analysis, func_name))
            
            # 2. Check for redundant loading
            patterns.extend(self._check_redundant_loading(node, analysis, loaded_files))
            
            # 3. Check for large dataset indicators without chunking
            patterns.extend(self._check_large_dataset_without_chunking(node, analysis))
        
        return patterns
    
    def _analyze_iteration_pattern(self, node: ast.For, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Analyze for loops for inefficient iteration patterns"""
        patterns = []
        
        # Check if this is iterating over a dataframe in an inefficient way
        if self._is_inefficient_dataframe_iteration(node):
            patterns.append(self._create_inefficient_iteration_pattern(node, analysis))
        
        return patterns
    
    def _get_function_name(self, node: ast.Call) -> str:
        """Extract function name from call node"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
            return node.func.attr
        return ""
    
    def _check_missing_efficiency_params(self, node: ast.Call, analysis: ASTAnalysisResult, 
                                       func_name: str) -> List[MLAntiPattern]:
        """Check if data loading call is missing efficiency parameters"""
        patterns = []
        
        # Get all keyword arguments
        kwargs = {kw.arg for kw in node.keywords if kw.arg}
        
        # Check for missing chunking on potentially large files
        if 'chunksize' not in kwargs and self._suggests_large_file(node):
            patterns.append(self._create_missing_chunking_pattern(node, analysis))
        
        # Check for missing dtype specification (but not for small files or when explicitly loading with defaults)
        if 'dtype' not in kwargs and not self._is_small_or_test_file(node):
            patterns.append(self._create_missing_dtype_pattern(node, analysis))
        
        # Check for missing column selection
        if 'usecols' not in kwargs and self._suggests_wide_dataset(node):
            patterns.append(self._create_missing_usecols_pattern(node, analysis))
        
        return patterns
    
    def _check_redundant_loading(self, node: ast.Call, analysis: ASTAnalysisResult,
                                loaded_files: Dict[str, List[int]]) -> List[MLAntiPattern]:
        """Check for redundant loading of the same file"""
        patterns = []
        
        # Extract file path from arguments
        file_path = self._extract_file_path(node)
        
        if file_path:
            if file_path in loaded_files:
                # This file was loaded before
                patterns.append(self._create_redundant_loading_pattern(
                    node, analysis, file_path, loaded_files[file_path]
                ))
                loaded_files[file_path].append(node.lineno)
            else:
                loaded_files[file_path] = [node.lineno]
        
        return patterns
    
    def _check_large_dataset_without_chunking(self, node: ast.Call, 
                                            analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Check for loading large datasets without chunking"""
        patterns = []
        
        # Check if filename or variable names suggest large datasets
        if self._suggests_large_file(node):
            # Check if chunking is used
            kwargs = {kw.arg for kw in node.keywords if kw.arg}
            if 'chunksize' not in kwargs:
                patterns.append(self._create_large_file_no_chunking_pattern(node, analysis))
        
        return patterns
    
    def _is_inefficient_dataframe_iteration(self, node: ast.For) -> bool:
        """Check if this is an inefficient dataframe iteration pattern"""
        # Look for patterns like: for i in range(len(df)): df.iloc[i]
        if isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                # Check if range uses len() on what looks like a dataframe
                if (node.iter.args and isinstance(node.iter.args[0], ast.Call) and
                    isinstance(node.iter.args[0].func, ast.Name) and 
                    node.iter.args[0].func.id == 'len'):
                    return True
        
        # Look for direct iteration over dataframe with iloc/loc usage
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr in self.inefficient_iteration_patterns:
                return True
        
        return False
    
    def _suggests_large_file(self, node: ast.Call) -> bool:
        """Check if the file path suggests a large dataset"""
        file_path = self._extract_file_path(node)
        if file_path:
            file_path_lower = file_path.lower()
            return any(indicator in file_path_lower for indicator in self.large_dataset_indicators)
        return False
    
    def _suggests_wide_dataset(self, node: ast.Call) -> bool:
        """Check if the file path suggests a wide dataset (many columns)"""
        file_path = self._extract_file_path(node)
        if file_path:
            file_path_lower = file_path.lower()
            wide_indicators = {'wide', 'features', 'columns', 'full_features'}
            return any(indicator in file_path_lower for indicator in wide_indicators)
        return False
    
    def _extract_file_path(self, node: ast.Call) -> str:
        """Extract file path from function call arguments"""
        if node.args and isinstance(node.args[0], ast.Constant):
            return str(node.args[0].value)
        return ""
    
    def _is_small_or_test_file(self, node: ast.Call) -> bool:
        """Check if this is a small file or test file that doesn't need dtype specification"""
        file_path = self._extract_file_path(node)
        if file_path:
            file_path_lower = file_path.lower()
            small_indicators = {'test', 'sample', 'demo', 'example', 'small'}
            return any(indicator in file_path_lower for indicator in small_indicators)
        return False
    
    def _create_missing_chunking_pattern(self, node: ast.Call, 
                                       analysis: ASTAnalysisResult) -> MLAntiPattern:
        """Create pattern for missing chunking on large files"""
        return MLAntiPattern(
            pattern_type="missing_data_chunking",
            severity=PatternSeverity.MEDIUM,
            line_number=node.lineno,
            column=node.col_offset,
            message="Large dataset loaded without chunking",
            explanation="Loading large datasets entirely into memory can cause memory errors. "
                       "Consider using chunking to process data in smaller batches.",
            suggested_fix="Use chunksize parameter: pd.read_csv('file.csv', chunksize=10000)",
            confidence=0.8,
            code_snippet=self._extract_code_snippet(analysis, node.lineno),
            fix_snippet="# Process large files in chunks\n"
                       "for chunk in pd.read_csv('large_file.csv', chunksize=10000):\n"
                       "    process_chunk(chunk)",
            references=[
                "https://pandas.pydata.org/docs/user_guide/io.html#io-chunking",
                "https://towardsdatascience.com/why-and-how-to-use-pandas-with-large-data-9594dda2ea4c"
            ]
        )
    
    def _create_missing_dtype_pattern(self, node: ast.Call, 
                                    analysis: ASTAnalysisResult) -> MLAntiPattern:
        """Create pattern for missing dtype specification"""
        return MLAntiPattern(
            pattern_type="missing_dtype_specification",
            severity=PatternSeverity.LOW,
            line_number=node.lineno,
            column=node.col_offset,
            message="Data loading without dtype specification",
            explanation="Not specifying data types leads to pandas inferring types, "
                       "which can waste memory and cause performance issues.",
            suggested_fix="Specify dtypes: pd.read_csv('file.csv', dtype={'col1': 'int32', 'col2': 'category'})",
            confidence=0.6,
            code_snippet=self._extract_code_snippet(analysis, node.lineno),
            fix_snippet="# Specify data types for memory efficiency\n"
                       "dtype_spec = {'user_id': 'int32', 'category': 'category'}\n"
                       "df = pd.read_csv('data.csv', dtype=dtype_spec)",
            references=[
                "https://pandas.pydata.org/docs/user_guide/basics.html#dtypes",
                "https://www.dataquest.io/blog/pandas-big-data/"
            ]
        )
    
    def _create_missing_usecols_pattern(self, node: ast.Call, 
                                      analysis: ASTAnalysisResult) -> MLAntiPattern:
        """Create pattern for missing column selection"""
        return MLAntiPattern(
            pattern_type="loading_unused_columns",
            severity=PatternSeverity.MEDIUM,
            line_number=node.lineno,
            column=node.col_offset,
            message="Loading all columns when subset may be sufficient",
            explanation="Loading all columns from wide datasets wastes memory and I/O. "
                       "Consider loading only the columns you need.",
            suggested_fix="Use usecols parameter: pd.read_csv('file.csv', usecols=['col1', 'col2', 'target'])",
            confidence=0.7,
            code_snippet=self._extract_code_snippet(analysis, node.lineno),
            fix_snippet="# Load only required columns\n"
                       "required_cols = ['feature1', 'feature2', 'target']\n"
                       "df = pd.read_csv('wide_data.csv', usecols=required_cols)",
            references=[
                "https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html",
                "https://towardsdatascience.com/5-methods-to-check-for-nan-values-in-in-python-3f21ddd17eed"
            ]
        )
    
    def _create_redundant_loading_pattern(self, node: ast.Call, analysis: ASTAnalysisResult,
                                        file_path: str, previous_lines: List[int]) -> MLAntiPattern:
        """Create pattern for redundant file loading"""
        return MLAntiPattern(
            pattern_type="redundant_data_loading",
            severity=PatternSeverity.MEDIUM,
            line_number=node.lineno,
            column=node.col_offset,
            message=f"File '{file_path}' loaded multiple times",
            explanation=f"The same file is being loaded multiple times (previous loads at lines {previous_lines}). "
                       "Consider loading once and reusing the dataframe, or implement caching.",
            suggested_fix="Load once and reuse: df = pd.read_csv('file.csv') # then reuse df",
            confidence=0.9,
            code_snippet=self._extract_code_snippet(analysis, node.lineno),
            fix_snippet="# Load once and cache\n"
                       "import functools\n"
                       "@functools.lru_cache(maxsize=None)\n"
                       "def load_data(file_path):\n"
                       "    return pd.read_csv(file_path)",
            references=[
                "https://docs.python.org/3/library/functools.html#functools.lru_cache",
                "https://realpython.com/lru-cache-python/"
            ]
        )
    
    def _create_large_file_no_chunking_pattern(self, node: ast.Call, 
                                             analysis: ASTAnalysisResult) -> MLAntiPattern:
        """Create pattern for large files loaded without chunking"""
        return MLAntiPattern(
            pattern_type="large_file_memory_risk",
            severity=PatternSeverity.HIGH,
            line_number=node.lineno,
            column=node.col_offset,
            message="Large dataset loaded entirely into memory",
            explanation="Loading large datasets without chunking can cause memory errors and poor performance. "
                       "For large files, use chunking or streaming approaches.",
            suggested_fix="Implement chunking: for chunk in pd.read_csv('large_file.csv', chunksize=50000):",
            confidence=0.85,
            code_snippet=self._extract_code_snippet(analysis, node.lineno),
            fix_snippet="# Safe loading of large files\n"
                       "def load_large_file(file_path, chunk_size=50000):\n"
                       "    chunks = []\n"
                       "    for chunk in pd.read_csv(file_path, chunksize=chunk_size):\n"
                       "        chunks.append(chunk)\n"
                       "    return pd.concat(chunks, ignore_index=True)",
            references=[
                "https://pandas.pydata.org/docs/user_guide/io.html#io-chunking",
                "https://towardsdatascience.com/efficiently-iterating-over-rows-in-a-pandas-dataframe-7dd5f9992c01"
            ]
        )
    
    def _create_inefficient_iteration_pattern(self, node: ast.For, 
                                            analysis: ASTAnalysisResult) -> MLAntiPattern:
        """Create pattern for inefficient dataframe iteration"""
        return MLAntiPattern(
            pattern_type="inefficient_dataframe_iteration",
            severity=PatternSeverity.MEDIUM,
            line_number=node.lineno,
            column=node.col_offset,
            message="Inefficient row-by-row dataframe iteration",
            explanation="Iterating over dataframes row-by-row is very slow. "
                       "Use vectorized operations or apply() for better performance.",
            suggested_fix="Use vectorized operations: df['new_col'] = df['col1'] * df['col2']",
            confidence=0.85,
            code_snippet=self._extract_code_snippet(analysis, node.lineno),
            fix_snippet="# Vectorized operation instead of loop\n"
                       "# Instead of: for i in range(len(df)): df.loc[i, 'result'] = df.loc[i, 'value'] * 2\n"
                       "df['result'] = df['value'] * 2  # Much faster!",
            references=[
                "https://pandas.pydata.org/docs/user_guide/enhancingperf.html",
                "https://towardsdatascience.com/efficiently-iterating-over-rows-in-a-pandas-dataframe-7dd5f9992c01"
            ]
        )
    
    def _extract_code_snippet(self, analysis: ASTAnalysisResult, line_number: int) -> str:
        """Extract a code snippet around the given line number"""
        try:
            with open(analysis.file_path, 'r') as f:
                lines = f.readlines()
                
            start = max(0, line_number - 2)
            end = min(len(lines), line_number + 1)
            
            snippet_lines = []
            for i in range(start, end):
                prefix = ">>> " if i == line_number - 1 else "    "
                snippet_lines.append(f"{prefix}{lines[i].rstrip()}")
            
            return "\n".join(snippet_lines)
        except (IOError, IndexError):
            return f"Line {line_number}: [code snippet unavailable]"
