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
            return expr.id in tensor_vars or any(keyword in expr.id.lower() for keyword in
                                               ['tensor', 'output', 'prediction', 'loss', 'result'])

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
                            if arg.value not in [0, 1, -1]:  # Ignore obvious non-magic numbers
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
                            if keyword.value.value not in [0, 1, -1]:  # Ignore obvious non-magic numbers
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
                        if arg.value > 10:  # Likely a magic dimension
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
            'LinearRegression', 'RandomForestClassifier', 'SVC',
            'Adam', 'SGD', 'AdamW',
            'Linear', 'Conv2d', 'LSTM'
        }

        if isinstance(node.func, ast.Name):
            return node.func.id in ml_constructors
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr in ml_constructors

        return False

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
            'RandomForestClassifier', 'random', 'rand', 'randn'
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

                    # For numpy random functions, check if global seed is set
                    is_numpy_random = (isinstance(node.func, ast.Attribute) and
                                     isinstance(node.func.value, ast.Attribute) and
                                     node.func.value.attr == 'random')

                    # Skip numpy random functions if global seed is set
                    if is_numpy_random and has_global_seed:
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
                    if (node.func.attr == 'seed' and
                        isinstance(node.func.value, ast.Attribute) and
                        node.func.value.attr == 'random'):
                        return True
                    elif node.func.attr in ['manual_seed', 'seed']:
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


class MLPatternDetector:
    """
    Main orchestrator for all ML anti-pattern detection.

    This class coordinates all the specialized detectors and provides
    a unified interface for pattern detection.
    """

    def __init__(self):
        self.detectors = {
            'data_leakage': DataLeakageDetector(),
            'gpu_memory': GPUMemoryLeakDetector(),
            'magic_numbers': MagicNumberExtractor(),
            'reproducibility': ReproducibilityChecker(),
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
