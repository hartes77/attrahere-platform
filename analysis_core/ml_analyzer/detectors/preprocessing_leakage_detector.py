"""
Preprocessing Leakage Detector - Critical ML Pattern Detection

Detects the most common and critical data leakage pattern: preprocessing applied
to the entire dataset before train/test splitting. This causes the model to gain
access to test set statistics during training, leading to overly optimistic
performance estimates that don't generalize.

Key Pattern Detected:
- Preprocessing operations (fit/fit_transform) before data splitting
- Critical 0.95 confidence pattern with clear fix guidance
- Order-of-operations tracking for accurate sequence detection

Extracted from DataLeakageDetector and modernized for BaseMLDetector architecture.
Focused exclusively on preprocessing leakage - other patterns handled by specialized detectors.
"""

import ast
from typing import Dict, List, Any, Optional, Set

from .base_detector import BaseMLDetector, MLAntiPattern, PatternSeverity
from ..ast_engine import ASTAnalysisResult
from ..scope_analyzer import ScopeAwareAnalyzer


class PreprocessingLeakageDetector(BaseMLDetector):
    """
    Detects preprocessing applied to entire dataset before train/test split.

    This is the most common and critical data leakage pattern in ML.
    When preprocessing (StandardScaler, encoders, etc.) is fitted on the complete
    dataset before splitting, it leaks test set statistics into training.
    """

    def __init__(self):
        super().__init__()

        # Initialize scope-aware analyzer for V4 intelligence
        self.scope_analyzer = ScopeAwareAnalyzer()

        # Preprocessing classes that commonly cause leakage
        self.preprocessor_classes = {
            'StandardScaler', 'MinMaxScaler', 'RobustScaler',
            'Normalizer', 'QuantileTransformer', 'PowerTransformer',
            'LabelEncoder', 'OneHotEncoder', 'OrdinalEncoder',
            'SimpleImputer', 'KNNImputer', 'IterativeImputer'
        }

        # Methods that fit/learn from data (cause leakage if used before split)
        self.fit_methods = {'fit', 'fit_transform'}
        
        # Data splitting functions
        self.split_functions = {'train_test_split', 'KFold', 'StratifiedKFold', 'TimeSeriesSplit'}

    def detect_patterns(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Main entry point for scope-aware preprocessing leakage detection"""
        patterns = []

        # V4 Enhancement: Perform scope analysis first
        scope_context = self.scope_analyzer.analyze_scope_context(analysis.ast_tree)

        # Focus exclusively on preprocessing-before-split detection with scope awareness
        patterns.extend(self._detect_preprocessing_before_split_with_scope(analysis, scope_context))

        return patterns

    def _detect_preprocessing_before_split_with_scope(
        self, analysis: ASTAnalysisResult, scope_context: Dict[str, Any]
    ) -> List[MLAntiPattern]:
        """
        V4 Scope-aware preprocessing leakage detection.
        
        Uses scope analysis to eliminate false positives from function-local operations.
        """
        patterns = []

        # Get operation contexts from scope analysis
        operation_contexts = scope_context['operation_contexts']
        
        # Sort operations by line number for chronological analysis
        sorted_operations = sorted(operation_contexts, key=lambda x: x.line)

        # Find problematic sequences: preprocessing before splitting
        for i, op in enumerate(sorted_operations):
            if (op.operation_type == 'preprocessing' and 
                op.details['method'] in self.fit_methods):
                
                # Look for subsequent split operations
                for j in range(i + 1, len(sorted_operations)):
                    split_op = sorted_operations[j]
                    if split_op.operation_type == 'data_split':
                        
                        # V4 BREAKTHROUGH: Scope-aware validation
                        validation = self.scope_analyzer.validate_pattern_in_scope(
                            op.line, 'preprocessing_before_split', scope_context
                        )
                        
                        # Only create pattern if it's a real issue (not a false positive)
                        if validation['is_valid']:
                            patterns.append(self._create_preprocessing_leakage_pattern_v4(
                                op, split_op, analysis, validation
                            ))
                        # Note: False positives are automatically filtered out
                        break

        return patterns

    def _detect_preprocessing_before_split(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """
        Detect preprocessing applied to entire dataset before train/test split.

        This is the most common and critical data leakage pattern.
        """
        patterns = []

        # Track the chronological order of operations
        operations = []
        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Assign):
                operation_info = self._analyze_assignment_for_leakage(node)
                if operation_info:
                    operations.append(operation_info)

        # Sort operations by line number to ensure chronological order
        operations.sort(key=lambda x: x['line'])

        # Find problematic sequences: preprocessing before splitting
        for i, op in enumerate(operations):
            if op['type'] == 'preprocessing' and op['method'] in self.fit_methods:
                # Look for subsequent split operations
                for j in range(i + 1, len(operations)):
                    if operations[j]['type'] == 'data_split':
                        # Found preprocessing before split - CRITICAL LEAKAGE!
                        patterns.append(self._create_preprocessing_leakage_pattern(
                            op, operations[j], analysis
                        ))
                        break

        return patterns

    def _create_preprocessing_leakage_pattern_v4(
        self, preprocessing_op, split_op, analysis: ASTAnalysisResult, validation: Dict[str, Any]
    ) -> MLAntiPattern:
        """Create V4 scope-aware pattern for preprocessing leakage"""
        preprocessor = preprocessing_op.details['object']
        method = preprocessing_op.details['method']
        split_func = split_op.details['function']
        
        # Enhanced confidence based on scope validation
        base_confidence = 0.95
        if validation['scope_info']['scope_type'] == 'global':
            confidence = base_confidence  # Global scope = definitely a problem
        else:
            confidence = base_confidence * 0.9  # Function scope = slightly less confident
        
        # Enhanced message with scope context
        scope_type = validation['scope_info']['scope_type']
        scope_name = validation['scope_info'].get('scope_name', 'unknown')
        
        if scope_type == 'global':
            message = f"{preprocessor}.{method}() applied before {split_func} (global scope)"
        else:
            message = f"{preprocessor}.{method}() applied before {split_func} (in {scope_name}())"

        return self.create_pattern(
            pattern_type="preprocessing_before_split",
            severity=PatternSeverity.CRITICAL,
            node=preprocessing_op.details['node'],
            analysis=analysis,
            message=message,
            explanation=(
                f"Preprocessing operation {preprocessor}.{method}() is applied to the entire dataset "
                f"before {split_func} in {scope_type} scope. This causes data leakage because the "
                "preprocessor learns statistics from the test set, leading to overly optimistic "
                "performance estimates that don't generalize to real-world deployment."
            ),
            suggested_fix=(
                f"Move {preprocessor}.{method}() after {split_func}. "
                "Fit the preprocessor only on training data, then transform both train and test sets separately."
            ),
            confidence=confidence,
            fix_snippet=self._generate_preprocessing_fix_v4(preprocessor, method, split_func, scope_type),
            references=[
                "https://scikit-learn.org/stable/common_pitfalls.html#data-leakage",
                "https://machinelearningmastery.com/data-leakage-machine-learning/",
                "https://towardsdatascience.com/data-leakage-in-machine-learning-10bdd3eec742"
            ]
        )

    def _generate_preprocessing_fix_v4(
        self, preprocessor: str, method: str, split_func: str, scope_type: str
    ) -> str:
        """Generate V4 scope-aware fix snippet"""
        if scope_type == 'global':
            return f"""# CORRECT: Preprocessing after splitting (global scope)
X_train, X_test, y_train, y_test = {split_func}(X, y, test_size=0.2, random_state=42)

# Fit preprocessor only on training data
{preprocessor} = StandardScaler()  # or your preprocessor
X_train_scaled = {preprocessor}.fit_transform(X_train)  # Learn from train only
X_test_scaled = {preprocessor}.transform(X_test)        # Apply to test without learning

# AVOID: {preprocessor}.{method}(X) before splitting - this causes leakage!"""
        else:
            return f"""# CORRECT: Preprocessing after splitting (function scope)
def your_function():
    # Split data first inside function
    X_train, X_test, y_train, y_test = {split_func}(X, y, test_size=0.2, random_state=42)
    
    # Then apply preprocessing
    {preprocessor} = StandardScaler()
    X_train_scaled = {preprocessor}.fit_transform(X_train)  # Learn from train only
    X_test_scaled = {preprocessor}.transform(X_test)        # Apply to test without learning
    
    return X_train_scaled, X_test_scaled

# AVOID: {preprocessor}.{method}(X) before splitting - this causes leakage!"""

    def _analyze_assignment_for_leakage(self, node: ast.Assign) -> Optional[Dict[str, Any]]:
        """Analyze an assignment to see if it's a leakage-relevant operation"""
        if not isinstance(node.value, ast.Call):
            return None

        # Check for preprocessing operations (method calls)
        if isinstance(node.value.func, ast.Attribute):
            obj_name = None
            if isinstance(node.value.func.value, ast.Name):
                obj_name = node.value.func.value.id

            method_name = node.value.func.attr

            # Detect fit/fit_transform calls on preprocessors
            if method_name in self.fit_methods:
                return {
                    'type': 'preprocessing',
                    'preprocessor': obj_name or 'preprocessor',
                    'method': method_name,
                    'line': node.lineno,
                    'column': node.col_offset,
                    'node': node
                }

        # Check for data splitting operations (function calls)
        elif isinstance(node.value.func, ast.Name):
            func_name = node.value.func.id
            if func_name in self.split_functions:
                return {
                    'type': 'data_split',
                    'function': func_name,
                    'line': node.lineno,
                    'column': node.col_offset,
                    'node': node
                }

        return None

    def _create_preprocessing_leakage_pattern(
        self, preprocessing_op: Dict, split_op: Dict, analysis: ASTAnalysisResult
    ) -> MLAntiPattern:
        """Create pattern for preprocessing before split leakage"""
        preprocessor = preprocessing_op['preprocessor']
        method = preprocessing_op['method']
        split_func = split_op['function']

        return self.create_pattern(
            pattern_type="preprocessing_before_split",
            severity=PatternSeverity.CRITICAL,
            node=preprocessing_op['node'],
            analysis=analysis,
            message=f"{preprocessor}.{method}() applied before {split_func}",
            explanation=(
                f"Preprocessing operation {preprocessor}.{method}() is applied to the entire dataset "
                f"before {split_func}. This causes data leakage because the preprocessor learns "
                "statistics from the test set, leading to overly optimistic performance estimates "
                "that don't generalize to real-world deployment."
            ),
            suggested_fix=(
                f"Move {preprocessor}.{method}() after {split_func}. "
                "Fit the preprocessor only on training data, then transform both train and test sets separately."
            ),
            confidence=0.95,
            fix_snippet=self._generate_preprocessing_fix(preprocessor, method, split_func),
            references=[
                "https://scikit-learn.org/stable/common_pitfalls.html#data-leakage",
                "https://machinelearningmastery.com/data-leakage-machine-learning/",
                "https://towardsdatascience.com/data-leakage-in-machine-learning-10bdd3eec742"
            ]
        )

    def _generate_preprocessing_fix(self, preprocessor: str, method: str, split_func: str) -> str:
        """Generate fix snippet for preprocessing leakage"""
        return f"""# CORRECT: Preprocessing after splitting
X_train, X_test, y_train, y_test = {split_func}(X, y, test_size=0.2, random_state=42)

# Fit preprocessor only on training data
{preprocessor} = StandardScaler()  # or your preprocessor
X_train_scaled = {preprocessor}.fit_transform(X_train)  # Learn from train only
X_test_scaled = {preprocessor}.transform(X_test)        # Apply to test without learning

# AVOID: {preprocessor}.{method}(X) before splitting - this causes leakage!"""

    def _is_preprocessor_operation(self, node: ast.Call) -> bool:
        """Check if this is a preprocessing operation that could cause leakage"""
        # Check method calls
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if method_name in self.fit_methods:
                # Check if it's on a preprocessor object
                if isinstance(node.func.value, ast.Name):
                    obj_name = node.func.value.id
                    # Heuristic: common preprocessor variable names
                    preprocessor_names = {'scaler', 'encoder', 'imputer', 'transformer', 'normalizer'}
                    return any(prep in obj_name.lower() for prep in preprocessor_names)
                return True  # Conservative: assume it could be a preprocessor

        # Check constructor calls
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
            return func_name in self.preprocessor_classes

        return False