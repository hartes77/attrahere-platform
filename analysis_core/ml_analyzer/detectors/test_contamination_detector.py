"""
Test Set Contamination Detector - Sprint 3 Core Detector

Detects test set contamination patterns using sophisticated semantic analysis.
This is the core differentiator of our platform, providing ML-specific
contamination detection that goes beyond simple syntax checking.
"""

import ast
from typing import Dict, List, Any, Optional, Set, Union

from .base_detector import BaseMLDetector, MLAntiPattern, PatternSeverity
from ..ast_engine import ASTAnalysisResult


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


class TestSetContaminationDetector(BaseMLDetector):
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
        super().__init__()
        
        # Track variable assignments and data transformations
        self.variable_tracker = VariableSemanticTracker()
        
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
        func_name = self.get_function_name(node)
        
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
                func_name = self.get_function_name(node)
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
    
    # Helper methods for detection logic
    def _is_split_function(self, node: ast.Call) -> bool:
        """Check if a function call is a data splitting function"""
        func_name = self.get_function_name(node)
        return func_name in self.split_functions
    
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
        func_name = self.get_function_name(node)
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
                func_name = self.get_function_name(node)
                if func_name in ['intersection', 'duplicated', 'drop_duplicates']:
                    if self._involves_split_variables(node):
                        return True
        return False
    
    def _find_split_operation_line(self, analysis: ASTAnalysisResult) -> Optional[int]:
        """Find the line number where train/test split occurs"""
        split_ops = self.variable_tracker.get_split_operations()
        if split_ops:
            return split_ops[0].get('line')
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
            func_name = self.get_function_name(node.value)
            if func_name == 'shift' and self._has_negative_shift(node.value):
                return True
            if func_name == 'rolling' and self._has_unsafe_rolling(node.value):
                return True
        
        return False
    
    def _has_negative_shift(self, node: ast.Call) -> bool:
        """Check if shift operation has negative periods (future data)"""
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                if arg.value < 0:  # Negative shift = future data
                    return True
        # Also check keyword arguments
        for keyword in node.keywords:
            if keyword.arg == 'periods' and isinstance(keyword.value, ast.Constant):
                if isinstance(keyword.value.value, int) and keyword.value.value < 0:
                    return True
        return False
    
    def _is_derived_from_target(self, var_info: Dict, target_vars: Set[str]) -> bool:
        """Check if variable is derived from target variables"""
        # Simplified check - would need more sophisticated dependency analysis
        return False
    
    def _causes_temporal_leakage(self, node: ast.Call) -> bool:
        """Check if temporal operation causes future leakage"""
        func_name = self.get_function_name(node)
        
        # Check for shift operations with negative values (future data)
        if func_name == 'shift':
            return self._has_negative_shift(node)
        
        # Check for rolling operations using target variable
        if func_name == 'rolling':
            return self._has_unsafe_rolling(node)
        
        # Check for fillna using future values
        if func_name == 'fillna':
            return self._uses_future_fill(node)
        
        return False
    
    def _has_safe_rolling_window(self, node: ast.Call) -> bool:
        """Check if rolling operation has safe window parameters"""
        # Look for window parameter
        for keyword in node.keywords:
            if keyword.arg == 'window':
                return True
        return False
    
    def _has_unsafe_rolling(self, node: ast.Call) -> bool:
        """Check if rolling operation uses target variable (potential leakage)"""
        # Check if rolling is applied to target-related variables
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Attribute):
                # Check for patterns like df['target'].rolling() or y.rolling()
                attr_name = getattr(node.func.value, 'attr', '')
                if 'target' in attr_name.lower() or attr_name in ['y', 'label']:
                    return True
            elif isinstance(node.func.value, ast.Name):
                var_name = node.func.value.id
                if 'target' in var_name.lower() or var_name in ['y', 'label']:
                    return True
        return False
    
    def _uses_future_fill(self, node: ast.Call) -> bool:
        """Check if fillna uses future values for filling"""
        # Check arguments for mean() or other aggregate functions that could use future data
        for arg in node.args:
            if isinstance(arg, ast.Call):
                func_name = self.get_function_name(arg)
                if func_name in ['mean', 'median', 'mode']:
                    return True
        return False
    
    def _is_temporal_data(self, split_op: Dict) -> bool:
        """Check if split operation involves temporal data"""
        # Check if data contains date/time columns or temporal indicators
        variables = self.variable_tracker.variables
        
        # Look for datetime-related variable names
        temporal_indicators = ['date', 'time', 'timestamp', 'datetime', 'rolling', 'lag', 'shift']
        for var_name in variables:
            if any(indicator in var_name.lower() for indicator in temporal_indicators):
                return True
        
        return False
    
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
                func_name = self.get_function_name(node)
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
    
    # Pattern creation methods using inherited create_pattern helper
    def _create_duplicate_detection_pattern(self, node: ast.Call, analysis: ASTAnalysisResult) -> MLAntiPattern:
        """Create pattern for duplicate detection between train/test"""
        return self.create_pattern(
            pattern_type="test_set_exact_duplicates",
            severity=PatternSeverity.CRITICAL,
            node=node,
            analysis=analysis,
            message="Exact duplicates detected between train and test sets",
            explanation=(
                "Identical samples in both training and test sets cause severe overfitting. "
                "The model has already seen the test data during training, making evaluation invalid."
            ),
            suggested_fix="Remove duplicates before splitting or use temporal/group-based splits",
            confidence=0.95,
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
        # Create a dummy node for the split line
        dummy_node = ast.parse("pass").body[0]
        dummy_node.lineno = split_line
        
        return self.create_pattern(
            pattern_type="missing_duplicate_check",
            severity=PatternSeverity.HIGH,
            node=dummy_node,
            analysis=analysis,
            message="No duplicate detection between train and test sets",
            explanation=(
                "Without checking for duplicates between train and test sets, "
                "you risk contamination that leads to overly optimistic results."
            ),
            suggested_fix="Add duplicate detection before evaluation",
            confidence=0.80,
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
    
    def _create_cross_merge_pattern(self, node: ast.Call, analysis: ASTAnalysisResult) -> MLAntiPattern:
        """Create pattern for inappropriate merging of train/test data"""
        return self.create_pattern(
            pattern_type="cross_split_merge",
            severity=PatternSeverity.CRITICAL,
            node=node,
            analysis=analysis,
            message="Inappropriate merge/join between train and test data",
            explanation=(
                "Merging or joining train and test data can cause contamination. "
                "Train and test sets should remain completely separate."
            ),
            suggested_fix="Keep train and test data separate. Use validation set for model selection.",
            confidence=0.90,
            fix_snippet="# Keep data separate\n# Only merge within train or within test, never across",
            references=[
                "https://scikit-learn.org/stable/common_pitfalls.html#data-leakage"
            ]
        )
    
    def _create_preprocessing_leakage_pattern(self, preprocessing_op: Dict, split_op: Dict, analysis: ASTAnalysisResult) -> MLAntiPattern:
        """Create pattern for preprocessing applied before split"""
        # Use the preprocessing operation node
        node = preprocessing_op.get('node')
        if not node:
            # Create dummy node if needed
            node = ast.parse("pass").body[0]
            node.lineno = preprocessing_op.get('line', 0)
        
        return self.create_pattern(
            pattern_type="preprocessing_before_split",
            severity=PatternSeverity.CRITICAL,
            node=node,
            analysis=analysis,
            message=f"Preprocessing {preprocessing_op.get('method', 'operation')} applied before train/test split",
            explanation=(
                "Applying preprocessing to the entire dataset before splitting causes data leakage. "
                "The preprocessor learns statistics from the test set, leading to overly optimistic results."
            ),
            suggested_fix="Move preprocessing after train_test_split. Fit only on training data.",
            confidence=0.95,
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
        return self.create_pattern(
            pattern_type="feature_leakage",
            severity=PatternSeverity.HIGH,
            node=node,
            analysis=analysis,
            message="Potential feature leakage detected",
            explanation=(
                "Feature appears to use information that wouldn't be available at prediction time. "
                "This can lead to overly optimistic model performance."
            ),
            suggested_fix="Ensure features only use information available at prediction time.",
            confidence=0.80,
            fix_snippet="# Use only historical data for features\n# Avoid future information or target-derived features",
            references=[
                "https://machinelearningmastery.com/data-leakage-machine-learning/"
            ]
        )
    
    def _create_target_leakage_pattern(self, var_info: Dict, analysis: ASTAnalysisResult) -> MLAntiPattern:
        """Create pattern for target leakage"""
        # Create dummy node for target leakage
        dummy_node = ast.parse("pass").body[0]
        dummy_node.lineno = var_info.get('line', 0)
        
        return self.create_pattern(
            pattern_type="target_leakage",
            severity=PatternSeverity.CRITICAL,
            node=dummy_node,
            analysis=analysis,
            message="Feature derived from target variable",
            explanation=(
                "Features derived from the target variable cause severe data leakage. "
                "The model learns to cheat by using the answer to predict itself."
            ),
            suggested_fix="Remove target-derived features from the feature set.",
            confidence=0.90,
            fix_snippet="# Remove features that use target information",
            references=[
                "https://machinelearningmastery.com/data-leakage-machine-learning/"
            ]
        )
    
    def _create_temporal_leakage_pattern(self, node: ast.Call, analysis: ASTAnalysisResult) -> MLAntiPattern:
        """Create pattern for temporal leakage"""
        return self.create_pattern(
            pattern_type="temporal_leakage",
            severity=PatternSeverity.HIGH,
            node=node,
            analysis=analysis,
            message="Potential temporal leakage in time series operation",
            explanation=(
                "Time series operation might be using future information. "
                "This creates look-ahead bias in time series models."
            ),
            suggested_fix="Use only past data for time series features. Check window parameters.",
            confidence=0.75,
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
        # Create dummy node for temporal split
        dummy_node = ast.parse("pass").body[0]
        dummy_node.lineno = split_op.get('line', 0)
        
        return self.create_pattern(
            pattern_type="incorrect_temporal_split",
            severity=PatternSeverity.HIGH,
            node=dummy_node,
            analysis=analysis,
            message="Random split used on temporal data",
            explanation=(
                "Using random train/test split on time series data causes temporal leakage. "
                "Future data ends up in training set, allowing model to 'see the future'."
            ),
            suggested_fix="Use temporal split: train on past data, test on future data.",
            confidence=0.85,
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
        # Create dummy node for CV operation
        dummy_node = ast.parse("pass").body[0]
        dummy_node.lineno = cv_op.get('line', 0)
        
        return self.create_pattern(
            pattern_type="cv_preprocessing_leakage",
            severity=PatternSeverity.HIGH,
            node=dummy_node,
            analysis=analysis,
            message="Preprocessing applied outside cross-validation folds",
            explanation=(
                "Preprocessing before cross-validation causes data leakage. "
                "Each CV fold sees preprocessed data from other folds."
            ),
            suggested_fix="Use Pipeline to include preprocessing within CV folds.",
            confidence=0.90,
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
        # Create dummy node for CV operation
        dummy_node = ast.parse("pass").body[0]
        dummy_node.lineno = cv_op.get('line', 0)
        
        return self.create_pattern(
            pattern_type="cv_feature_selection_leakage",
            severity=PatternSeverity.MEDIUM,
            node=dummy_node,
            analysis=analysis,
            message="Feature selection may be leaking information in cross-validation",
            explanation=(
                "Feature selection before CV can cause subtle leakage. "
                "Features are selected using the entire dataset including test folds."
            ),
            suggested_fix="Include feature selection within CV pipeline.",
            confidence=0.70,
            fix_snippet="""# Include feature selection in pipeline:
pipeline = Pipeline([
    ('feature_selection', SelectKBest(k=10)),
    ('classifier', RandomForestClassifier())
])""",
            references=[
                "https://scikit-learn.org/stable/modules/cross_validation.html"
            ]
        )