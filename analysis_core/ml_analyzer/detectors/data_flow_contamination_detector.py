"""
Data Flow Contamination Detector - Sprint 4 Core Enhancement

Detects data flow contamination patterns that cause subtle data leakage through
pipeline ordering, feature engineering, and preprocessing operations.

This detector extends the sophisticated semantic analysis from TestSetContaminationDetector
to identify contamination patterns that occur through improper data flow in ML pipelines.
"""

import ast
from typing import Dict, List, Any, Optional, Set, Union

from .base_detector import BaseMLDetector, MLAntiPattern, PatternSeverity
from .test_contamination_detector import VariableSemanticTracker
from ..ast_engine import ASTAnalysisResult


class DataFlowSemanticTracker(VariableSemanticTracker):
    """
    Extended semantic tracker for data flow analysis.

    Builds upon VariableSemanticTracker to add specific tracking
    for pipeline operations, feature engineering, and data transformations.
    """

    def __init__(self):
        super().__init__()
        # Additional tracking for data flow patterns
        self.pipeline_operations = []  # chronological pipeline steps
        self.feature_engineering_ops = []  # feature creation operations
        self.global_statistics = {}  # global stats computations
        self.data_transformations = {}  # transformation operations

    def track_pipeline_operation(
        self,
        operation_type: str,
        node: ast.AST,
        target_vars: List[str] = None,
        source_vars: List[str] = None,
    ):
        """Track pipeline operations for contamination analysis"""
        self.pipeline_operations.append(
            {
                "type": operation_type,
                "node": node,
                "line": getattr(node, "lineno", 0),
                "target_vars": target_vars or [],
                "source_vars": source_vars or [],
                "function": (
                    self._get_function_name(node)
                    if isinstance(node, ast.Call)
                    else None
                ),
            }
        )

    def track_feature_engineering(self, node: ast.Assign, feature_type: str):
        """Track feature engineering operations"""
        target_vars = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                target_vars.append(target.id)

        self.feature_engineering_ops.append(
            {
                "type": feature_type,
                "node": node,
                "line": node.lineno,
                "target_vars": target_vars,
                "source_operation": (
                    node.value if isinstance(node.value, ast.Call) else None
                ),
            }
        )

    def track_global_statistics(self, var_name: str, stat_type: str, node: ast.Call):
        """Track global statistics computations"""
        self.global_statistics[var_name] = {
            "stat_type": stat_type,
            "node": node,
            "line": node.lineno,
            "affects_variables": self._extract_affected_variables(node),
        }

    def get_pipeline_sequence(self) -> List[Dict]:
        """Get pipeline operations in chronological order"""
        return sorted(self.pipeline_operations, key=lambda x: x.get("line", 0))

    def get_feature_engineering_operations(self) -> List[Dict]:
        """Get all feature engineering operations"""
        return sorted(self.feature_engineering_ops, key=lambda x: x.get("line", 0))

    def has_global_statistics_before_split(self) -> bool:
        """Check if global statistics are computed before data split"""
        split_line = self._get_first_split_line()
        if not split_line:
            return False

        for stat_name, stat_info in self.global_statistics.items():
            if stat_info["line"] < split_line:
                return True
        return False

    def _extract_affected_variables(self, node: ast.Call) -> List[str]:
        """Extract variables affected by a statistics operation"""
        affected = []
        for arg in ast.walk(node):
            if isinstance(arg, ast.Name):
                affected.append(arg.id)
        return affected

    def _get_first_split_line(self) -> Optional[int]:
        """Get line number of first data split operation"""
        split_ops = self.get_split_operations()
        if split_ops:
            return min(op.get("line", float("inf")) for op in split_ops)
        return None


class DataFlowContaminationDetector(BaseMLDetector):
    """
    Detects data flow contamination patterns - Sprint 4 core enhancement.

    Data flow contamination occurs when information flows inappropriately through
    ML pipelines, causing subtle data leakage that's hard to detect with simple
    syntax checking. This detector uses sophisticated semantic analysis to identify:

    1. Pipeline contamination (preprocessing before split)
    2. Feature engineering leakage (global statistics)
    3. Cross-validation contamination (preprocessing outside folds)
    4. Temporal contamination in sequential data
    5. Resource accumulation patterns

    Extends TestSetContaminationDetector's proven semantic analysis approach.
    """

    def __init__(self):
        super().__init__()

        # Enhanced variable tracker for data flow analysis
        self.data_flow_tracker = DataFlowSemanticTracker()

        # Methods that fit on data (potential leakage sources)
        self.fit_methods = {"fit", "fit_transform"}

        # Operations that cause pipeline contamination
        self.contaminating_operations = {
            "fit",
            "fit_transform",
            "normalize",
            "standardize",
            "scale",
            "transform",
            "encode",
            "select_features",
            "impute",
        }

        # Global statistics functions
        self.global_stat_functions = {
            "mean",
            "std",
            "var",
            "min",
            "max",
            "median",
            "mode",
            "quantile",
            "percentile",
            "describe",
            "info",
            "nunique",
        }

        # Feature engineering patterns that indicate potential leakage
        self.leaky_feature_patterns = {
            "target_encoding",
            "frequency_encoding",
            "count_encoding",
            "group_statistics",
            "aggregation",
            "rollup",
        }

    def detect_patterns(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Main entry point for data flow contamination detection"""
        patterns = []

        # Reset and build enhanced semantic model
        self.data_flow_tracker.reset()
        self._build_data_flow_model(analysis)

        # Apply data flow contamination detection algorithms
        patterns.extend(self._detect_pipeline_contamination(analysis))
        patterns.extend(self._detect_feature_engineering_leakage(analysis))
        patterns.extend(self._detect_global_statistics_leakage(analysis))
        patterns.extend(self._detect_cross_validation_contamination(analysis))
        patterns.extend(self._detect_temporal_data_flow_issues(analysis))

        return patterns

    def _build_data_flow_model(self, analysis: ASTAnalysisResult):
        """Build enhanced semantic model for data flow analysis"""
        # Build basic semantic model first (simplified approach)
        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Assign):
                # Basic variable tracking
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.data_flow_tracker.assign_variable(
                            target.id, node.value, node.lineno
                        )
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        # Handle tuple unpacking for splits
                        if isinstance(node.value, ast.Call) and self._is_split_function(
                            node.value
                        ):
                            elements = target.elts
                            for i, elt in enumerate(elements):
                                if isinstance(elt, ast.Name):
                                    var_type = self._infer_split_variable_type(
                                        elt.id, i
                                    )
                                    self.data_flow_tracker.assign_split_variable(
                                        elt.id,
                                        var_type,
                                        node.value,
                                        (
                                            elt.lineno
                                            if hasattr(elt, "lineno")
                                            else node.lineno
                                        ),
                                    )

        # Then add data flow specific tracking
        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Assign):
                self._track_data_flow_assignment(node)
            elif isinstance(node, ast.Call):
                self._track_data_flow_call(node)

    def _track_data_flow_assignment(self, node: ast.Assign):
        """Track assignments for data flow analysis"""
        # Check if this is feature engineering
        if self._is_feature_engineering_assignment(node):
            feature_type = self._classify_feature_engineering(node)
            self.data_flow_tracker.track_feature_engineering(node, feature_type)

        # Check if this involves global statistics
        if isinstance(node.value, ast.Call) and self._is_global_statistics_call(
            node.value
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    stat_type = self.get_function_name(node.value)
                    self.data_flow_tracker.track_global_statistics(
                        target.id, stat_type, node.value
                    )

    def _track_data_flow_call(self, node: ast.Call):
        """Track function calls for pipeline operation detection"""
        func_name = self.get_function_name(node)

        if func_name in self.contaminating_operations:
            # This is a pipeline operation that could cause contamination
            operation_type = "preprocessing_operation"
            if func_name in self.fit_methods:
                operation_type = "preprocessing_fit"

            self.data_flow_tracker.track_pipeline_operation(
                operation_type,
                node,
                target_vars=self._extract_target_variables(node),
                source_vars=self._extract_source_variables(node),
            )

    def _detect_pipeline_contamination(
        self, analysis: ASTAnalysisResult
    ) -> List[MLAntiPattern]:
        """
        Detect preprocessing applied before train/test split.

        Extends TestSetContaminationDetector logic with enhanced pipeline tracking.
        Enhanced with surgical precision to avoid false positives on split data.
        """
        patterns = []

        # Get chronological pipeline sequence
        pipeline_sequence = self.data_flow_tracker.get_pipeline_sequence()
        split_operations = self.data_flow_tracker.get_split_operations()

        if not split_operations:
            return patterns  # No splits found

        first_split_line = min(op.get("line", float("inf")) for op in split_operations)

        # Find contaminating operations that have splits AFTER them
        for operation in pipeline_sequence:
            if operation["type"] in ["preprocessing_fit", "preprocessing_operation"]:

                # Look for splits that happen AFTER this operation
                future_splits = [
                    split
                    for split in split_operations
                    if split.get("line", 0) > operation["line"]
                ]

                if future_splits:
                    # SURGICAL FIX: Check if operation works on unsplit data
                    if self._operates_on_unsplit_data(operation, analysis):
                        # This preprocessing happens on unsplit data before a split - CONTAMINATION!
                        patterns.append(
                            self._create_pipeline_contamination_pattern(
                                operation, analysis
                            )
                        )
                    # If it operates on split data (like X_train_raw), it's safe - no pattern

        return patterns

    def _detect_feature_engineering_leakage(
        self, analysis: ASTAnalysisResult
    ) -> List[MLAntiPattern]:
        """
        Detect feature engineering that uses global information.

        Enhanced with surgical precision to distinguish training-only operations.
        """
        patterns = []

        feature_ops = self.data_flow_tracker.get_feature_engineering_operations()

        for feature_op in feature_ops:
            if self._uses_global_information(feature_op):
                # SEMANTIC ENHANCEMENT: Check if this involves target variable leakage
                if self._involves_target_leakage(feature_op, analysis):
                    # SURGICAL FIX: Check if feature engineering operates on training data only
                    # Use same unified logic as pipeline contamination
                    if self._operates_on_unsplit_data(feature_op, analysis):
                        patterns.append(
                            self._create_feature_engineering_leakage_pattern(
                                feature_op, analysis
                            )
                        )
                # If it's just feature statistics (non-target) on unsplit data, it's legitimate

        return patterns

    def _involves_target_leakage(self, feature_op: Dict, analysis: ASTAnalysisResult) -> bool:
        """
        SEMANTIC ENHANCEMENT: Determine if feature engineering involves target variable.
        
        Distinguishes between:
        - Target leakage: statistics on y variable (CRITICAL)
        - Feature engineering: statistics on X variables (OK)
        """
        operation_node = feature_op.get("node")
        if not operation_node:
            return True  # Conservative: assume problematic if unclear
            
        # Common target variable names
        target_indicators = {
            'target', 'y', 'label', 'outcome', 'response', 'dependent',
            'claim', 'fraud', 'churn', 'conversion', 'class'
        }
        
        # Analyze the AST node to find what variables are being grouped/aggregated
        if isinstance(operation_node, ast.Call):
            # Look for groupby operations and check what columns are involved
            return self._check_target_involvement_in_call(operation_node, target_indicators)
        elif isinstance(operation_node, ast.Assign):
            return self._check_target_involvement_in_assignment(operation_node, target_indicators)
            
        return True  # Conservative default

    def _check_target_involvement_in_call(self, node: ast.Call, target_indicators: set) -> bool:
        """Check if a function call involves target variables"""
        # Look for column names in the call arguments
        for arg in node.args:
            if self._contains_target_reference(arg, target_indicators):
                return True
        
        # Check keyword arguments  
        for kw in node.keywords:
            if self._contains_target_reference(kw.value, target_indicators):
                return True
                
        return False

    def _check_target_involvement_in_assignment(self, node: ast.Assign, target_indicators: set) -> bool:
        """Check if an assignment involves target variables"""
        if isinstance(node.value, ast.Call):
            return self._check_target_involvement_in_call(node.value, target_indicators)
        return False

    def _contains_target_reference(self, node: ast.AST, target_indicators: set) -> bool:
        """Check if an AST node contains references to target variables"""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value.lower() in target_indicators
        elif isinstance(node, ast.Name):
            return node.id.lower() in target_indicators
        elif isinstance(node, ast.List) or isinstance(node, ast.Tuple):
            for elt in node.elts:
                if self._contains_target_reference(elt, target_indicators):
                    return True
        elif isinstance(node, ast.Subscript):
            return self._contains_target_reference(node.slice, target_indicators)
            
        return False

    def _detect_global_statistics_leakage(
        self, analysis: ASTAnalysisResult
    ) -> List[MLAntiPattern]:
        """
        Detect global statistics computed before train/test split.

        Statistics like mean, std computed on entire dataset before splitting.
        """
        patterns = []

        if self.data_flow_tracker.has_global_statistics_before_split():
            for (
                stat_name,
                stat_info,
            ) in self.data_flow_tracker.global_statistics.items():
                split_line = self.data_flow_tracker._get_first_split_line()
                if stat_info["line"] < split_line:
                    patterns.append(
                        self._create_global_statistics_leakage_pattern(
                            stat_name, stat_info, analysis
                        )
                    )

        return patterns

    def _detect_cross_validation_contamination(
        self, analysis: ASTAnalysisResult
    ) -> List[MLAntiPattern]:
        """
        Detect preprocessing outside cross-validation folds.

        Enhanced with surgical precision to avoid false positives on split data.
        """
        patterns = []

        # Find CV operations (reuse existing logic)
        cv_operations = self._find_cv_operations(analysis)

        for cv_op in cv_operations:
            # Check if preprocessing happens outside CV folds
            contaminating_ops = self._find_preprocessing_outside_cv(cv_op)
            for op in contaminating_ops:
                # SURGICAL FIX: Only flag if operating on unsplit data
                if self._operates_on_unsplit_data(op, analysis):
                    patterns.append(
                        self._create_cv_contamination_pattern(cv_op, op, analysis)
                    )
                # If it operates on split data (like X_train), it's safe - no pattern

        return patterns

    def _detect_temporal_data_flow_issues(
        self, analysis: ASTAnalysisResult
    ) -> List[MLAntiPattern]:
        """
        Detect temporal contamination in sequential data processing.

        Identifies look-ahead bias and improper temporal ordering.
        """
        patterns = []

        # Check for temporal operations in wrong order
        temporal_ops = self._find_temporal_operations(analysis)

        for temp_op in temporal_ops:
            if self._causes_temporal_contamination(temp_op):
                patterns.append(
                    self._create_temporal_contamination_pattern(temp_op, analysis)
                )

        return patterns

    # Helper methods for classification and detection
    def _is_feature_engineering_assignment(self, node: ast.Assign) -> bool:
        """Check if assignment creates engineered features"""

        # Check for direct function calls
        if isinstance(node.value, ast.Call):
            func_name = self.get_function_name(node.value)
            if (
                func_name in self.leaky_feature_patterns
                or "encode" in func_name
                or "transform" in func_name
                or "group" in func_name
            ):
                return True

        # Check for chained operations like X.groupby().mean()
        if isinstance(node.value, ast.Attribute):
            # Check if this is a method call on a groupby result
            if isinstance(node.value.value, ast.Call):
                inner_func = self.get_function_name(node.value.value)
                if "groupby" in inner_func:
                    return True

        # Check for subscript operations like X.groupby()['target'].mean()
        if isinstance(node.value, ast.Call) and isinstance(
            node.value.func, ast.Attribute
        ):
            # This handles X.groupby()['target'].mean() pattern
            if isinstance(node.value.func.value, ast.Subscript):
                subscript_value = node.value.func.value.value
                if isinstance(subscript_value, ast.Call):
                    inner_func = self.get_function_name(subscript_value)
                    if "groupby" in inner_func:
                        return True

        return False

    def _classify_feature_engineering(self, node: ast.Assign) -> str:
        """Classify the type of feature engineering"""
        if isinstance(node.value, ast.Call):
            func_name = self.get_function_name(node.value)
            if "encode" in func_name:
                return "encoding"
            elif "group" in func_name or "agg" in func_name:
                return "aggregation"
            elif "transform" in func_name:
                return "transformation"
            elif "mean" in func_name:
                # Check if this is target encoding (groupby().mean())
                return "target_encoding"

        # Check for complex patterns
        node_str = (
            ast.unparse(node.value) if hasattr(ast, "unparse") else str(node.value)
        )
        if "groupby" in node_str.lower():
            return "target_encoding"

        return "unknown"

    def _is_global_statistics_call(self, node: ast.Call) -> bool:
        """Check if call computes global statistics"""
        func_name = self.get_function_name(node)
        return func_name in self.global_stat_functions

    def _affects_split_data(
        self, operation: Dict, split_operations: List[Dict]
    ) -> bool:
        """Check if operation affects the same data that gets split"""
        # Simplified check - in practice would need more sophisticated analysis
        operation_vars = set(operation.get("source_vars", []))

        for split_op in split_operations:
            split_vars = set(split_op.get("variables", []))
            if operation_vars.intersection(split_vars):
                return True

        return False

    def _uses_global_information(self, feature_op: Dict) -> bool:
        """Check if feature engineering uses global dataset information"""
        if feature_op["type"] in ["aggregation", "encoding"]:
            return True

        # Check if the operation involves global statistics
        if feature_op.get("source_operation"):
            func_name = self.get_function_name(feature_op["source_operation"])
            return func_name in self.global_stat_functions

        return False

    def _find_cv_operations(self, analysis: ASTAnalysisResult) -> List[Dict]:
        """Find cross-validation operations (reuse from TestSetContaminationDetector)"""
        cv_ops = []
        cv_functions = [
            "cross_val_score",
            "cross_validate",
            "GridSearchCV",
            "RandomizedSearchCV",
        ]

        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Call):
                func_name = self.get_function_name(node)
                if func_name in cv_functions:
                    cv_ops.append(
                        {"function": func_name, "node": node, "line": node.lineno}
                    )

        return cv_ops

    def _find_preprocessing_outside_cv(self, cv_op: Dict) -> List[Dict]:
        """Find preprocessing operations outside CV folds"""
        cv_line = cv_op["line"]
        pipeline_ops = self.data_flow_tracker.get_pipeline_sequence()

        return [
            op
            for op in pipeline_ops
            if op["line"] < cv_line and op["type"] == "preprocessing_fit"
        ]

    def _find_temporal_operations(self, analysis: ASTAnalysisResult) -> List[Dict]:
        """Find temporal operations that might cause contamination"""
        temporal_ops = []
        temporal_functions = [
            "shift",
            "rolling",
            "expanding",
            "resample",
            "lag",
            "lead",
        ]

        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Call):
                func_name = self.get_function_name(node)
                if any(temp_func in func_name for temp_func in temporal_functions):
                    temporal_ops.append(
                        {"function": func_name, "node": node, "line": node.lineno}
                    )

        return temporal_ops

    def _causes_temporal_contamination(self, temp_op: Dict) -> bool:
        """Check if temporal operation causes future leakage"""
        func_name = temp_op["function"]
        node = temp_op["node"]

        # Check for negative shifts (future data)
        if "shift" in func_name:
            return self._has_negative_shift(node)

        # Check for unsafe rolling operations
        if "rolling" in func_name:
            return self._has_unsafe_rolling_window(node)

        return False

    def _has_negative_shift(self, node: ast.Call) -> bool:
        """Check if shift operation uses negative periods (future data)"""
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                if arg.value < 0:
                    return True

        for keyword in node.keywords:
            if keyword.arg == "periods" and isinstance(keyword.value, ast.Constant):
                if isinstance(keyword.value.value, int) and keyword.value.value < 0:
                    return True

        return False

    def _has_unsafe_rolling_window(self, node: ast.Call) -> bool:
        """Check if rolling operation has unsafe parameters"""
        # Check if rolling window includes future data
        for keyword in node.keywords:
            if keyword.arg == "center" and isinstance(keyword.value, ast.Constant):
                if keyword.value.value is True:  # center=True uses future data
                    return True
        return False

    def _extract_target_variables(self, node: ast.Call) -> List[str]:
        """Extract target variables from function call"""
        # Simplified implementation
        return []

    def _extract_source_variables(self, node: ast.Call) -> List[str]:
        """Extract source variables from function call"""
        # Simplified implementation
        return []

    def _operates_on_unsplit_data(
        self, operation: Dict, analysis: ASTAnalysisResult
    ) -> bool:
        """
        SURGICAL PRECISION FIX: Determine if operation works on unsplit vs split data.

        Key insight: fit_transform(X) is contamination, fit_transform(X_train_raw) is safe.
        This method traces data lineage to distinguish between:
        - Unsplit data (original dataset): CONTAMINATION
        - Split data (result of train_test_split): SAFE

        Enhanced to handle both direct operations and feature engineering operations.
        """
        operation_node = operation.get("node")

        # Handle feature engineering operations (like groupby)
        if isinstance(operation_node, ast.Assign):
            return self._operates_on_unsplit_data_assignment(operation_node, analysis)

        # Handle direct operations (like fit_transform)
        elif isinstance(operation_node, ast.Call):
            return self._operates_on_unsplit_data_call(operation_node, analysis)

        return True  # Conservative: assume contamination if unclear

    def _operates_on_unsplit_data_call(
        self, node: ast.Call, analysis: ASTAnalysisResult
    ) -> bool:
        """Handle direct function calls like fit_transform"""
        # Extract the input variable name from the operation
        input_variable = self._extract_operation_input_variable(node)
        if not input_variable:
            return True  # Conservative: assume contamination if can't identify input

        # Trace the lineage of the input variable
        variable_lineage = self._trace_variable_lineage(input_variable, analysis)

        # Check if the variable originates from a split operation
        return not self._originates_from_split(variable_lineage)

    def _operates_on_unsplit_data_assignment(
        self, node: ast.Assign, analysis: ASTAnalysisResult
    ) -> bool:
        """Handle feature engineering assignments like groupby operations"""
        # Extract all variables used in the assignment value
        used_variables = self._extract_all_variables_from_node(node.value)

        if not used_variables:
            return True  # Conservative: assume contamination if no variables found

        # Check if ALL used variables come from split operations
        all_from_splits = True
        for var_name in used_variables:
            variable_lineage = self._trace_variable_lineage(var_name, analysis)
            if not self._originates_from_split(variable_lineage):
                all_from_splits = False
                break

        # If all variables come from splits, it's safe (not contamination)
        return not all_from_splits

    def _extract_all_variables_from_node(self, node: ast.AST) -> List[str]:
        """Extract all variable names used in an AST node"""
        variables = []
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                variables.append(child.id)
        return list(set(variables))  # Remove duplicates

    def _extract_operation_input_variable(self, node: ast.Call) -> Optional[str]:
        """Extract the primary input variable from a preprocessing operation"""
        if not node.args:
            return None

        # First argument is typically the data being processed
        first_arg = node.args[0]

        if isinstance(first_arg, ast.Name):
            return first_arg.id
        elif isinstance(first_arg, ast.Subscript):
            # Handle cases like scaler.fit_transform(X[:, features])
            if isinstance(first_arg.value, ast.Name):
                return first_arg.value.id
        elif isinstance(first_arg, ast.Attribute):
            # Handle cases like scaler.fit_transform(dataset.values)
            return self._extract_variable_from_attribute(first_arg)

        return None

    def _extract_variable_from_attribute(self, node: ast.Attribute) -> Optional[str]:
        """Extract variable name from attribute access like X.values"""
        if isinstance(node.value, ast.Name):
            return node.value.id
        return None

    def _trace_variable_lineage(
        self, variable_name: str, analysis: ASTAnalysisResult
    ) -> Dict:
        """
        Trace the lineage of a variable to understand its data origin.

        Returns information about how the variable was created:
        - origin_type: 'split', 'original', 'derived', 'unknown'
        - source_line: where it was defined
        - source_operation: what operation created it
        """
        lineage = {
            "variable": variable_name,
            "origin_type": "unknown",
            "source_line": None,
            "source_operation": None,
        }

        # Search for variable assignments in AST
        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Assign):
                # Check if this assigns to our target variable
                for target in node.targets:
                    if self._target_matches_variable(target, variable_name):
                        lineage["source_line"] = node.lineno
                        lineage["source_operation"] = node.value
                        lineage["origin_type"] = self._classify_assignment_origin(node)
                        break

        return lineage

    def _target_matches_variable(self, target: ast.AST, variable_name: str) -> bool:
        """Check if assignment target matches the variable we're tracing"""
        if isinstance(target, ast.Name):
            return target.id == variable_name
        elif isinstance(target, (ast.Tuple, ast.List)):
            # Handle tuple unpacking like X_train, X_test = ...
            for elt in target.elts:
                if isinstance(elt, ast.Name) and elt.id == variable_name:
                    return True
        return False

    def _classify_assignment_origin(self, node: ast.Assign) -> str:
        """Classify how a variable was created to determine if it's split or original data"""
        if isinstance(node.value, ast.Call):
            func_name = self.get_function_name(node.value)

            # Check if it's a split function
            if func_name in ["train_test_split", "cross_val_split", "split"]:
                return "split"

            # Check if it's a data loading function
            if func_name in ["read_csv", "load_data", "get_data", "load_dataset"]:
                return "original"

            # Check if it's a transformation of existing data
            if func_name in ["transform", "fit_transform", "select", "filter"]:
                return "derived"

        elif isinstance(node.value, ast.Subscript):
            # Handle slicing like X = data[:, features]
            return "derived"

        elif isinstance(node.value, ast.Attribute):
            # Handle attribute access like X = data.drop('target')
            return "derived"

        return "unknown"

    def _originates_from_split(self, lineage: Dict) -> bool:
        """Check if variable lineage indicates it comes from a split operation"""
        return lineage.get("origin_type") == "split"

    def _is_split_function(self, node: ast.Call) -> bool:
        """Check if a function call is a data splitting function"""
        func_name = self.get_function_name(node)
        split_functions = [
            "train_test_split",
            "cross_val_split",
            "split",
            "StratifiedShuffleSplit",
        ]
        return func_name in split_functions

    def _infer_split_variable_type(self, var_name: str, position: int) -> str:
        """Infer the type of variable from train_test_split unpacking"""
        var_lower = var_name.lower()

        if "train" in var_lower:
            return "train"
        elif "test" in var_lower or "val" in var_lower:
            return "test"
        elif position in [0, 2]:  # Typically X_train, y_train positions
            return "train"
        elif position in [1, 3]:  # Typically X_test, y_test positions
            return "test"
        else:
            return "unknown"

    # Pattern creation methods using inherited create_pattern helper
    def _create_pipeline_contamination_pattern(
        self, operation: Dict, analysis: ASTAnalysisResult
    ) -> MLAntiPattern:
        """Create pattern for pipeline contamination"""
        return self.create_pattern(
            pattern_type="pipeline_contamination",
            severity=PatternSeverity.HIGH,
            node=operation["node"],
            analysis=analysis,
            message=f"Pipeline operation '{operation.get('function', 'unknown')}' applied before train/test split",
            explanation=(
                f"The operation '{operation.get('function', 'unknown')}' is applied to the entire dataset "
                "before splitting into train/test sets. This causes data leakage because the preprocessing "
                "learns global statistics that include information from the test set."
            ),
            suggested_fix=(
                "Move the preprocessing operation after train_test_split. "
                "Fit the preprocessor only on training data, then transform both train and test sets separately."
            ),
            confidence=0.90,
            fix_snippet="""# Correct approach:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
preprocessor = StandardScaler()
X_train_processed = preprocessor.fit_transform(X_train)  # Fit only on training data
X_test_processed = preprocessor.transform(X_test)  # Only transform test data""",
            references=[
                "https://scikit-learn.org/stable/common_pitfalls.html#data-leakage",
                "https://machinelearningmastery.com/data-leakage-machine-learning/",
            ],
        )

    def _create_feature_engineering_leakage_pattern(
        self, feature_op: Dict, analysis: ASTAnalysisResult
    ) -> MLAntiPattern:
        """Create pattern for feature engineering leakage"""
        return self.create_pattern(
            pattern_type="feature_engineering_leakage",
            severity=PatternSeverity.MEDIUM,
            node=feature_op["node"],
            analysis=analysis,
            message=f"Feature engineering using global information",
            explanation=(
                f"Feature engineering operation of type '{feature_op['type']}' uses "
                "statistics computed on the entire dataset, including the test set. "
                "This can lead to subtle data leakage and overly optimistic results."
            ),
            suggested_fix=(
                "Compute feature engineering statistics only on the training set, "
                "then apply the same transformations to the test set."
            ),
            confidence=0.80,
            fix_snippet="""# Correct approach for feature engineering:
# Compute statistics only on training data
train_mean = X_train['feature'].mean()
train_std = X_train['feature'].std()

# Apply to both sets
X_train['feature_normalized'] = (X_train['feature'] - train_mean) / train_std
X_test['feature_normalized'] = (X_test['feature'] - train_mean) / train_std""",
            references=[
                "https://machinelearningmastery.com/data-leakage-machine-learning/",
                "https://towardsdatascience.com/data-leakage-in-machine-learning-10bdd3eec742",
            ],
        )

    def _create_global_statistics_leakage_pattern(
        self, stat_name: str, stat_info: Dict, analysis: ASTAnalysisResult
    ) -> MLAntiPattern:
        """Create pattern for global statistics leakage"""
        return self.create_pattern(
            pattern_type="global_statistics_leakage",
            severity=PatternSeverity.HIGH,
            node=stat_info["node"],
            analysis=analysis,
            message=f"Global statistics '{stat_info['stat_type']}' computed before train/test split",
            explanation=(
                f"The statistic '{stat_info['stat_type']}' is computed on the entire dataset "
                "before splitting into train and test sets. These global statistics include "
                "information from the test set, causing data leakage."
            ),
            suggested_fix=(
                "Move statistics computation after train_test_split. "
                "Compute statistics only on training data."
            ),
            confidence=0.95,
            fix_snippet=f"""# Correct approach:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
{stat_name} = X_train.{stat_info['stat_type']}()  # Compute only on training data""",
            references=[
                "https://scikit-learn.org/stable/common_pitfalls.html#data-leakage"
            ],
        )

    def _create_cv_contamination_pattern(
        self, cv_op: Dict, preprocessing_op: Dict, analysis: ASTAnalysisResult
    ) -> MLAntiPattern:
        """Create pattern for cross-validation contamination"""
        return self.create_pattern(
            pattern_type="cv_preprocessing_contamination",
            severity=PatternSeverity.HIGH,
            node=preprocessing_op["node"],
            analysis=analysis,
            message=f"Preprocessing '{preprocessing_op.get('function', 'operation')}' applied outside CV folds",
            explanation=(
                f"The preprocessing operation '{preprocessing_op.get('function', 'operation')}' "
                "is applied before cross-validation. This causes data leakage because each CV fold "
                "sees data that has been preprocessed using statistics from other folds."
            ),
            suggested_fix=(
                "Use Pipeline to include preprocessing within CV folds, or apply preprocessing "
                "separately within each fold."
            ),
            confidence=0.90,
            fix_snippet="""# Correct CV with Pipeline:
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('preprocessing', StandardScaler()),
    ('model', RandomForestClassifier())
])
cross_val_score(pipeline, X, y, cv=5)  # Preprocessing happens within each fold""",
            references=[
                "https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-pipelines"
            ],
        )

    def _create_temporal_contamination_pattern(
        self, temp_op: Dict, analysis: ASTAnalysisResult
    ) -> MLAntiPattern:
        """Create pattern for temporal contamination"""
        return self.create_pattern(
            pattern_type="temporal_data_flow_contamination",
            severity=PatternSeverity.HIGH,
            node=temp_op["node"],
            analysis=analysis,
            message=f"Temporal operation '{temp_op['function']}' may cause future leakage",
            explanation=(
                f"The temporal operation '{temp_op['function']}' appears to use future information "
                "in a time series context. This creates look-ahead bias where the model has access "
                "to information that wouldn't be available at prediction time."
            ),
            suggested_fix=(
                "Ensure temporal operations only use past data. "
                "Check shift directions and rolling window parameters."
            ),
            confidence=0.80,
            fix_snippet="""# Correct temporal operations:
# Use positive shifts for past data
df['lag_1'] = df['value'].shift(1)  # Previous value (safe)
df['rolling_mean'] = df['value'].rolling(window=3).mean()  # Past 3 values (safe)

# Avoid negative shifts
# df['future'] = df['value'].shift(-1)  # Future value (dangerous!)""",
            references=[
                "https://otexts.com/fpp3/",
                "https://machinelearningmastery.com/time-series-forecasting-supervised-learning/",
            ],
        )
