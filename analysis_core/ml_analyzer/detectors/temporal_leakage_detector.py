"""
Temporal Leakage Detector - High-Value ML Pattern Detection

Detects temporal data leakage patterns that cause look-ahead bias in time series
and sequential data modeling. This detector identifies when models gain access
to future information that wouldn't be available at prediction time.

Key Patterns Detected:
1. Negative shifts (future_value = data.shift(-1))
2. Centered rolling windows (rolling(center=True))
3. Global statistics on temporal data
4. Random splits on time series data
5. Forward-looking feature engineering

Built on the proven data lineage architecture from DataFlowContaminationDetector.
"""

import ast
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

from .base_detector import BaseMLDetector, MLAntiPattern, PatternSeverity
from .data_flow_contamination_detector import DataFlowSemanticTracker
from ..ast_engine import ASTAnalysisResult


class TemporalSemanticTracker(DataFlowSemanticTracker):
    """
    Extended semantic tracker for temporal data analysis.

    Builds upon DataFlowSemanticTracker to add specific tracking
    for temporal operations, time series patterns, and chronological data flow.
    """

    def __init__(self):
        super().__init__()
        # Additional tracking for temporal patterns
        self.temporal_operations = []  # shift, rolling, expanding operations
        self.time_column_references = []  # timestamp, date, datetime columns
        self.split_operations_temporal = []  # train_test_split on temporal data
        self.global_temporal_stats = {}  # global stats on time series
        self.temporal_feature_engineering = []  # forward-looking feature creation

    def track_temporal_operation(
        self, operation_type: str, node: ast.AST, parameters: Dict[str, Any] = None
    ):
        """Track temporal operations like shift, rolling, etc."""
        self.temporal_operations.append(
            {
                "type": operation_type,
                "node": node,
                "line": getattr(node, "lineno", 0),
                "parameters": parameters or {},
                "function": (
                    self._get_function_name(node)
                    if isinstance(node, ast.Call)
                    else None
                ),
            }
        )

    def track_time_column_reference(self, column_name: str, node: ast.AST):
        """Track references to temporal columns"""
        self.time_column_references.append(
            {"column": column_name, "node": node, "line": getattr(node, "lineno", 0)}
        )

    def track_temporal_split(self, node: ast.Call, split_type: str):
        """Track data splitting operations on temporal data"""
        self.split_operations_temporal.append(
            {
                "type": split_type,
                "node": node,
                "line": node.lineno,
                "function": self._get_function_name(node),
            }
        )

    def get_temporal_operations(self) -> List[Dict]:
        """Get all temporal operations in chronological order"""
        return sorted(self.temporal_operations, key=lambda x: x.get("line", 0))

    def has_future_references(self) -> bool:
        """Check if any temporal operations reference future data"""
        for op in self.temporal_operations:
            if op["type"] in ["negative_shift", "centered_rolling", "forward_looking"]:
                return True
        return False


class TemporalLeakageDetector(BaseMLDetector):
    """
    Detects temporal data leakage patterns in time series and sequential data.

    Temporal leakage occurs when models gain access to future information during
    training, creating look-ahead bias that leads to overly optimistic performance
    that doesn't generalize to real-world deployment.

    This detector identifies:
    - Future data access through negative shifts
    - Centered rolling windows that peek into future
    - Global statistics computed across entire time range
    - Random splits on temporal data (should use temporal splits)
    - Forward-looking feature engineering

    Extends the proven semantic analysis approach from DataFlowContaminationDetector.
    """

    def __init__(self, analysis_context: str = 'time-series'):
        super().__init__()

        # Context-aware analysis configuration
        self.analysis_context = analysis_context  # 'time-series', 'computer-vision', 'nlp', etc.
        
        # Enhanced temporal tracker
        self.temporal_tracker = TemporalSemanticTracker()

        # Temporal-specific patterns to detect
        self.temporal_functions = {
            "shift",
            "rolling",
            "expanding",
            "resample",
            "groupby",
            "lag",
            "lead",
            "diff",
            "pct_change",
        }

        # Functions that suggest temporal data
        self.time_indicators = {
            "timestamp",
            "datetime",
            "date",
            "time",
            "period",
            "dt",
            "to_datetime",
            "date_range",
            "period_range",
        }

        # Split functions that should not be used on temporal data
        self.problematic_split_functions = {
            "train_test_split",
            "cross_val_score",
            "KFold",
            "StratifiedKFold",
        }

        # Safe temporal split functions
        self.safe_temporal_splits = {
            "TimeSeriesSplit",
            "temporal_split",
            "chronological_split",
        }

    def detect_patterns(self, analysis: ASTAnalysisResult) -> List[MLAntiPattern]:
        """Main entry point for temporal leakage detection"""
        patterns = []

        # Reset and build temporal semantic model
        self.temporal_tracker.reset()
        self._build_temporal_model(analysis)

        # Apply temporal leakage detection algorithms
        patterns.extend(self._detect_negative_shifts(analysis))
        patterns.extend(self._detect_centered_rolling_windows(analysis))
        patterns.extend(self._detect_global_temporal_statistics(analysis))
        patterns.extend(self._detect_random_splits_on_temporal_data(analysis))
        patterns.extend(self._detect_forward_looking_features(analysis))

        return patterns

    def _build_temporal_model(self, analysis: ASTAnalysisResult):
        """Build semantic model for temporal data analysis"""
        # Walk AST to identify temporal patterns
        for node in ast.walk(analysis.ast_tree):
            if isinstance(node, ast.Call):
                self._analyze_temporal_call(node)
            elif isinstance(node, ast.Assign):
                self._analyze_temporal_assignment(node)

    def _analyze_temporal_call(self, node: ast.Call):
        """Analyze function calls for temporal patterns"""
        func_name = self.get_function_name(node)

        # Check for temporal operations
        if any(temp_func in func_name for temp_func in self.temporal_functions):
            if "shift" in func_name:
                self._analyze_shift_operation(node)
            elif "rolling" in func_name:
                self._analyze_rolling_operation(node)
            elif any(
                split_func in func_name
                for split_func in self.problematic_split_functions
            ):
                self._analyze_split_operation(node)

    def _analyze_shift_operation(self, node: ast.Call):
        """Analyze shift operations for negative values (future leakage)"""
        # Check arguments for negative values
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                if arg.value < 0:  # Negative shift = future data
                    self.temporal_tracker.track_temporal_operation(
                        "negative_shift", node, {"shift_value": arg.value}
                    )

        # Check keyword arguments
        for keyword in node.keywords:
            if keyword.arg == "periods" and isinstance(keyword.value, ast.Constant):
                if isinstance(keyword.value.value, int) and keyword.value.value < 0:
                    self.temporal_tracker.track_temporal_operation(
                        "negative_shift", node, {"shift_value": keyword.value.value}
                    )

    def _analyze_rolling_operation(self, node: ast.Call):
        """Analyze rolling operations for center=True (uses future data)"""
        for keyword in node.keywords:
            if keyword.arg == "center" and isinstance(keyword.value, ast.Constant):
                if keyword.value.value is True:  # center=True uses future data
                    self.temporal_tracker.track_temporal_operation(
                        "centered_rolling", node, {"center": True}
                    )

    def _analyze_split_operation(self, node: ast.Call):
        """Analyze data split operations on potential temporal data"""
        func_name = self.get_function_name(node)
        if func_name in self.problematic_split_functions:
            # Check if this might be operating on temporal data
            # (This is a simplified heuristic - could be enhanced)
            self.temporal_tracker.track_temporal_split(node, "random_split")

    def _analyze_temporal_assignment(self, node: ast.Assign):
        """Analyze assignments for temporal feature engineering"""
        # Look for assignments that might create temporal features
        if isinstance(node.value, ast.Call):
            func_name = self.get_function_name(node.value)

            # Check for global statistics operations, but exclude safe operations
            if func_name in ["mean", "std", "max", "min", "median"]:
                # Check if this is a safe expanding operation
                if not self._is_safe_temporal_operation(node.value):
                    # This could be global temporal statistics
                    self.temporal_tracker.track_temporal_operation(
                        "global_temporal_stats", node.value
                    )

    def _is_safe_temporal_operation(self, node: ast.Call) -> bool:
        """Check if a temporal operation is safe (uses only past data)"""
        # Check if this is an expanding operation (safe)
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Call):
                inner_func = self.get_function_name(node.func.value)
                if "expanding" in inner_func:
                    return True  # expanding().mean() is safe

        # Check if this is a rolling operation with center=False (safe)
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Call):
                inner_func = self.get_function_name(node.func.value)
                if "rolling" in inner_func:
                    # Check if center=False or not specified (default is False)
                    rolling_call = node.func.value
                    has_center_true = False
                    for keyword in rolling_call.keywords:
                        if keyword.arg == "center" and isinstance(
                            keyword.value, ast.Constant
                        ):
                            if keyword.value.value is True:
                                has_center_true = True
                                break
                    if not has_center_true:
                        return True  # rolling(..., center=False).mean() is safe

        return False

    def _detect_negative_shifts(
        self, analysis: ASTAnalysisResult
    ) -> List[MLAntiPattern]:
        """Detect negative shift operations (future data access)"""
        patterns = []

        temporal_ops = self.temporal_tracker.get_temporal_operations()

        for op in temporal_ops:
            if op["type"] == "negative_shift":
                shift_value = op["parameters"].get("shift_value", 0)
                patterns.append(
                    self._create_negative_shift_pattern(op, shift_value, analysis)
                )

        return patterns

    def _detect_centered_rolling_windows(
        self, analysis: ASTAnalysisResult
    ) -> List[MLAntiPattern]:
        """Detect rolling windows with center=True (uses future data)"""
        patterns = []

        temporal_ops = self.temporal_tracker.get_temporal_operations()

        for op in temporal_ops:
            if op["type"] == "centered_rolling":
                patterns.append(self._create_centered_rolling_pattern(op, analysis))

        return patterns

    def _detect_global_temporal_statistics(
        self, analysis: ASTAnalysisResult
    ) -> List[MLAntiPattern]:
        """Detect global statistics computed on temporal data"""
        patterns = []

        # Context-aware filtering: only flag global stats in time-series contexts
        if self.analysis_context != 'time-series':
            return patterns  # Skip temporal global stats detection for non-time-series contexts

        temporal_ops = self.temporal_tracker.get_temporal_operations()

        for op in temporal_ops:
            if op["type"] == "global_temporal_stats":
                patterns.append(
                    self._create_global_temporal_stats_pattern(op, analysis)
                )

        return patterns

    def _detect_random_splits_on_temporal_data(
        self, analysis: ASTAnalysisResult
    ) -> List[MLAntiPattern]:
        """Detect random data splits on temporal datasets"""
        patterns = []

        split_ops = self.temporal_tracker.split_operations_temporal

        for split_op in split_ops:
            if split_op["type"] == "random_split":
                patterns.append(
                    self._create_random_temporal_split_pattern(split_op, analysis)
                )

        return patterns

    def _detect_forward_looking_features(
        self, analysis: ASTAnalysisResult
    ) -> List[MLAntiPattern]:
        """Detect feature engineering that looks into the future"""
        patterns = []

        # This is a placeholder for more sophisticated forward-looking detection
        # Could be enhanced to detect complex patterns like:
        # - Target encoding using future values
        # - Rolling operations with negative offsets
        # - Features that depend on maximum/minimum of entire series

        return patterns

    # Pattern creation methods
    def _create_negative_shift_pattern(
        self, operation: Dict, shift_value: int, analysis: ASTAnalysisResult
    ) -> MLAntiPattern:
        """Create pattern for negative shift operations"""
        return self.create_pattern(
            pattern_type="temporal_negative_shift",
            severity=PatternSeverity.HIGH,
            node=operation["node"],
            analysis=analysis,
            message=f"Negative shift operation (shift({shift_value})) accesses future data",
            explanation=(
                f"The shift operation with value {shift_value} creates a feature using future data. "
                "This causes look-ahead bias where the model gains access to information that "
                "wouldn't be available at prediction time in a real deployment scenario."
            ),
            suggested_fix=(
                "Use positive shift values to create lag features from past data only. "
                f"Change shift({shift_value}) to shift({abs(shift_value)}) to use past values instead."
            ),
            confidence=0.95,
            fix_snippet=f"""# Correct approach - use only past data:
data['lag_feature'] = data['value'].shift({abs(shift_value)})  # Past values only

# Avoid future leakage:
# data['future_feature'] = data['value'].shift({shift_value})  # This uses future data!""",
            references=[
                "https://machinelearningmastery.com/time-series-forecasting-supervised-learning/",
                "https://otexts.com/fpp3/",
                "https://towardsdatascience.com/time-series-nested-cross-validation",
            ],
        )

    def _create_centered_rolling_pattern(
        self, operation: Dict, analysis: ASTAnalysisResult
    ) -> MLAntiPattern:
        """Create pattern for centered rolling windows"""
        return self.create_pattern(
            pattern_type="temporal_centered_rolling",
            severity=PatternSeverity.HIGH,
            node=operation["node"],
            analysis=analysis,
            message="Rolling window with center=True uses future data",
            explanation=(
                "Rolling operations with center=True use both past and future values to compute "
                "the statistic for each point. This creates look-ahead bias as the model gains "
                "access to future information that wouldn't be available during real-time prediction."
            ),
            suggested_fix=(
                "Set center=False in rolling operations to use only past data. "
                "This ensures temporal causality is respected."
            ),
            confidence=0.90,
            fix_snippet="""# Correct approach - only past data:
data['rolling_mean'] = data['value'].rolling(window=7, center=False).mean()

# Avoid future leakage:
# data['rolling_mean'] = data['value'].rolling(window=7, center=True).mean()  # Uses future!""",
            references=[
                "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html",
                "https://machinelearningmastery.com/time-series-forecasting-supervised-learning/",
            ],
        )

    def _create_global_temporal_stats_pattern(
        self, operation: Dict, analysis: ASTAnalysisResult
    ) -> MLAntiPattern:
        """Create pattern for global temporal statistics"""
        return self.create_pattern(
            pattern_type="temporal_global_statistics",
            severity=PatternSeverity.MEDIUM,
            node=operation["node"],
            analysis=analysis,
            message="Global statistics computed on entire temporal dataset",
            explanation=(
                "Computing statistics (mean, std, max, min) on the entire temporal dataset "
                "includes future data points in the calculation. When these statistics are used "
                "for normalization or feature engineering, they create subtle look-ahead bias."
            ),
            suggested_fix=(
                "Compute statistics using only past data up to each point in time, "
                "or use expanding window operations that respect temporal boundaries."
            ),
            confidence=0.75,
            fix_snippet="""# Correct approach - expanding statistics:
data['expanding_mean'] = data['value'].expanding().mean()  # Only past data
data['normalized'] = (data['value'] - data['expanding_mean']) / data['value'].expanding().std()

# Avoid global statistics:
# global_mean = data['value'].mean()  # Includes future data
# data['normalized'] = (data['value'] - global_mean) / global_std""",
            references=[
                "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.expanding.html"
            ],
        )

    def _create_random_temporal_split_pattern(
        self, split_op: Dict, analysis: ASTAnalysisResult
    ) -> MLAntiPattern:
        """Create pattern for random splits on temporal data"""
        return self.create_pattern(
            pattern_type="temporal_random_split",
            severity=PatternSeverity.HIGH,
            node=split_op["node"],
            analysis=analysis,
            message="Random data split used on temporal data",
            explanation=(
                "Using random splits (train_test_split) on temporal data destroys the chronological "
                "order and can result in training on future data to predict past events. "
                "This violates the fundamental assumption of temporal causality."
            ),
            suggested_fix=(
                "Use temporal splits that respect chronological order. Split data based on time "
                "boundaries, with training data from the past and test data from the future."
            ),
            confidence=0.85,
            fix_snippet="""# Correct approach - temporal split:
split_point = int(0.8 * len(data))
train_data = data[:split_point]  # Past data
test_data = data[split_point:]   # Future data

# For cross-validation, use TimeSeriesSplit:
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Avoid random splits:
# X_train, X_test, y_train, y_test = train_test_split(X, y)  # Destroys temporal order!""",
            references=[
                "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html",
                "https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/",
            ],
        )
