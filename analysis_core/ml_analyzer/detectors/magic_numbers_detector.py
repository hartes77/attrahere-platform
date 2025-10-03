"""
Magic Number Detector - High-Value ML Pattern Detection

Detects hardcoded magic numbers in ML code that should be extracted to configuration.
Magic numbers make ML experiments non-reproducible and hyperparameter tuning difficult.

Key Patterns Detected:
1. Hardcoded hyperparameters (learning rates, batch sizes, dimensions)
2. Magic dimensions in neural network operations
3. Hardcoded thresholds and ratios
4. Context-aware detection (distinguishes standard values from magic numbers)

Migrated from proven implementation in ml_patterns.py to BaseMLDetector architecture.
"""

import ast
from typing import Dict, List, Any, Set

from .base_detector import BaseMLDetector, MLAntiPattern, PatternSeverity
from ..ast_engine import ASTAnalysisResult


class MagicNumberExtractor(BaseMLDetector):
    """
    Detects magic numbers and suggests extracting them to configuration files.

    Magic numbers make ML experiments non-reproducible and hard to tune.
    This detector identifies hardcoded values that should be parameterized.
    """

    def __init__(self):
        super().__init__()

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
                                patterns.append(self._create_magic_hyperparameter_pattern(
                                    arg, arg.value, analysis, context="constructor"
                                ))

                    # Check keyword arguments
                    for keyword in node.keywords:
                        if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, (int, float)):
                            if self._is_magic_number(keyword.value.value, keyword.arg):  # Use context-aware detection
                                patterns.append(self._create_magic_hyperparameter_pattern(
                                    keyword.value, keyword.value.value, analysis, 
                                    context=keyword.arg, parameter_name=keyword.arg
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
                            patterns.append(self._create_magic_dimension_pattern(
                                arg, arg.value, analysis, node.func.attr
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
            return node.func.attr in ml_constructors
        
        return False

    def _is_magic_number(self, value: float, parameter_name: str = None) -> bool:
        """Check if a number is considered a magic number in the given context"""
        # Always acceptable values
        if value in self.standard_ml_values:
            return False
        
        # Context-specific checks
        if parameter_name and parameter_name in self.parameter_contexts:
            if value in self.parameter_contexts[parameter_name]:
                return False
        
        # Consider as magic number if it's not in any whitelist
        return True

    def _is_magic_number_in_constructor(self, value: float) -> bool:
        """Check if a number is magic when used as positional argument in ML constructor"""
        # More lenient for constructor positional arguments
        # Include standard NN dimensions and common ML values
        common_values = self.standard_ml_values | self.standard_nn_dimensions | {0.5, 2.0, 10.0}
        return value not in common_values

    def _is_magic_dimension(self, value: int, function_name: str) -> bool:
        """Check if a dimension value is considered magic in the given context"""
        if value in self.standard_nn_dimensions:
            return False
        
        # Function-specific checks
        if function_name in ['reshape', 'view']:
            # For reshape/view, -1 is standard (infer dimension)
            if value == -1:
                return False
        
        return True

    def _create_magic_hyperparameter_pattern(
        self, node: ast.AST, value: float, analysis: ASTAnalysisResult, 
        context: str = None, parameter_name: str = None
    ) -> MLAntiPattern:
        """Create pattern for magic hyperparameter"""
        if parameter_name:
            message = f"Magic number {value} in {parameter_name} parameter"
        else:
            message = f"Magic number {value} in ML constructor"
            
        return self.create_pattern(
            pattern_type="magic_hyperparameter",
            severity=PatternSeverity.MEDIUM,
            node=node,
            analysis=analysis,
            message=message,
            explanation=(
                "Hardcoded hyperparameters make experiments difficult to reproduce "
                "and hyperparameter tuning cumbersome. Consider extracting to configuration."
            ),
            suggested_fix="Extract to configuration file or use named constants.",
            confidence=0.80,
            fix_snippet=self._generate_config_fix(value, parameter_name),
            references=[
                "https://docs.python.org/3/tutorial/modules.html#more-on-modules",
                "https://realpython.com/python-constants/",
                "https://12factor.net/config"
            ]
        )

    def _create_magic_dimension_pattern(
        self, node: ast.AST, value: int, analysis: ASTAnalysisResult, function_name: str
    ) -> MLAntiPattern:
        """Create pattern for magic dimension"""
        return self.create_pattern(
            pattern_type="magic_dimension",
            severity=PatternSeverity.LOW,
            node=node,
            analysis=analysis,
            message=f"Magic dimension {value} in {function_name}",
            explanation="Hardcoded dimensions make model architecture changes difficult.",
            suggested_fix="Use named constants or configuration for dimensions.",
            confidence=0.70,
            fix_snippet=f"HIDDEN_SIZE = {value}  # Define as constant\n# Then use: {function_name}(..., HIDDEN_SIZE, ...)",
            references=[
                "https://realpython.com/python-constants/",
                "https://pytorch.org/docs/stable/notes/tensor_attributes.html"
            ]
        )

    def _generate_config_fix(self, value: float, parameter_name: str = None) -> str:
        """Generate configuration-based fix suggestion"""
        if parameter_name:
            config_name = parameter_name.upper()
        else:
            config_name = "HYPERPARAMETER"
            
        return f"""# config.py
{config_name} = {value}

# main.py
from config import {config_name}
# Use {config_name} instead of {value}"""