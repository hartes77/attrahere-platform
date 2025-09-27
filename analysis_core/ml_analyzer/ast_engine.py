"""
AST Analysis Engine - Core component for Python code analysis

This module provides deep AST analysis capabilities with ML-specific semantic understanding.
It goes beyond traditional linting to understand ML concepts and data flow patterns.
"""

import ast
import inspect
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Set, Union
from pathlib import Path

import libcst as cst
from libcst import matchers as m
from libcst.metadata import PositionProvider, ScopeProvider


@dataclass
class ASTAnalysisResult:
    """Results from AST analysis of a code file"""
    file_path: str
    ast_tree: ast.AST
    cst_tree: cst.Module
    imports: Dict[str, str]  # alias -> module_name
    functions: Dict[str, ast.FunctionDef]
    classes: Dict[str, ast.ClassDef] 
    variables: Dict[str, ast.Assign]
    ml_constructs: Dict[str, Any]  # ML-specific constructs found
    data_flow: Dict[str, List[str]]  # Variable dependencies
    complexity_metrics: Dict[str, int]


class MLSemanticAnalyzer:
    """
    Advanced AST analyzer with ML domain knowledge.
    
    This analyzer understands:
    - ML library patterns (sklearn, torch, tensorflow)
    - Data flow between ML operations
    - Training/validation/test data separation
    - Model lifecycle operations
    """
    
    def __init__(self):
        self.ml_libraries = {
            'sklearn': ['preprocessing', 'model_selection', 'metrics', 'ensemble'],
            'torch': ['nn', 'optim', 'utils', 'cuda'],
            'tensorflow': ['keras', 'data', 'train'],
            'pandas': ['DataFrame', 'Series'],
            'numpy': ['array', 'ndarray'],
        }
        
        # ML-specific AST patterns to recognize
        self.ml_patterns = {
            'data_split': ['train_test_split', 'KFold', 'StratifiedKFold'],
            'preprocessing': ['StandardScaler', 'MinMaxScaler', 'LabelEncoder'],
            'models': ['RandomForestClassifier', 'SVC', 'LinearRegression'],
            'training': ['fit', 'fit_transform', 'predict'],
            'gpu_ops': ['cuda', 'device', 'to_device'],
        }
    
    def analyze_file(self, file_path: Union[str, Path]) -> ASTAnalysisResult:
        """
        Perform comprehensive AST analysis on a Python file.
        
        Args:
            file_path: Path to Python file to analyze
            
        Returns:
            Complete analysis results with ML semantic understanding
        """
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Parse with both AST and LibCST for different analysis needs
        ast_tree = ast.parse(source_code)
        cst_tree = cst.parse_module(source_code)
        
        # Extract basic code structures
        imports = self._extract_imports(ast_tree)
        functions = self._extract_functions(ast_tree)
        classes = self._extract_classes(ast_tree)
        variables = self._extract_variables(ast_tree)
        
        # ML-specific analysis
        ml_constructs = self._identify_ml_constructs(ast_tree, imports)
        data_flow = self._analyze_data_flow(ast_tree)
        complexity_metrics = self._calculate_complexity(ast_tree)
        
        return ASTAnalysisResult(
            file_path=str(file_path),
            ast_tree=ast_tree,
            cst_tree=cst_tree,
            imports=imports,
            functions=functions,
            classes=classes,
            variables=variables,
            ml_constructs=ml_constructs,
            data_flow=data_flow,
            complexity_metrics=complexity_metrics
        )
    
    def _extract_imports(self, tree: ast.AST) -> Dict[str, str]:
        """Extract import statements with alias mapping"""
        imports = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ''
                for alias in node.names:
                    full_name = f"{module_name}.{alias.name}" if module_name else alias.name
                    imports[alias.asname or alias.name] = full_name
        
        return imports
    
    def _extract_functions(self, tree: ast.AST) -> Dict[str, ast.FunctionDef]:
        """Extract function definitions"""
        functions = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = node
        
        return functions
    
    def _extract_classes(self, tree: ast.AST) -> Dict[str, ast.ClassDef]:
        """Extract class definitions"""
        classes = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes[node.name] = node
        
        return classes
    
    def _extract_variables(self, tree: ast.AST) -> Dict[str, ast.Assign]:
        """Extract variable assignments"""
        variables = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables[target.id] = node
        
        return variables
    
    def _identify_ml_constructs(self, tree: ast.AST, imports: Dict[str, str]) -> Dict[str, Any]:
        """
        Identify ML-specific constructs and patterns in the code.
        
        This is the key differentiator - understanding ML semantics beyond syntax.
        """
        ml_constructs = {
            'data_splits': [],
            'preprocessors': [],
            'models': [],
            'training_loops': [],
            'gpu_operations': [],
            'data_loaders': [],
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_info = self._analyze_function_call(node, imports)
                
                # Categorize ML operations
                if call_info.get('is_data_split'):
                    ml_constructs['data_splits'].append({
                        'line': node.lineno,
                        'function': call_info['function_name'],
                        'args': call_info['args']
                    })
                
                elif call_info.get('is_preprocessor'):
                    ml_constructs['preprocessors'].append({
                        'line': node.lineno,
                        'function': call_info['function_name'],
                        'method': call_info.get('method'),
                        'target_var': call_info.get('target_var')
                    })
                
                elif call_info.get('is_model'):
                    ml_constructs['models'].append({
                        'line': node.lineno,
                        'model_type': call_info['function_name'],
                        'params': call_info['args']
                    })
        
        return ml_constructs
    
    def _analyze_function_call(self, node: ast.Call, imports: Dict[str, str]) -> Dict[str, Any]:
        """Analyze a function call to determine if it's ML-related"""
        call_info = {'args': [], 'kwargs': {}}
        
        # Extract function name
        if isinstance(node.func, ast.Name):
            call_info['function_name'] = node.func.id
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                call_info['function_name'] = f"{node.func.value.id}.{node.func.attr}"
            else:
                call_info['function_name'] = node.func.attr
        
        # Check if it's an ML operation
        func_name = call_info.get('function_name', '')
        
        # Data splitting operations
        if any(pattern in func_name for pattern in self.ml_patterns['data_split']):
            call_info['is_data_split'] = True
        
        # Preprocessing operations  
        elif any(pattern in func_name for pattern in self.ml_patterns['preprocessing']):
            call_info['is_preprocessor'] = True
            
        # Model operations
        elif any(pattern in func_name for pattern in self.ml_patterns['models']):
            call_info['is_model'] = True
        
        return call_info
    
    def _analyze_data_flow(self, tree: ast.AST) -> Dict[str, List[str]]:
        """
        Analyze data flow between variables - critical for data leakage detection.
        
        This tracks how data flows through transformations to detect when
        test data might be contaminating training operations.
        """
        data_flow = {}
        
        # Simple data flow analysis - tracks variable assignments
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = []
                sources = []
                
                # Extract target variables
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        targets.append(target.id)
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                targets.append(elt.id)
                
                # Extract source variables
                for source_node in ast.walk(node.value):
                    if isinstance(source_node, ast.Name):
                        sources.append(source_node.id)
                
                # Record data flow relationships
                for target in targets:
                    data_flow[target] = sources
        
        return data_flow
    
    def _calculate_complexity(self, tree: ast.AST) -> Dict[str, int]:
        """Calculate code complexity metrics"""
        complexity = {
            'cyclomatic_complexity': 0,
            'cognitive_complexity': 0,
            'function_count': 0,
            'class_count': 0,
            'line_count': 0
        }
        
        # Count different node types
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                complexity['cyclomatic_complexity'] += 1
            elif isinstance(node, ast.FunctionDef):
                complexity['function_count'] += 1
            elif isinstance(node, ast.ClassDef):
                complexity['class_count'] += 1
        
        return complexity
    
    def extract_pure_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Identify functions that can be extracted as pure functions.
        
        Pure functions are:
        - No side effects (no I/O, no global state modification)
        - Deterministic (same input -> same output)
        - Easily testable
        """
        pure_candidates = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                purity_score = self._assess_function_purity(node)
                
                if purity_score > 0.7:  # High purity threshold
                    pure_candidates.append({
                        'name': node.name,
                        'node': node,
                        'purity_score': purity_score,
                        'line_start': node.lineno,
                        'line_end': node.end_lineno or node.lineno
                    })
        
        return pure_candidates
    
    def _assess_function_purity(self, func_node: ast.FunctionDef) -> float:
        """
        Assess how 'pure' a function is (0.0 = impure, 1.0 = completely pure)
        """
        purity_score = 1.0
        
        for node in ast.walk(func_node):
            # Penalize side effects
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                purity_score -= 0.3
            elif isinstance(node, ast.Call):
                # Check for I/O operations
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['print', 'open', 'write']:
                        purity_score -= 0.4
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['append', 'extend', 'update']:
                        purity_score -= 0.2  # List/dict mutations
        
        return max(0.0, purity_score)


class DataFlowAnalyzer:
    """
    Specialized analyzer for tracking data flow in ML pipelines.
    
    This is crucial for detecting data leakage - when information from
    the test set inadvertently influences training.
    """
    
    def __init__(self):
        self.data_containers = {'DataFrame', 'Series', 'ndarray', 'Tensor'}
        self.split_operations = {'train_test_split', 'KFold', 'StratifiedKFold'}
        self.fit_operations = {'fit', 'fit_transform'}
    
    def trace_data_lineage(self, tree: ast.AST) -> Dict[str, Dict[str, Any]]:
        """
        Trace the lineage of data variables through the ML pipeline.
        
        Returns:
            Dictionary mapping variable names to their data lineage info
        """
        lineage = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                self._track_assignment(node, lineage)
        
        return lineage
    
    def _track_assignment(self, node: ast.Assign, lineage: Dict[str, Dict[str, Any]]):
        """Track a variable assignment and its data sources"""
        # Extract target variable names
        targets = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                targets.append(target.id)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        targets.append(elt.id)
        
        # Analyze the source of the assignment
        source_info = self._analyze_assignment_source(node.value)
        
        # Record lineage for each target
        for target in targets:
            lineage[target] = {
                'line': node.lineno,
                'source_type': source_info['type'],
                'source_vars': source_info['variables'],
                'operation': source_info['operation'],
                'is_data_split': source_info.get('is_data_split', False),
                'is_fit_operation': source_info.get('is_fit_operation', False)
            }
    
    def _analyze_assignment_source(self, node: ast.AST) -> Dict[str, Any]:
        """Analyze what type of operation is producing this assignment"""
        if isinstance(node, ast.Call):
            return self._analyze_function_call_source(node)
        elif isinstance(node, ast.Name):
            return {
                'type': 'variable_reference',
                'variables': [node.id],
                'operation': 'assignment'
            }
        else:
            return {
                'type': 'literal_or_expression',
                'variables': [],
                'operation': 'expression'
            }
    
    def _analyze_function_call_source(self, node: ast.Call) -> Dict[str, Any]:
        """Analyze a function call to understand its data implications"""
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        
        # Extract variables used in the call
        variables = []
        for arg in node.args:
            if isinstance(arg, ast.Name):
                variables.append(arg.id)
        
        source_info = {
            'type': 'function_call',
            'variables': variables,
            'operation': func_name
        }
        
        # Check for specific ML operations
        if func_name in self.split_operations:
            source_info['is_data_split'] = True
        elif func_name in self.fit_operations:
            source_info['is_fit_operation'] = True
        
        return source_info