"""
Scope-Aware Analysis Engine - V4 Breakthrough

Adds function-scope intelligence to ML pattern detection, enabling surgical precision
by understanding the hierarchical context of operations within function boundaries.

This solves the final false-positive challenge: distinguishing between global scope
leakage (real problems) and intra-function safe operations (false positives).

Key Capabilities:
1. Function Boundary Mapping - Maps def start/end lines during AST traversal
2. Hierarchical Context Analysis - Understands operation scope and context  
3. Intra-Function Sequencing - Tracks operation order within function boundaries
4. Contextual Validation - Eliminates false positives through scope awareness

This represents the V4 evolution of our detection engine:
V1: Pattern Recognition
V2: Data Lineage Tracking  
V3: Domain Awareness (CV vs TS)
V4: Scope Awareness (Function Context)
"""

import ast
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass


@dataclass
class FunctionScope:
    """Represents a function's scope and boundaries"""
    name: str
    start_line: int
    end_line: int
    node: ast.FunctionDef
    operations: List[Dict[str, Any]]  # Operations within this function


@dataclass
class OperationContext:
    """Context information for an operation"""
    line: int
    operation_type: str  # 'preprocessing', 'data_split', 'other'
    details: Dict[str, Any]
    scope: Optional[FunctionScope] = None


class ScopeAwareAnalyzer:
    """
    Scope-aware analysis engine that understands function boundaries
    and provides contextual validation for ML pattern detection.
    """

    def __init__(self):
        self.function_scopes: List[FunctionScope] = []
        self.global_operations: List[OperationContext] = []
        
        # Operation types we care about for scope analysis
        self.preprocessing_methods = {'fit', 'fit_transform', 'transform'}
        self.split_functions = {'train_test_split', 'KFold', 'StratifiedKFold', 'TimeSeriesSplit'}

    def analyze_scope_context(self, ast_tree: ast.AST) -> Dict[str, Any]:
        """
        Perform comprehensive scope analysis of the AST.
        
        Returns:
            Dictionary containing:
            - function_scopes: Mapped function boundaries
            - operation_contexts: All operations with their scope context
            - scope_relationships: Hierarchical relationships
        """
        # Phase 1: Map all function boundaries
        self._map_function_boundaries(ast_tree)
        
        # Phase 2: Analyze all operations with their scope context
        self._analyze_operations_with_scope(ast_tree)
        
        # Phase 3: Build scope relationships
        scope_relationships = self._build_scope_relationships()
        
        return {
            'function_scopes': self.function_scopes,
            'operation_contexts': self.global_operations,
            'scope_relationships': scope_relationships
        }

    def _map_function_boundaries(self, ast_tree: ast.AST):
        """Phase 1: Map all function boundaries in the AST"""
        self.function_scopes = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                # Calculate function end line by finding the last statement
                end_line = self._calculate_function_end_line(node)
                
                scope = FunctionScope(
                    name=node.name,
                    start_line=node.lineno,
                    end_line=end_line,
                    node=node,
                    operations=[]
                )
                
                self.function_scopes.append(scope)

    def _calculate_function_end_line(self, func_node: ast.FunctionDef) -> int:
        """Calculate the actual end line of a function"""
        max_line = func_node.lineno
        
        # Walk through all nodes in the function body
        for stmt in ast.walk(func_node):
            if hasattr(stmt, 'lineno') and stmt.lineno:
                max_line = max(max_line, stmt.lineno)
                # Also check for end_lineno if available (Python 3.8+)
                if hasattr(stmt, 'end_lineno') and stmt.end_lineno:
                    max_line = max(max_line, stmt.end_lineno)
        
        return max_line

    def _analyze_operations_with_scope(self, ast_tree: ast.AST):
        """Phase 2: Analyze all operations and assign scope context"""
        self.global_operations = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                operation_context = self._analyze_assignment_operation(node)
                if operation_context:
                    # Find which function scope this operation belongs to
                    operation_context.scope = self._find_containing_scope(operation_context.line)
                    self.global_operations.append(operation_context)

    def _analyze_assignment_operation(self, node: ast.Assign) -> Optional[OperationContext]:
        """Analyze an assignment to determine if it's a relevant operation"""
        if not isinstance(node.value, ast.Call):
            return None

        line = node.lineno
        
        # Check for preprocessing operations (method calls)
        if isinstance(node.value.func, ast.Attribute):
            method_name = node.value.func.attr
            
            if method_name in self.preprocessing_methods:
                obj_name = self._get_object_name(node.value.func.value)
                return OperationContext(
                    line=line,
                    operation_type='preprocessing',
                    details={
                        'method': method_name,
                        'object': obj_name,
                        'node': node
                    }
                )

        # Check for data splitting operations (function calls)
        elif isinstance(node.value.func, ast.Name):
            func_name = node.value.func.id
            
            if func_name in self.split_functions:
                return OperationContext(
                    line=line,
                    operation_type='data_split',
                    details={
                        'function': func_name,
                        'node': node
                    }
                )

        return None

    def _get_object_name(self, node: ast.AST) -> str:
        """Extract object name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_object_name(node.value)}.{node.attr}"
        else:
            return "unknown"

    def _find_containing_scope(self, line: int) -> Optional[FunctionScope]:
        """Find which function scope contains the given line number"""
        for scope in self.function_scopes:
            if scope.start_line <= line <= scope.end_line:
                return scope
        return None  # Global scope

    def _build_scope_relationships(self) -> Dict[str, Any]:
        """Phase 3: Build relationships between scopes and operations"""
        relationships = {
            'global_operations': [],
            'function_operations': {},
            'scope_summary': {}
        }
        
        # Categorize operations by scope
        for operation in self.global_operations:
            if operation.scope is None:
                # Global scope operation
                relationships['global_operations'].append(operation)
            else:
                # Function scope operation
                func_name = operation.scope.name
                if func_name not in relationships['function_operations']:
                    relationships['function_operations'][func_name] = []
                relationships['function_operations'][func_name].append(operation)

        # Build scope summary
        for func_name, operations in relationships['function_operations'].items():
            relationships['scope_summary'][func_name] = {
                'total_operations': len(operations),
                'preprocessing_ops': len([op for op in operations if op.operation_type == 'preprocessing']),
                'split_ops': len([op for op in operations if op.operation_type == 'data_split']),
                'operation_sequence': sorted(operations, key=lambda x: x.line)
            }

        return relationships

    def is_safe_preprocessing_in_function(
        self, preprocessing_line: int, split_line: int, scope_context: Dict[str, Any]
    ) -> bool:
        """
        Determine if preprocessing operation is safe within function context.
        
        Args:
            preprocessing_line: Line number of preprocessing operation
            split_line: Line number of split operation  
            scope_context: Scope analysis context
            
        Returns:
            True if preprocessing is safe (false positive), False if it's a real problem
        """
        # Find the function scope for the preprocessing operation
        preprocessing_scope = self._find_containing_scope(preprocessing_line)
        split_scope = self._find_containing_scope(split_line)
        
        # If operations are in different scopes, it's likely a real problem
        if preprocessing_scope != split_scope:
            return False
            
        # If both are in global scope, it's definitely a problem
        if preprocessing_scope is None and split_scope is None:
            return False
            
        # If both are in the same function, check the operation sequence
        if preprocessing_scope and split_scope and preprocessing_scope == split_scope:
            func_name = preprocessing_scope.name
            function_ops = scope_context['scope_relationships']['function_operations'].get(func_name, [])
            
            # Sort operations by line number
            sorted_ops = sorted(function_ops, key=lambda x: x.line)
            
            # Find the preprocessing and split operations in the sequence
            preprocessing_index = None
            split_index = None
            
            for i, op in enumerate(sorted_ops):
                if op.line == preprocessing_line and op.operation_type == 'preprocessing':
                    preprocessing_index = i
                elif op.line == split_line and op.operation_type == 'data_split':
                    split_index = i
            
            # If split comes before preprocessing in the same function, it's safe
            if split_index is not None and preprocessing_index is not None:
                return split_index < preprocessing_index
        
        # Default: assume it's a problem
        return False

    def validate_pattern_in_scope(
        self, pattern_line: int, pattern_type: str, scope_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate a detected pattern considering its scope context.
        
        Returns:
            Dictionary with validation results:
            - is_valid: Whether the pattern is a real issue
            - scope_info: Information about the scope context
            - recommendation: What action to take
        """
        scope = self._find_containing_scope(pattern_line)
        
        validation_result = {
            'is_valid': True,  # Default: assume it's a real problem
            'scope_info': {
                'scope_type': 'global' if scope is None else 'function',
                'scope_name': None if scope is None else scope.name,
                'line': pattern_line
            },
            'recommendation': 'flag_as_issue'
        }
        
        # If it's in global scope, it's definitely a problem
        if scope is None:
            validation_result['recommendation'] = 'flag_as_critical_issue'
            return validation_result
        
        # If it's in function scope, do contextual analysis
        if scope and pattern_type == 'preprocessing_before_split':
            func_operations = scope_context['scope_relationships']['function_operations'].get(scope.name, [])
            
            # Check if there's a split operation in the same function that comes before
            preprocessing_ops = [op for op in func_operations if op.operation_type == 'preprocessing' and op.line == pattern_line]
            split_ops = [op for op in func_operations if op.operation_type == 'data_split']
            
            if preprocessing_ops and split_ops:
                # Find if any split comes before the preprocessing
                preprocessing_line = preprocessing_ops[0].line
                earlier_splits = [op for op in split_ops if op.line < preprocessing_line]
                
                if earlier_splits:
                    # There's a split before the preprocessing in the same function
                    validation_result['is_valid'] = False
                    validation_result['recommendation'] = 'ignore_false_positive'
                    validation_result['scope_info']['context'] = 'safe_function_context'
        
        return validation_result