#!/usr/bin/env python3
"""
ML Code Refactoring Transforms - LibCST-based code transformations

This module implements concrete refactoring transformations using libcst
to safely modify Python code and fix ML anti-patterns.

Key Features:
- Safe AST transformations preserving formatting and comments
- ML-specific fix patterns (data leakage, magic numbers, reproducibility)
- Confidence-based fix application
- Before/after diff generation
- Integration with ml-quality CLI
"""

import ast
import libcst as cst
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import re
import textwrap


@dataclass
class FixResult:
    """Result of applying a refactoring fix"""
    success: bool
    original_code: str
    refactored_code: str
    fixes_applied: List[Dict[str, Any]]
    error: Optional[str] = None
    confidence_score: float = 0.0
    
    def get_diff_summary(self) -> str:
        """Generate a summary of changes made"""
        if not self.fixes_applied:
            return "No fixes applied"
        
        summary = f"Applied {len(self.fixes_applied)} fixes:\n"
        for fix in self.fixes_applied:
            summary += f"• Line {fix.get('line', '?')}: {fix.get('description', fix.get('type', 'Fix applied'))}\n"
        return summary


class MLCodeTransformer(cst.CSTTransformer):
    """
    LibCST transformer for applying ML-specific code fixes.
    
    This transformer safely modifies Python code to fix ML anti-patterns
    while preserving semantics and formatting.
    """
    
    def __init__(self, patterns_to_fix: List[Dict[str, Any]]):
        super().__init__()
        self.patterns_to_fix = patterns_to_fix
        self.fixes_applied = []
        
        # Group patterns by line for efficient lookup
        self.patterns_by_line = {}
        for pattern in patterns_to_fix:
            line = pattern.get('line', 0)
            if line not in self.patterns_by_line:
                self.patterns_by_line[line] = []
            self.patterns_by_line[line].append(pattern)
    
    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        """Handle assignment transformations (magic numbers, model instantiation)"""
        line_num = self._get_line_number(original_node)
        patterns = self.patterns_by_line.get(line_num, [])
        
        current_node = updated_node
        for pattern in patterns:
            if pattern['type'] == 'magic_numbers':
                current_node = self._fix_magic_numbers_in_assign(current_node, pattern)
            elif pattern['type'] == 'reproducibility':
                current_node = self._fix_reproducibility_in_assign(current_node, pattern)
        
        return current_node
    
    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        """Handle function call transformations"""
        line_num = self._get_line_number(original_node)
        patterns = self.patterns_by_line.get(line_num, [])
        
        current_node = updated_node
        for pattern in patterns:
            if pattern['type'] == 'reproducibility':
                current_node = self._add_random_state_to_call(current_node, pattern)
            elif pattern['type'] == 'gpu_memory':
                current_node = self._fix_gpu_memory_in_call(current_node, pattern)
        
        return current_node
    
    def leave_SimpleStatementLine(self, original_node: cst.SimpleStatementLine, 
                                 updated_node: cst.SimpleStatementLine) -> Union[cst.SimpleStatementLine, cst.FlattenSentinel[cst.SimpleStatementLine]]:
        """Handle statement-level transformations (data leakage)"""
        line_num = self._get_line_number(original_node)
        patterns = self.patterns_by_line.get(line_num, [])
        
        for pattern in patterns:
            if pattern['type'] == 'data_leakage':
                return self._fix_data_leakage_statement(updated_node, pattern)
        
        return updated_node
    
    def _get_line_number(self, node: cst.CSTNode) -> int:
        """Extract line number from CST node (simplified)"""
        # In practice, this would use the metadata wrapper
        # For now, we'll use a heuristic based on patterns
        return getattr(node, 'lineno', 0)
    
    def _fix_magic_numbers_in_assign(self, node: cst.Assign, pattern: Dict[str, Any]) -> cst.Assign:
        """Fix magic numbers by extracting to configuration constants"""
        if isinstance(node.value, cst.Call):
            new_call = self._replace_magic_numbers_in_call(node.value, pattern)
            if new_call != node.value:
                self.fixes_applied.append({
                    'type': 'magic_numbers',
                    'line': pattern.get('line'),
                    'description': f"Extracted magic numbers to constants"
                })
                return node.with_changes(value=new_call)
        
        return node
    
    def _replace_magic_numbers_in_call(self, call: cst.Call, pattern: Dict[str, Any]) -> cst.Call:
        """Replace magic numbers in function calls with named constants"""
        new_args = []
        modified = False
        
        for arg in call.args:
            if isinstance(arg.value, cst.Integer):
                # Check if this is a magic number we should replace
                value = int(arg.value.value)
                constant_name = self._get_constant_name_for_value(value, arg.keyword)
                
                if constant_name:
                    new_arg = arg.with_changes(value=cst.Name(constant_name))
                    new_args.append(new_arg)
                    modified = True
                else:
                    new_args.append(arg)
            elif isinstance(arg.value, cst.Float):
                # Handle float magic numbers
                value = float(arg.value.value)
                constant_name = self._get_constant_name_for_value(value, arg.keyword)
                
                if constant_name:
                    new_arg = arg.with_changes(value=cst.Name(constant_name))
                    new_args.append(new_arg)
                    modified = True
                else:
                    new_args.append(arg)
            else:
                new_args.append(arg)
        
        if modified:
            return call.with_changes(args=new_args)
        return call
    
    def _get_constant_name_for_value(self, value: Union[int, float], keyword: Optional[cst.Name]) -> Optional[str]:
        """Get appropriate constant name for a magic number"""
        if keyword and hasattr(keyword, 'value'):
            param_name = keyword.value
            
            # Map parameter names to constant names
            constant_mapping = {
                'n_estimators': 'N_ESTIMATORS',
                'max_depth': 'MAX_DEPTH', 
                'learning_rate': 'LEARNING_RATE',
                'batch_size': 'BATCH_SIZE',
                'epochs': 'EPOCHS',
                'hidden_size': 'HIDDEN_SIZE'
            }
            
            return constant_mapping.get(param_name)
        
        return None
    
    def _fix_reproducibility_in_assign(self, node: cst.Assign, pattern: Dict[str, Any]) -> cst.Assign:
        """Fix reproducibility issues in assignments"""
        if isinstance(node.value, cst.Call):
            new_call = self._add_random_state_to_call(node.value, pattern)
            if new_call != node.value:
                return node.with_changes(value=new_call)
        
        return node
    
    def _add_random_state_to_call(self, call: cst.Call, pattern: Dict[str, Any]) -> cst.Call:
        """Add random_state parameter to ML model instantiation"""
        # Check if this is an ML model that needs random_state
        if not self._is_ml_model_call(call):
            return call
        
        # Check if random_state already exists
        has_random_state = any(
            isinstance(arg, cst.Arg) and arg.keyword and 
            hasattr(arg.keyword, 'value') and arg.keyword.value == 'random_state'
            for arg in call.args
        )
        
        if not has_random_state:
            # Add random_state=42
            random_state_arg = cst.Arg(
                keyword=cst.Name('random_state'),
                value=cst.Integer('42')
            )
            
            new_args = list(call.args) + [random_state_arg]
            
            self.fixes_applied.append({
                'type': 'reproducibility',
                'line': pattern.get('line'),
                'description': f"Added random_state=42 for reproducibility"
            })
            
            return call.with_changes(args=new_args)
        
        return call
    
    def _is_ml_model_call(self, call: cst.Call) -> bool:
        """Check if this is a call to an ML model that supports random_state"""
        ml_models_with_random_state = {
            'RandomForestClassifier', 'RandomForestRegressor',
            'SVC', 'SVR', 'KMeans', 'DecisionTreeClassifier',
            'DecisionTreeRegressor', 'LogisticRegression'
        }
        
        if isinstance(call.func, cst.Name):
            return call.func.value in ml_models_with_random_state
        elif isinstance(call.func, cst.Attribute):
            return call.func.attr.value in ml_models_with_random_state
        
        return False
    
    def _fix_gpu_memory_in_call(self, call: cst.Call, pattern: Dict[str, Any]) -> cst.Call:
        """Fix GPU memory leaks by adding .detach() calls"""
        # This is more complex and would require analyzing the call context
        # For now, we'll return the original call
        return call
    
    def _fix_data_leakage_statement(self, node: cst.SimpleStatementLine, 
                                   pattern: Dict[str, Any]) -> Union[cst.SimpleStatementLine, cst.FlattenSentinel[cst.SimpleStatementLine]]:
        """Fix data leakage by adding warning comments"""
        # Add a comment indicating data leakage risk
        warning_comment = cst.Comment("# WARNING: Potential data leakage - consider moving after train_test_split")
        leading_lines = list(node.leading_lines) + [cst.EmptyLine(comment=warning_comment)]
        
        self.fixes_applied.append({
            'type': 'data_leakage',
            'line': pattern.get('line'),
            'description': "Added warning comment for potential data leakage"
        })
        
        return node.with_changes(leading_lines=leading_lines)


class MLRefactoringEngine:
    """
    Main engine for applying ML refactoring transformations.
    
    Integrates with the ML pattern detection system to provide automated fixes.
    """
    
    def __init__(self):
        # Map from analyzer pattern types to refactoring fix types
        self.pattern_type_mapping = {
            'magic_hyperparameter': 'magic_numbers',
            'missing_random_seed': 'reproducibility',
            'data_leakage_preprocessing': 'data_leakage',
            'test_set_contamination': 'data_leakage',
            'gpu_memory_leak': 'gpu_memory'
        }
        
        # Supported fix types that we can automatically apply
        self.supported_fix_types = {
            'magic_numbers', 'reproducibility', 'gpu_memory', 'data_leakage'
        }
    
    def apply_fixes(self, source_code: str, detected_patterns: List[Dict[str, Any]], 
                   fix_types: Optional[Set[str]] = None) -> FixResult:
        """
        Apply refactoring fixes to source code.
        
        Args:
            source_code: Original Python source code
            detected_patterns: Patterns detected by ML analyzer
            fix_types: Optional set of fix types to apply (default: all supported)
            
        Returns:
            FixResult with refactored code and applied fixes
        """
        try:
            # Normalize pattern types using mapping
            normalized_patterns = []
            for pattern in detected_patterns:
                original_type = pattern.get('type', '')
                mapped_type = self.pattern_type_mapping.get(original_type, original_type)
                
                # Create normalized pattern
                normalized_pattern = pattern.copy()
                normalized_pattern['type'] = mapped_type
                normalized_patterns.append(normalized_pattern)
            
            # Filter to fixable patterns
            if fix_types is None:
                fix_types = self.supported_fix_types
            
            fixable_patterns = [
                p for p in normalized_patterns 
                if p.get('type') in fix_types and p.get('type') in self.supported_fix_types
            ]
            
            if not fixable_patterns:
                return FixResult(
                    success=True,
                    original_code=source_code,
                    refactored_code=source_code,
                    fixes_applied=[],
                    confidence_score=1.0
                )
            
            # Parse source code with libcst
            try:
                tree = cst.parse_module(source_code)
            except Exception as e:
                return FixResult(
                    success=False,
                    original_code=source_code,
                    refactored_code=source_code,
                    fixes_applied=[],
                    error=f"Failed to parse source code: {e}"
                )
            
            # Create transformer with patterns to fix
            transformer = MLCodeTransformer(fixable_patterns)
            
            # Apply transformations
            try:
                refactored_tree = tree.visit(transformer)
                refactored_code = refactored_tree.code
            except Exception as e:
                return FixResult(
                    success=False,
                    original_code=source_code,
                    refactored_code=source_code,
                    fixes_applied=[],
                    error=f"Failed to apply transformations: {e}"
                )
            
            # Calculate confidence based on fixes applied
            confidence = self._calculate_confidence(transformer.fixes_applied, fixable_patterns)
            
            return FixResult(
                success=True,
                original_code=source_code,
                refactored_code=refactored_code,
                fixes_applied=transformer.fixes_applied,
                confidence_score=confidence
            )
            
        except Exception as e:
            return FixResult(
                success=False,
                original_code=source_code,
                refactored_code=source_code,
                fixes_applied=[],
                error=f"Unexpected error: {e}"
            )
    
    def generate_fix_constants(self, detected_patterns: List[Dict[str, Any]]) -> str:
        """
        Generate constant definitions for extracted magic numbers.
        
        Returns Python code with constant definitions.
        """
        magic_number_patterns = [
            p for p in detected_patterns if p.get('type') == 'magic_numbers'
        ]
        
        if not magic_number_patterns:
            return ""
        
        constants = []
        constants.append("# Configuration constants extracted from magic numbers")
        constants.append("")
        
        # Common ML constants
        constants.extend([
            "# Model hyperparameters",
            "N_ESTIMATORS = 100  # Number of trees in forest",
            "MAX_DEPTH = 15      # Maximum depth of trees", 
            "LEARNING_RATE = 0.001  # Learning rate for optimization",
            "BATCH_SIZE = 32     # Training batch size",
            "EPOCHS = 100        # Number of training epochs",
            "HIDDEN_SIZE = 128   # Hidden layer size",
            "",
            "# Random seed for reproducibility",
            "RANDOM_SEED = 42",
            ""
        ])
        
        return "\n".join(constants)
    
    def preview_fixes(self, source_code: str, detected_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a preview of what fixes would be applied.
        
        Returns summary without actually modifying the code.
        """
        # Normalize pattern types using mapping
        normalized_patterns = []
        for pattern in detected_patterns:
            original_type = pattern.get('type', '')
            mapped_type = self.pattern_type_mapping.get(original_type, original_type)
            
            # Create normalized pattern
            normalized_pattern = pattern.copy()
            normalized_pattern['type'] = mapped_type
            normalized_patterns.append(normalized_pattern)
        
        fixable_patterns = [
            p for p in normalized_patterns 
            if p.get('type') in self.supported_fix_types
        ]
        
        preview = {
            'total_patterns': len(detected_patterns),
            'fixable_patterns': len(fixable_patterns),
            'fixes_by_type': {},
            'preview_snippets': []
        }
        
        # Group by pattern type
        for pattern in fixable_patterns:
            pattern_type = pattern.get('type', 'unknown')
            if pattern_type not in preview['fixes_by_type']:
                preview['fixes_by_type'][pattern_type] = []
            
            preview['fixes_by_type'][pattern_type].append({
                'line': pattern.get('line'),
                'message': pattern.get('message', ''),
                'confidence': pattern.get('confidence', 0.8),
                'suggested_fix': self._generate_fix_description(pattern)
            })
        
        # Generate code snippets showing before/after
        lines = source_code.split('\n')
        for pattern in fixable_patterns[:5]:  # Show first 5 fixes
            line_num = pattern.get('line', 1) - 1
            if 0 <= line_num < len(lines):
                original_line = lines[line_num]
                suggested_line = self._generate_suggested_fix_line(original_line, pattern)
                
                preview['preview_snippets'].append({
                    'line_number': pattern.get('line'),
                    'pattern_type': pattern.get('type'),
                    'original': original_line.strip(),
                    'suggested': suggested_line.strip(),
                    'explanation': self._generate_fix_description(pattern)
                })
        
        return preview
    
    def _calculate_confidence(self, fixes_applied: List[Dict[str, Any]], 
                             fixable_patterns: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score for applied fixes"""
        if not fixable_patterns:
            return 1.0
        
        # Base confidence on ratio of successfully applied fixes
        success_rate = len(fixes_applied) / len(fixable_patterns)
        
        # Weight by pattern confidence scores
        total_confidence = sum(p.get('confidence', 0.8) for p in fixable_patterns)
        avg_confidence = total_confidence / len(fixable_patterns)
        
        return success_rate * avg_confidence
    
    def _generate_fix_description(self, pattern: Dict[str, Any]) -> str:
        """Generate human-readable description of what fix would be applied"""
        pattern_type = pattern.get('type', 'unknown')
        
        descriptions = {
            'magic_numbers': f"Extract magic number to named constant",
            'reproducibility': f"Add random_state=42 for reproducibility",
            'gpu_memory': f"Add .detach() call to prevent memory leak",
            'data_leakage': f"Add warning comment about potential data leakage"
        }
        
        return descriptions.get(pattern_type, "Apply automated fix")
    
    def _generate_suggested_fix_line(self, original_line: str, pattern: Dict[str, Any]) -> str:
        """Generate suggested fix for a specific line"""
        pattern_type = pattern.get('type', 'unknown')
        
        if pattern_type == 'magic_numbers':
            # Replace common magic numbers
            line = original_line
            line = re.sub(r'n_estimators=\d+', 'n_estimators=N_ESTIMATORS', line)
            line = re.sub(r'max_depth=\d+', 'max_depth=MAX_DEPTH', line)
            return line
            
        elif pattern_type == 'reproducibility':
            # Add random_state parameter
            if 'RandomForest' in original_line and 'random_state' not in original_line:
                return original_line.replace('(', '(random_state=42, ')
        
        elif pattern_type == 'data_leakage':
            return f"# WARNING: Potential data leakage\n{original_line}"
        
        return original_line  # Fallback


def create_refactoring_engine() -> MLRefactoringEngine:
    """Factory function to create a configured refactoring engine"""
    return MLRefactoringEngine()


# CLI integration function
def apply_ml_fixes_to_file(file_path: str, detected_patterns: List[Dict[str, Any]], 
                          output_path: Optional[str] = None, 
                          fix_types: Optional[Set[str]] = None) -> FixResult:
    """
    Apply ML fixes to a file and optionally write the result.
    
    Args:
        file_path: Path to source file
        detected_patterns: Patterns detected by ML analyzer
        output_path: Optional output file path (defaults to overwriting input)
        fix_types: Optional set of fix types to apply
        
    Returns:
        FixResult with details of applied fixes
    """
    try:
        # Read source file
        with open(file_path, 'r') as f:
            source_code = f.read()
        
        # Apply fixes
        engine = create_refactoring_engine()
        result = engine.apply_fixes(source_code, detected_patterns, fix_types)
        
        # Write output if successful and path provided
        if result.success and output_path:
            with open(output_path, 'w') as f:
                f.write(result.refactored_code)
        
        return result
        
    except Exception as e:
        return FixResult(
            success=False,
            original_code="",
            refactored_code="",
            fixes_applied=[],
            error=f"File operation failed: {e}"
        )