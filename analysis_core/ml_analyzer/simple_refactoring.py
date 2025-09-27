#!/usr/bin/env python3
"""
Simple ML refactoring engine using string replacement for common patterns.

This provides a working implementation of automated ML code fixes using
simple but effective string replacement patterns. While not as sophisticated
as AST transformations, it handles the most common ML anti-patterns reliably.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Set
import textwrap


@dataclass
class SimpleFixResult:
    """Result of applying simple refactoring fixes"""
    success: bool
    original_code: str
    refactored_code: str
    fixes_applied: List[Dict[str, Any]]
    error: Optional[str] = None
    confidence_score: float = 0.0
    constants_generated: str = ""
    
    def get_diff_summary(self) -> str:
        """Generate a summary of changes made"""
        if not self.fixes_applied:
            return "No fixes applied"
        
        summary = f"Applied {len(self.fixes_applied)} fixes:\n"
        for fix in self.fixes_applied:
            summary += f"• Line {fix.get('line', '?')}: {fix.get('description', fix.get('type', 'Fix applied'))}\n"
        return summary


class SimpleMLRefactoring:
    """
    Simple but effective ML code refactoring using pattern-based string replacement.
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
        
        # Supported fix types
        self.supported_fix_types = {
            'magic_numbers', 'reproducibility', 'data_leakage'
        }
        
        # Constants for magic number replacement
        self.constants_map = {
            '100': 'N_ESTIMATORS_DEFAULT',
            '50': 'N_ESTIMATORS_SMALL', 
            '200': 'N_ESTIMATORS_LARGE',
            '15': 'MAX_DEPTH_DEFAULT',
            '10': 'MAX_DEPTH_SMALL',
            '20': 'MAX_DEPTH_LARGE',
            '0.001': 'LEARNING_RATE_DEFAULT',
            '0.01': 'LEARNING_RATE_HIGH',
            '32': 'BATCH_SIZE_DEFAULT',
            '64': 'BATCH_SIZE_LARGE'
        }
    
    def apply_fixes(self, source_code: str, detected_patterns: List[Dict[str, Any]], 
                   fix_types: Optional[Set[str]] = None) -> SimpleFixResult:
        """
        Apply simple pattern-based fixes to ML code.
        """
        try:
            # Normalize pattern types
            normalized_patterns = []
            for pattern in detected_patterns:
                original_type = pattern.get('type', '')
                mapped_type = self.pattern_type_mapping.get(original_type, original_type)
                
                normalized_pattern = pattern.copy()
                normalized_pattern['type'] = mapped_type
                normalized_patterns.append(normalized_pattern)
            
            # Filter fixable patterns
            if fix_types is None:
                fix_types = self.supported_fix_types
            
            fixable_patterns = [
                p for p in normalized_patterns 
                if p.get('type') in fix_types and p.get('type') in self.supported_fix_types
            ]
            
            if not fixable_patterns:
                return SimpleFixResult(
                    success=True,
                    original_code=source_code,
                    refactored_code=source_code,
                    fixes_applied=[],
                    confidence_score=1.0
                )
            
            # Apply fixes line by line
            refactored_code = source_code
            fixes_applied = []
            constants_needed = set()
            
            # Sort patterns by line number (descending) to avoid line number shifts
            fixable_patterns.sort(key=lambda p: p.get('line', 0), reverse=True)
            
            for pattern in fixable_patterns:
                fix_result = self._apply_single_fix(refactored_code, pattern)
                if fix_result:
                    refactored_code = fix_result['refactored_code']
                    fixes_applied.append(fix_result['fix_info'])
                    if fix_result.get('constants'):
                        constants_needed.update(fix_result['constants'])
            
            # Generate constants file content
            constants_content = self._generate_constants_file(constants_needed)
            
            # Add imports if needed
            if constants_needed:
                refactored_code = self._add_constants_import(refactored_code)
            
            confidence = self._calculate_confidence(fixes_applied, fixable_patterns)
            
            return SimpleFixResult(
                success=True,
                original_code=source_code,
                refactored_code=refactored_code,
                fixes_applied=fixes_applied,
                confidence_score=confidence,
                constants_generated=constants_content
            )
            
        except Exception as e:
            return SimpleFixResult(
                success=False,
                original_code=source_code,
                refactored_code=source_code,
                fixes_applied=[],
                error=str(e)
            )
    
    def _apply_single_fix(self, code: str, pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply a single fix to the code"""
        pattern_type = pattern.get('type')
        line_num = pattern.get('line', 0)
        
        lines = code.split('\n')
        if line_num < 1 or line_num > len(lines):
            return None
        
        original_line = lines[line_num - 1]  # Convert to 0-based indexing
        
        if pattern_type == 'magic_numbers':
            fix_result = self._fix_magic_numbers(original_line, pattern)
        elif pattern_type == 'reproducibility':
            fix_result = self._fix_reproducibility(original_line, pattern)
        elif pattern_type == 'data_leakage':
            fix_result = self._fix_data_leakage(original_line, pattern)
        else:
            return None
        
        if fix_result and fix_result['new_line'] != original_line:
            # Replace the line in code
            lines[line_num - 1] = fix_result['new_line']
            refactored_code = '\n'.join(lines)
            
            return {
                'refactored_code': refactored_code,
                'fix_info': {
                    'type': pattern_type,
                    'line': line_num,
                    'description': fix_result['description'],
                    'original': original_line.strip(),
                    'fixed': fix_result['new_line'].strip()
                },
                'constants': fix_result.get('constants', set())
            }
        
        return None
    
    def _fix_magic_numbers(self, line: str, pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fix magic numbers in a line of code"""
        new_line = line
        constants_needed = set()
        
        # Common ML hyperparameter patterns
        magic_patterns = [
            (r'n_estimators\s*=\s*(\d+)', r'n_estimators=N_ESTIMATORS'),
            (r'max_depth\s*=\s*(\d+)', r'max_depth=MAX_DEPTH'),
            (r'learning_rate\s*=\s*([\d.]+)', r'learning_rate=LEARNING_RATE'),
            (r'batch_size\s*=\s*(\d+)', r'batch_size=BATCH_SIZE'),
            (r'epochs\s*=\s*(\d+)', r'epochs=EPOCHS'),
        ]
        
        changed = False
        for pattern_re, replacement in magic_patterns:
            match = re.search(pattern_re, new_line)
            if match:
                # Extract the constant name from replacement
                const_name = replacement.split('=')[1]
                constants_needed.add(const_name)
                new_line = re.sub(pattern_re, replacement, new_line)
                changed = True
        
        if changed:
            return {
                'new_line': new_line,
                'description': f'Extracted magic numbers to constants',
                'constants': constants_needed
            }
        
        return None
    
    def _fix_reproducibility(self, line: str, pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fix reproducibility issues by adding random_state parameters"""
        new_line = line
        
        # Patterns for ML functions that need random_state
        reproducibility_patterns = [
            # RandomForestClassifier/Regressor
            (r'(RandomForest\w+)\(((?:(?!random_state)[^)])*)\)', r'\1(\2, random_state=42)'),
            # train_test_split
            (r'(train_test_split)\(((?:(?!random_state)[^)])*)\)', r'\1(\2, random_state=42)'),
            # Other sklearn models
            (r'(SVC|SVR|KMeans|DecisionTree\w+|LogisticRegression)\(((?:(?!random_state)[^)])*)\)', r'\1(\2, random_state=42)'),
        ]
        
        for pattern_re, replacement in reproducibility_patterns:
            if re.search(pattern_re, new_line):
                # Clean up extra commas and spaces
                new_line = re.sub(pattern_re, replacement, new_line)
                new_line = re.sub(r',\s*,', ',', new_line)  # Remove double commas
                new_line = re.sub(r'\(\s*,', '(', new_line)  # Remove leading comma
                
                return {
                    'new_line': new_line,
                    'description': f'Added random_state=42 for reproducibility'
                }
        
        return None
    
    def _fix_data_leakage(self, line: str, pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fix data leakage by adding warning comments"""
        # For data leakage, we add a warning comment above the problematic line
        indent = len(line) - len(line.lstrip())
        warning_comment = ' ' * indent + '# WARNING: Potential data leakage - review preprocessing order'
        new_line = warning_comment + '\n' + line
        
        return {
            'new_line': new_line,
            'description': f'Added data leakage warning comment'
        }
    
    def _generate_constants_file(self, constants_needed: Set[str]) -> str:
        """Generate constants file content"""
        if not constants_needed:
            return ""
        
        content = [
            "#!/usr/bin/env python3",
            '"""',
            'ML Configuration Constants',
            '',
            'Constants extracted from magic numbers to improve',
            'maintainability and experimentation flexibility.',
            '"""',
            '',
            '# Model hyperparameters'
        ]
        
        # Add constants with reasonable defaults
        constant_defaults = {
            'N_ESTIMATORS': 100,
            'MAX_DEPTH': 15,
            'LEARNING_RATE': 0.001,
            'BATCH_SIZE': 32,
            'EPOCHS': 100,
            'HIDDEN_SIZE': 128
        }
        
        for const_name in sorted(constants_needed):
            default_value = constant_defaults.get(const_name, 42)
            comment = f"  # {const_name.lower().replace('_', ' ').title()}"
            content.append(f"{const_name} = {default_value}{comment}")
        
        content.extend([
            '',
            '# Random seed for reproducibility',
            'RANDOM_SEED = 42'
        ])
        
        return '\n'.join(content)
    
    def _add_constants_import(self, code: str) -> str:
        """Add import for constants if needed"""
        if 'from constants import' in code or 'import constants' in code:
            return code
        
        # Add import at the top after existing imports
        lines = code.split('\n')
        import_lines = []
        other_lines = []
        in_imports = True
        
        for line in lines:
            if in_imports and (line.startswith('import ') or line.startswith('from ') or line.strip() == ''):
                import_lines.append(line)
            else:
                in_imports = False
                other_lines.append(line)
        
        # Add constants import
        import_lines.append('from constants import *')
        import_lines.append('')
        
        return '\n'.join(import_lines + other_lines)
    
    def _calculate_confidence(self, fixes_applied: List[Dict[str, Any]], 
                             total_patterns: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on fixes applied"""
        if not total_patterns:
            return 1.0
        
        # Simple confidence calculation
        success_rate = len(fixes_applied) / len(total_patterns)
        return success_rate * 0.9  # Conservative confidence
    
    def preview_fixes(self, source_code: str, detected_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate preview of fixes without applying them"""
        # Normalize pattern types
        normalized_patterns = []
        for pattern in detected_patterns:
            original_type = pattern.get('type', '')
            mapped_type = self.pattern_type_mapping.get(original_type, original_type)
            
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
        
        # Generate preview snippets
        lines = source_code.split('\n')
        for pattern in fixable_patterns[:5]:  # Show first 5
            line_num = pattern.get('line', 1) - 1
            if 0 <= line_num < len(lines):
                original_line = lines[line_num]
                suggested_line = self._generate_preview_line(original_line, pattern)
                
                preview['preview_snippets'].append({
                    'line_number': pattern.get('line'),
                    'pattern_type': pattern.get('type'),
                    'original': original_line.strip(),
                    'suggested': suggested_line.strip(),
                    'explanation': self._generate_fix_description(pattern)
                })
        
        return preview
    
    def _generate_fix_description(self, pattern: Dict[str, Any]) -> str:
        """Generate description of what fix would be applied"""
        pattern_type = pattern.get('type', 'unknown')
        
        descriptions = {
            'magic_numbers': "Extract magic number to named constant",
            'reproducibility': "Add random_state=42 for reproducibility", 
            'data_leakage': "Add warning comment about potential data leakage"
        }
        
        return descriptions.get(pattern_type, "Apply automated fix")
    
    def _generate_preview_line(self, original_line: str, pattern: Dict[str, Any]) -> str:
        """Generate preview of how line would be fixed"""
        pattern_type = pattern.get('type', 'unknown')
        
        if pattern_type == 'magic_numbers':
            # Preview magic number replacements
            line = original_line
            line = re.sub(r'n_estimators\s*=\s*\d+', 'n_estimators=N_ESTIMATORS', line)
            line = re.sub(r'max_depth\s*=\s*\d+', 'max_depth=MAX_DEPTH', line)
            return line
            
        elif pattern_type == 'reproducibility':
            # Preview random_state additions
            if 'RandomForest' in original_line and 'random_state' not in original_line:
                return re.sub(r'\)', ', random_state=42)', original_line)
            elif 'train_test_split' in original_line and 'random_state' not in original_line:
                return re.sub(r'\)', ', random_state=42)', original_line)
        
        elif pattern_type == 'data_leakage':
            indent = len(original_line) - len(original_line.lstrip())
            warning = ' ' * indent + '# WARNING: Potential data leakage\n'
            return warning + original_line
        
        return original_line


def create_simple_refactoring_engine() -> SimpleMLRefactoring:
    """Factory function to create simple refactoring engine"""
    return SimpleMLRefactoring()