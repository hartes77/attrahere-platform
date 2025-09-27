#!/usr/bin/env python3
"""
Debug script for GPU detector issues
"""
import sys
from pathlib import Path

# Add analysis_core to path
sys.path.insert(0, str(Path(__file__).parent))

from analysis_core.ml_analyzer.analyzer import MLCodeAnalyzer

def debug_file(file_path):
    """Debug a specific file and show exact patterns detected"""
    print(f"\n🔍 DEBUGGING: {file_path}")
    print("=" * 60)
    
    analyzer = MLCodeAnalyzer()
    
    # Read file content
    with open(file_path, 'r') as f:
        code = f.read()
    
    # Analyze
    result = analyzer.analyze_file(file_path)
    
    patterns = result.get('patterns', [])
    print(f"📊 Total patterns found: {len(patterns)}")
    
    for pattern in patterns:
        print(f"\n🚨 Pattern: {pattern.get('pattern_type', 'unknown')}")
        print(f"   Line: {pattern.get('line_number', 'unknown')}")
        print(f"   Message: {pattern.get('message', 'no message')}")
        print(f"   Code: {pattern.get('code_snippet', 'N/A')}")

if __name__ == "__main__":
    # Debug the problematic file
    debug_file("tests/clean_code/clean_gpu_usage.py")