#!/usr/bin/env python3
"""
Sprint 2 Integration Test - Verifica che Hardcoded Thresholds e Inefficient Data Loading funzionino
"""

from analysis_core.ml_analyzer.analyzer import MLCodeAnalyzer

def test_hardcoded_thresholds_detection():
    """Test Hardcoded Thresholds detection with problematic code"""
    
    # Codice problematico - hardcoded thresholds
    problematic_code = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# PROBLEMATICO: Hardcoded threshold senza giustificazione
threshold = 0.73625  # Magic number
anomaly_threshold = 0.05823  # Overly precise

# PROBLEMATICO: Magic numbers in comparisons
if y_proba.max() > 0.9847:
    confidence = "high"
elif y_proba.max() > 0.6123:
    confidence = "medium"

# PROBLEMATICO: Performance threshold without context
performance_cutoff = 0.8234567
"""
    
    analyzer = MLCodeAnalyzer()
    result = analyzer.analyze_code_string(problematic_code, "hardcoded_thresholds_test.py")
    
    print("🎯 HARDCODED THRESHOLDS TEST")
    print(f"Patterns found: {len(result.get('patterns', []))}")
    
    for pattern in result.get('patterns', []):
        print(f"  - Type: {pattern.get('type')}")
        print(f"    Line: {pattern.get('line')}")
        print(f"    Message: {pattern.get('message')}")
    
    # Verifica che i threshold hardcoded siano stati rilevati
    threshold_issues = [p for p in result.get('patterns', []) if 'threshold' in p.get('type', '').lower()]
    print(f"✅ Hardcoded Threshold Issues Detected: {len(threshold_issues)}")
    
    return len(threshold_issues) > 0

def test_inefficient_data_loading_detection():
    """Test Inefficient Data Loading detection"""
    
    # Codice problematico - inefficient data loading
    problematic_code = """
import pandas as pd
import os

# PROBLEMATICO: Loading large file without chunking
df = pd.read_csv('huge_dataset.csv')

# PROBLEMATICO: Redundant loading
df1 = pd.read_csv('data.csv')
df2 = pd.read_csv('data.csv')  # Same file loaded again!

# PROBLEMATICO: No dtype specification
df_untyped = pd.read_csv('numeric_data.csv')

# PROBLEMATICO: Loading all columns from wide dataset
df_all_columns = pd.read_csv('wide_features.csv')

# PROBLEMATICO: Inefficient iteration
for i in range(len(df)):
    row = df.iloc[i]  # Very slow indexing
    result = row['value'] * 2
"""
    
    analyzer = MLCodeAnalyzer()
    result = analyzer.analyze_code_string(problematic_code, "inefficient_loading_test.py")
    
    print("\\n💾 INEFFICIENT DATA LOADING TEST")
    print(f"Patterns found: {len(result.get('patterns', []))}")
    
    for pattern in result.get('patterns', []):
        print(f"  - Type: {pattern.get('type')}")
        print(f"    Line: {pattern.get('line')}")
        print(f"    Message: {pattern.get('message')}")
    
    # Verifica che i data loading issues siano stati rilevati
    loading_issues = [p for p in result.get('patterns', []) if 
                     any(keyword in p.get('type', '').lower() for keyword in 
                         ['loading', 'dtype', 'redundant', 'inefficient', 'chunking'])]
    print(f"✅ Data Loading Issues Detected: {len(loading_issues)}")
    
    return len(loading_issues) > 0

def test_clean_code_sprint2():
    """Test that clean code doesn't generate false positives"""
    
    # Codice pulito - best practices
    clean_code = """
import pandas as pd
import functools

# CORRETTO: Well-documented business threshold
BUSINESS_PRECISION_REQUIREMENT = 0.85  # Required precision for business acceptance
CONFIDENCE_LEVELS = {
    'high': 0.90,    # 90% probability threshold for high confidence
    'medium': 0.70,  # 70% probability threshold for medium confidence
    'low': 0.50      # 50% probability threshold for low confidence
}

# CORRETTO: Efficient data loading with dtypes
dtype_spec = {
    'user_id': 'int32',
    'score': 'float32',
    'category': 'category'
}
df_optimized = pd.read_csv('data.csv', dtype=dtype_spec)

# CORRETTO: Load only required columns with dtypes
required_columns = ['feature1', 'feature2', 'target']
df_selective = pd.read_csv('wide_dataset.csv', usecols=required_columns, dtype={'feature1': 'float32'})

# CORRETTO: Cached loading to avoid redundancy
@functools.lru_cache(maxsize=None)
def load_cached_data(file_path):
    return pd.read_csv(file_path, dtype='string')  # Example with dtype for caching

# CORRETTO: Vectorized operation instead of iteration
df['result'] = df['value'] * 2  # Much faster than loops
"""
    
    analyzer = MLCodeAnalyzer()
    result = analyzer.analyze_code_string(clean_code, "clean_sprint2_test.py")
    
    print("\\n✨ CLEAN CODE SPRINT 2 TEST")
    print(f"Patterns found: {len(result.get('patterns', []))}")
    
    for pattern in result.get('patterns', []):
        print(f"  - Type: {pattern.get('type')}")
        print(f"    Line: {pattern.get('line')}")
        print(f"    Message: {pattern.get('message')}")
    
    # Verifica nessun false positive per Sprint 2 patterns
    sprint2_issues = [p for p in result.get('patterns', []) if 
                     any(keyword in p.get('type', '').lower() for keyword in 
                         ['threshold', 'loading', 'dtype', 'redundant', 'inefficient'])]
    print(f"✅ Sprint 2 Issues (should be 0): {len(sprint2_issues)}")
    
    return len(sprint2_issues) == 0

if __name__ == "__main__":
    print("🚀 SPRINT 2 INTEGRATION TEST")
    print("=" * 50)
    
    # Test tutti i detector Sprint 2
    thresholds_ok = test_hardcoded_thresholds_detection()
    loading_ok = test_inefficient_data_loading_detection()
    clean_code_ok = test_clean_code_sprint2()
    
    print("\\n" + "=" * 50)
    print("📊 SPRINT 2 RESULTS")
    print(f"✅ Hardcoded Thresholds Detection: {'PASS' if thresholds_ok else 'FAIL'}")
    print(f"✅ Inefficient Data Loading Detection: {'PASS' if loading_ok else 'FAIL'}")
    print(f"✅ Clean Code (No False Positives): {'PASS' if clean_code_ok else 'FAIL'}")
    
    all_tests_pass = thresholds_ok and loading_ok and clean_code_ok
    
    if all_tests_pass:
        print("\\n🎉 SPRINT 2 - BEST PRACTICE & PERFORMANCE: 100% COMPLETE!")
        print("Ready for production deployment with advanced ML optimization!")
    else:
        print("\\n❌ Some tests failed. Check detector implementation.")