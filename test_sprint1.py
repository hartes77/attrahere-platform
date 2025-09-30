#!/usr/bin/env python3
"""
Sprint 1 Integration Test - Verifica che Data Leakage e Random Seeds funzionino
"""

from analysis_core.ml_analyzer.analyzer import MLCodeAnalyzer

def test_data_leakage_detection():
    """Test Data Leakage detection with problematic code"""
    
    # Codice problematico - preprocessing su tutto il dataset prima dello split
    problematic_code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data
X = pd.read_csv('data.csv')
y = X.pop('target')

# PROBLEMATICO: Preprocessing su tutto il dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Data leakage qui!

# Split dopo preprocessing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Training
model = LogisticRegression()
model.fit(X_train, y_train)
"""
    
    analyzer = MLCodeAnalyzer()
    result = analyzer.analyze_code_string(problematic_code, "data_leakage_test.py")
    
    print("🔍 DATA LEAKAGE TEST")
    print(f"Patterns found: {len(result.get('patterns', []))}")
    
    for pattern in result.get('patterns', []):
        print(f"  - Type: {pattern.get('type')}")
        print(f"    Line: {pattern.get('line')}")
        print(f"    Message: {pattern.get('message')}")
    
    # Verifica che il data leakage sia stato rilevato
    data_leakage_found = any(p.get('type') == 'data_leakage_preprocessing' for p in result.get('patterns', []))
    print(f"✅ Data Leakage Detected: {data_leakage_found}")
    
    return data_leakage_found

def test_missing_random_state():
    """Test Missing Random State detection"""
    
    # Codice problematico - mancano random_state
    problematic_code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
X = pd.read_csv('data.csv')
y = X.pop('target')

# PROBLEMATICO: Manca random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# PROBLEMATICO: Manca random_state
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
"""
    
    analyzer = MLCodeAnalyzer()
    result = analyzer.analyze_code_string(problematic_code, "random_state_test.py")
    
    print("\n🎲 MISSING RANDOM STATE TEST")
    print(f"Patterns found: {len(result.get('patterns', []))}")
    
    for pattern in result.get('patterns', []):
        print(f"  - Type: {pattern.get('type')}")
        print(f"    Line: {pattern.get('line')}")
        print(f"    Message: {pattern.get('message')}")
    
    # Verifica che i missing random state siano stati rilevati
    random_state_issues = [p for p in result.get('patterns', []) if 'RANDOM' in p.get('type', '').upper()]
    print(f"✅ Random State Issues Detected: {len(random_state_issues)}")
    
    return len(random_state_issues) > 0

def test_clean_code():
    """Test that clean code doesn't generate false positives"""
    
    # Codice pulito - senza problemi
    clean_code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load data
X = pd.read_csv('data.csv')
y = X.pop('target')

# CORRETTO: Split prima del preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CORRETTO: Preprocessing solo sui dati di training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Solo transform per test

# CORRETTO: Random state impostato
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
"""
    
    analyzer = MLCodeAnalyzer()
    result = analyzer.analyze_code_string(clean_code, "clean_test.py")
    
    print("\n✨ CLEAN CODE TEST")
    print(f"Patterns found: {len(result.get('patterns', []))}")
    
    for pattern in result.get('patterns', []):
        print(f"  - Type: {pattern.get('type')}")
        print(f"    Line: {pattern.get('line')}")
        print(f"    Message: {pattern.get('message')}")
    
    # Verifica nessun false positive
    critical_issues = [p for p in result.get('patterns', []) if p.get('type') in ['data_leakage_preprocessing', 'missing_random_seed']]
    print(f"✅ Critical Issues (should be 0): {len(critical_issues)}")
    
    return len(critical_issues) == 0

if __name__ == "__main__":
    print("🚀 SPRINT 1 INTEGRATION TEST")
    print("=" * 50)
    
    # Test tutti i detector
    data_leakage_ok = test_data_leakage_detection()
    random_state_ok = test_missing_random_state()
    clean_code_ok = test_clean_code()
    
    print("\n" + "=" * 50)
    print("📊 SPRINT 1 RESULTS")
    print(f"✅ Data Leakage Detection: {'PASS' if data_leakage_ok else 'FAIL'}")
    print(f"✅ Random State Detection: {'PASS' if random_state_ok else 'FAIL'}")
    print(f"✅ Clean Code (No False Positives): {'PASS' if clean_code_ok else 'FAIL'}")
    
    all_tests_pass = data_leakage_ok and random_state_ok and clean_code_ok
    
    if all_tests_pass:
        print("\n🎉 SPRINT 1 - DETECTOR FUNCTIONALITY: 100% COMPLETE!")
        print("Ready for API integration and frontend updates!")
    else:
        print("\n❌ Some tests failed. Check detector implementation.")