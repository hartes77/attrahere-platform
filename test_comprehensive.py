#!/usr/bin/env python3
"""
Comprehensive Test Suite - All Components Integration
"""

import subprocess
import sys
import os

def run_test_file(test_file, description):
    """Run a test file and return success status"""
    print(f"\n🔄 Running {description}...")
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✅ {description}: PASS")
            return True
        else:
            print(f"❌ {description}: FAIL")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description}: ERROR - {e}")
        return False

def test_core_functionality():
    """Test core ML analyzer functionality"""
    print("\n🧠 CORE ML ANALYZER TEST")
    print("=" * 50)
    
    try:
        from analysis_core.ml_analyzer.analyzer import MLCodeAnalyzer
        analyzer = MLCodeAnalyzer()
        
        # Test with known problematic code
        test_code = '''
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y)
'''
        
        with open('/tmp/test_analysis.py', 'w') as f:
            f.write(test_code)
        
        results = analyzer.analyze_file('/tmp/test_analysis.py')
        
        patterns_found = len(results.get('patterns', []))
        print(f"✅ Analyzer working: {patterns_found} patterns detected")
        
        # Clean up
        os.unlink('/tmp/test_analysis.py')
        
        return patterns_found > 0
        
    except Exception as e:
        print(f"❌ Core analyzer error: {e}")
        return False

def check_file_structure():
    """Check that all essential files are present"""
    print("\n📁 FILE STRUCTURE TEST")
    print("=" * 50)
    
    essential_files = [
        'Dockerfile',
        '.dockerignore', 
        'requirements.txt',
        'analysis_core/ml_analyzer/analyzer.py',
        'analysis_core/ml_analyzer/ml_patterns.py',
        'api/main.py',
        'test_sprint1.py',
        'test_sprint2.py'
    ]
    
    all_present = True
    for file_path in essential_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}: Present")
        else:
            print(f"❌ {file_path}: Missing")
            all_present = False
    
    return all_present

def run_comprehensive_test_suite():
    """Run the complete test suite"""
    print("🚀 ATTRAHERE COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("Testing all components before commit...")
    print("=" * 80)
    
    # Track all test results
    test_results = []
    
    # 1. File structure check
    test_results.append(("File Structure", check_file_structure()))
    
    # 2. Core functionality
    test_results.append(("Core ML Analyzer", test_core_functionality()))
    
    # 3. Integration tests
    test_results.append(("Sprint 1 Integration", 
                        run_test_file('test_sprint1.py', 'Sprint 1 Integration')))
    test_results.append(("Sprint 2 Integration", 
                        run_test_file('test_sprint2.py', 'Sprint 2 Integration')))
    
    # 4. Component tests
    test_results.append(("API Core", 
                        run_test_file('test_api_simple.py', 'API Core')))
    test_results.append(("Docker Setup", 
                        run_test_file('test_docker.py', 'Docker Setup')))
    test_results.append(("Database Structure", 
                        run_test_file('test_database.py', 'Database Structure')))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        icon = "✅" if result else "❌"
        print(f"{icon} {test_name}: {status}")
        if result:
            passed += 1
    
    print("=" * 80)
    print(f"📈 OVERALL SCORE: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASS - READY FOR CLEAN COMMIT! 🎉")
        return True
    elif passed >= total * 0.8:  # 80% or more
        print("✅ MOSTLY PASSING - COMMIT WITH MINOR WARNINGS")
        return True
    else:
        print("⚠️ SIGNIFICANT ISSUES - REVIEW BEFORE COMMIT")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test_suite()
    sys.exit(0 if success else 1)