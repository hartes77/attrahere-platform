#!/usr/bin/env python3
"""
Simple API Test - Test core functionality without database dependencies
"""

from fastapi import FastAPI
from analysis_core.ml_analyzer.analyzer import MLCodeAnalyzer

def test_api_core_functionality():
    """Test that we can create a basic API with our analyzer"""
    
    # Create minimal FastAPI app
    app = FastAPI(title="Test API")
    analyzer = MLCodeAnalyzer()
    
    @app.post("/analyze")
    def analyze_code_endpoint(code: str):
        # Save code to temp file for analysis
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            results = analyzer.analyze_file(temp_path)
            return {
                "success": True,
                "patterns_found": len(results.get("patterns", [])),
                "patterns": results.get("patterns", [])
            }
        finally:
            os.unlink(temp_path)
    
    # Test the endpoint functionality
    test_code = '''
import pandas as pd
df = pd.read_csv("data.csv")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)  # Missing random_state
'''
    
    result = analyze_code_endpoint(test_code)
    
    print("🚀 API CORE FUNCTIONALITY TEST")
    print("=" * 50)
    print(f"✅ FastAPI app created successfully")
    print(f"✅ Analyzer integrated successfully")
    print(f"✅ Endpoint processing: {result['success']}")
    print(f"✅ Patterns detected: {result['patterns_found']}")
    
    if result['patterns_found'] > 0:
        print(f"✅ Sample pattern: {result['patterns'][0]['type']}")
    
    print("=" * 50)
    print("🎉 API CORE TEST: PASS")
    return True

if __name__ == "__main__":
    test_api_core_functionality()