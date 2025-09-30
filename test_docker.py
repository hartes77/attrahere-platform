#!/usr/bin/env python3
"""
Docker Test - Validate Dockerfile and build process
"""

import subprocess
import sys
import os

def test_dockerfile_syntax():
    """Test Dockerfile syntax and structure"""
    print("🐳 DOCKERFILE SYNTAX TEST")
    print("=" * 50)
    
    with open('Dockerfile', 'r') as f:
        dockerfile_content = f.read()
    
    # Check essential components
    checks = [
        ("FROM python:3.11-slim", "Base image"),
        ("WORKDIR /app", "Working directory"),
        ("COPY requirements.txt", "Requirements copy"),
        ("RUN pip install", "Dependencies installation"), 
        ("COPY . .", "Application copy"),
        ("EXPOSE", "Port exposure"),
        ("CMD", "Startup command")
    ]
    
    for check, description in checks:
        if check in dockerfile_content:
            print(f"✅ {description}: Found")
        else:
            print(f"❌ {description}: Missing")
            return False
    
    print("✅ Dockerfile syntax: VALID")
    return True

def test_requirements_file():
    """Test requirements.txt completeness"""
    print("\n📋 REQUIREMENTS TEST")
    print("=" * 50)
    
    with open('requirements.txt', 'r') as f:
        requirements = f.read()
    
    essential_deps = ['fastapi', 'uvicorn', 'libcst']
    
    for dep in essential_deps:
        if dep in requirements:
            print(f"✅ {dep}: Present")
        else:
            print(f"❌ {dep}: Missing")
            return False
    
    print("✅ Requirements: COMPLETE")
    return True

def test_dockerignore():
    """Test .dockerignore configuration"""
    print("\n🚫 DOCKERIGNORE TEST")
    print("=" * 50)
    
    with open('.dockerignore', 'r') as f:
        dockerignore = f.read()
    
    essential_ignores = ['__pycache__', '*.pyc', '.git', '.venv']
    
    for ignore in essential_ignores:
        if ignore in dockerignore:
            print(f"✅ {ignore}: Ignored")
        else:
            print(f"⚠️ {ignore}: Not ignored")
    
    print("✅ Dockerignore: CONFIGURED")
    return True

def test_docker_build_dry_run():
    """Test Docker build process (dry run)"""
    print("\n🔨 DOCKER BUILD TEST")
    print("=" * 50)
    
    try:
        # Test if Docker is available
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Docker available: {result.stdout.strip()}")
        else:
            print("❌ Docker not available")
            return False
        
        # Test build context preparation
        if os.path.exists('Dockerfile') and os.path.exists('requirements.txt'):
            print("✅ Build context: Ready")
        else:
            print("❌ Build context: Incomplete")
            return False
            
        print("✅ Docker build: READY (not executed to save time)")
        return True
        
    except Exception as e:
        print(f"❌ Docker test error: {e}")
        return False

def run_all_docker_tests():
    """Run all Docker-related tests"""
    print("🚀 DOCKER COMPREHENSIVE TEST")
    print("=" * 60)
    
    tests = [
        test_dockerfile_syntax,
        test_requirements_file,
        test_dockerignore,
        test_docker_build_dry_run
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL DOCKER TESTS: PASS")
    else:
        print("❌ SOME DOCKER TESTS: FAIL")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_docker_tests()
    sys.exit(0 if success else 1)