#!/usr/bin/env python3
"""
Simple test script to verify the ML Playground app works correctly.
"""

import os
import sys
import tempfile
import shutil
from unittest.mock import patch

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_app_import():
    """Test that the app can be imported without errors"""
    try:
        from app import app
        print("✅ App imports successfully")
        return True
    except Exception as e:
        print(f"❌ App import failed: {e}")
        return False

def test_app_configuration():
    """Test that the app has proper configuration"""
    try:
        from app import app
        assert app.config['SECRET_KEY'] is not None
        assert app.config['UPLOAD_FOLDER'] == 'uploads'
        print("✅ App configuration is correct")
        return True
    except Exception as e:
        print(f"❌ App configuration test failed: {e}")
        return False

def test_directories_exist():
    """Test that required directories exist"""
    required_dirs = ['templates', 'static', 'datasets', 'uploads']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"❌ Required directory '{dir_name}' does not exist")
            return False
    print("✅ All required directories exist")
    return True

def test_requirements():
    """Test that all required packages can be imported"""
    required_packages = [
        'flask',
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError as e:
            print(f"❌ Required package '{package}' not available: {e}")
            return False
    
    print("✅ All required packages are available")
    return True

def main():
    """Run all tests"""
    print("🧪 Testing ML Playground App...")
    print("=" * 50)
    
    tests = [
        test_app_import,
        test_app_configuration,
        test_directories_exist,
        test_requirements
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The app is ready for deployment.")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues before deployment.")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 