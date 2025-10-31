#!/usr/bin/env python3
"""
Test script to verify the AI photo analysis service is working
"""

import requests
import json
import time
import subprocess
import sys
from pathlib import Path

def test_health_endpoint():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_models_loading():
    """Test that models can be loaded"""
    try:
        from models import model_manager
        
        # Test face model loading (will trigger lazy loading)
        print("Testing face model loading...")
        # We can't test without an actual image, but we can check if the model loads
        model_manager._load_face_model()
        print("✅ Face model loaded successfully")
        
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    print("🧪 Testing AI Photo Analysis Service")
    print("=" * 50)
    
    # Test 1: Model loading
    print("\n1. Testing model loading...")
    models_ok = test_models_loading()
    
    # Test 2: Start service and test health
    print("\n2. Testing service startup...")
    print("Starting service in background...")
    
    # Start the service
    process = subprocess.Popen([
        sys.executable, "main.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a moment for startup
    time.sleep(5)
    
    # Test health endpoint
    health_ok = test_health_endpoint()
    
    # Clean up
    process.terminate()
    process.wait()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Models Loading: {'✅ PASS' if models_ok else '❌ FAIL'}")
    print(f"   Service Health: {'✅ PASS' if health_ok else '❌ FAIL'}")
    
    if models_ok and health_ok:
        print("\n🎉 All tests passed! Service is ready to use.")
        print("\nTo start the service:")
        print("   python main.py")
        print("\nAPI will be available at: http://localhost:8000")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()