#!/usr/bin/env python3
"""
Simple test script to verify backend functionality
"""

import requests
import json
import time

def test_backend_health():
    """Test if backend is running"""
    try:
        response = requests.get("http://127.0.0.1:8000/model/info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend is running!")
            print(f"   Project: {data.get('project_name', 'N/A')}")
            print(f"   System Status: {data.get('system_status', 'N/A')}")
            print(f"   Recommended Endpoint: {data.get('recommended_endpoint', 'N/A')}")
            return True
        else:
            print(f"❌ Backend responded with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend not accessible: {e}")
        return False

def test_ultra_fast_endpoint():
    """Test the ultra-fast endpoint"""
    test_data = {
        "question": "do braces hurt?",
        "context": "pain and discomfort",
        "answers": ["Yes, initially they can cause some discomfort but you get used to them over time."]
    }
    
    try:
        print("\n🔄 Testing ultra-fast endpoint...")
        start_time = time.time()
        
        response = requests.post(
            "http://127.0.0.1:8000/summarize/ultra-fast",
            json=test_data,
            headers={"Authorization": "Bearer test_token"},
            timeout=30
        )
        
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Ultra-fast endpoint working!")
            print(f"   Processing time: {total_time:.3f} seconds")
            print(f"   Patient Summary: {data.get('patient_summary', '')[:100]}...")
            print(f"   Clinical Summary: {data.get('clinician_summary', '')[:100]}...")
            return True
        else:
            print(f"❌ Ultra-fast endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Ultra-fast endpoint error: {e}")
        return False

def test_swasthyasetu_endpoint():
    """Test the SwasthyaSetu T5 endpoint"""
    test_data = {
        "question": "do braces hurt?",
        "context": "pain and discomfort",
        "answers": ["Yes, initially they can cause some discomfort but you get used to them over time."]
    }
    
    try:
        print("\n🔄 Testing SwasthyaSetu T5 endpoint...")
        start_time = time.time()
        
        response = requests.post(
            "http://127.0.0.1:8000/summarize/swasthyasetu-t5",
            json=test_data,
            headers={"Authorization": "Bearer test_token"},
            timeout=30
        )
        
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SwasthyaSetu T5 endpoint working!")
            print(f"   Processing time: {total_time:.3f} seconds")
            print(f"   Model used: {data.get('processing_info', {}).get('model_used', 'N/A')}")
            print(f"   Patient Summary: {data.get('patient_summary', '')[:100]}...")
            print(f"   Clinical Summary: {data.get('clinician_summary', '')[:100]}...")
            return True
        else:
            print(f"❌ SwasthyaSetu T5 endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ SwasthyaSetu T5 endpoint error: {e}")
        return False

if __name__ == "__main__":
    print("🏥 SwasthyaSetu Backend Test")
    print("=" * 40)
    
    # Test backend health
    if not test_backend_health():
        print("\n❌ Backend is not running. Please start it first:")
        print("   cd backend")
        print("   source .venv/bin/activate")
        print("   python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000")
        exit(1)
    
    # Test endpoints
    ultra_fast_working = test_ultra_fast_endpoint()
    swasthyasetu_working = test_swasthyasetu_endpoint()
    
    print("\n📊 Test Results:")
    print(f"   Ultra-Fast Endpoint: {'✅ Working' if ultra_fast_working else '❌ Failed'}")
    print(f"   SwasthyaSetu T5 Endpoint: {'✅ Working' if swasthyasetu_working else '❌ Failed'}")
    
    if ultra_fast_working:
        print("\n💡 Ultra-fast endpoint is working - you can process your braces data!")
        print("   Use: POST /summarize/ultra-fast")
    
    if swasthyasetu_working:
        print("\n💡 SwasthyaSetu T5 endpoint is working - AI-powered summaries available!")
        print("   Use: POST /summarize/swasthyasetu-t5")
    
    print("\n✨ Backend test complete!")
