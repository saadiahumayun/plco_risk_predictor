#!/usr/bin/env python
"""
Quick test script to verify backend is working properly.

Usage: python scripts/test_backend.py
"""
import requests
import json
import sys
from datetime import datetime


def test_endpoint(name, method, url, data=None, expected_status=200):
    """Test a single endpoint."""
    print(f"\n[{name}]")
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            print(f"❌ Unknown method: {method}")
            return False
        
        print(f"  Status: {response.status_code}")
        
        if response.status_code == expected_status:
            print(f"  ✓ Success")
            if response.status_code == 200:
                data = response.json()
                print(f"  Response keys: {list(data.keys())[:5]}...")
            return True
        else:
            print(f"  ❌ Expected {expected_status}, got {response.status_code}")
            print(f"  Response: {response.text[:200]}...")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"  ❌ Connection error - is the backend running?")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    base_url = "http://localhost:8000"
    
    print("=" * 80)
    print("Breast Cancer Risk Prediction Backend Test")
    print(f"Base URL: {base_url}")
    print(f"Time: {datetime.now()}")
    print("=" * 80)
    
    # Test data
    prediction_data = {
        "demographics": {
            "age": 55,
            "race": "white"
        },
        "reproductive_history": {
            "age_at_menarche": 12,
            "age_at_first_birth": 28,
            "number_of_live_births": 2,
            "first_degree_bc": 1
        },
        "body_metrics": {
            "current_bmi": 26.5,
            "height_cm": 165,
            "weight_kg": 72
        },
        "medical_history": {
            "personal_cancer_history": False,
            "benign_breast_disease": True,
            "breast_biopsies": 1
        }
    }
    
    # Run tests
    tests = [
        ("Root Endpoint", "GET", f"{base_url}/", None, 200),
        ("Health Check", "GET", f"{base_url}/health", None, 200),
        ("API Docs", "GET", f"{base_url}/api/v1/docs", None, 200),
        ("Model Info", "GET", f"{base_url}/api/v1/models/info", None, 200),
        ("Features", "GET", f"{base_url}/api/v1/features", None, 200),
        ("Prediction", "POST", f"{base_url}/api/v1/predict", prediction_data, 200),
        ("Validation", "POST", f"{base_url}/api/v1/validate", prediction_data, 200),
    ]
    
    results = []
    for test in tests:
        success = test_endpoint(*test)
        results.append((test[0], success))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary:")
    print("-" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"  {name:<30} {status}")
    
    print("-" * 80)
    print(f"Total: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n✅ All tests passed! Backend is working properly.")
        
        # Show sample prediction response
        print("\nSample prediction response:")
        response = requests.post(f"{base_url}/api/v1/predict", json=prediction_data)
        if response.status_code == 200:
            data = response.json()
            print(f"  Risk Score: {data.get('risk_score', 'N/A'):.3f}")
            print(f"  Risk Category: {data.get('risk_category', 'N/A')}")
            print(f"  Percentile: {data.get('percentile', 'N/A')}")
            print(f"  Model Version: {data.get('model_version', 'N/A')}")
    else:
        print("\n❌ Some tests failed. Please check the backend logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()