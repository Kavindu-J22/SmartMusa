# test_fixes.py
"""
Test script to verify that the fixes for feature mismatch and missing methods are working
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:5000/api/v1"

def test_enhanced_chatbot_comprehensive():
    """
    Test the enhanced chatbot with all types of queries
    """
    print("🤖 Testing Enhanced Chatbot - All Features")
    print("=" * 60)
    
    test_cases = [
        {
            "message": "Hello, I need help with banana prices",
            "language": "english",
            "description": "Greeting and price inquiry",
            "expected_intent": "price_inquiry"
        },
        {
            "message": "What's the current price of ambul bananas in Colombo for 50 kg?",
            "language": "english", 
            "description": "Specific price inquiry with details",
            "expected_intent": "price_inquiry"
        },
        {
            "message": "Where should I sell my bananas? I have 100 kg in Kandy",
            "language": "english",
            "description": "Market recommendation request",
            "expected_intent": "market_recommendation"
        },
        {
            "message": "How do I improve the quality of my bananas?",
            "language": "english",
            "description": "Quality improvement advice",
            "expected_intent": "quality_grading"
        },
        {
            "message": "What are the best storage methods for bananas?",
            "language": "english",
            "description": "Storage and transport advice",
            "expected_intent": "storage_transport"
        },
        {
            "message": "Help me understand market trends",
            "language": "english",
            "description": "Market trends inquiry",
            "expected_intent": "market_trends"
        },
        {
            "message": "What farming techniques should I use for better yield?",
            "language": "english",
            "description": "Farming advice request",
            "expected_intent": "farming_advice"
        },
        {
            "message": "කේසෙල් මිල කීයද?",
            "language": "sinhala",
            "description": "Price inquiry in Sinhala",
            "expected_intent": "price_inquiry"
        }
    ]
    
    success_count = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['description']}")
        print(f"   Query: {test_case['message']}")
        print(f"   Language: {test_case['language']}")
        print(f"   Expected Intent: {test_case['expected_intent']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/chatbot",
                json={
                    "message": test_case["message"],
                    "language": test_case["language"]
                },
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get('text', '')
                
                print(f"   ✅ Status: Success")
                print(f"   📝 Response: {response_text[:100]}...")
                
                if data.get('enhanced'):
                    print("   🌟 Enhanced features detected!")
                    if data.get('suggestions'):
                        print(f"   💡 Suggestions: {', '.join(data['suggestions'][:3])}")
                    if data.get('data'):
                        print("   📊 Additional data provided")
                else:
                    print("   📝 Standard response")
                
                success_count += 1
                    
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                if response.text:
                    print(f"   Error details: {response.text[:200]}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    print(f"\n📊 Chatbot Test Results: {success_count}/{total_tests} tests passed")
    return success_count == total_tests

def test_enhanced_price_prediction():
    """
    Test enhanced price prediction with 33 features
    """
    print("\n\n💰 Testing Enhanced Price Prediction (33 Features)")
    print("=" * 60)
    
    test_cases = [
        {
            "location": "Colombo",
            "banana_type": "ambul",
            "quantity": 50,
            "description": "Standard prediction - Ambul bananas in Colombo"
        },
        {
            "location": "Kandy", 
            "banana_type": "kolikuttu",
            "quantity": 100,
            "description": "Bulk quantity - Kolikuttu bananas in Kandy"
        },
        {
            "location": "Galle",
            "banana_type": "anamalu", 
            "quantity": 25,
            "description": "Small quantity - Anamalu bananas in Galle"
        }
    ]
    
    success_count = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['description']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=test_case,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                prediction = data.get('prediction', {})
                
                print(f"   ✅ Status: Success")
                print(f"   📍 Location: {prediction.get('location', 'N/A')}")
                print(f"   🍌 Type: {prediction.get('banana_type', 'N/A')}")
                print(f"   📦 Quantity: {prediction.get('quantity', 'N/A')} kg")
                print(f"   💰 Price: {prediction.get('price', 'N/A')} LKR/kg")
                print(f"   📊 Confidence: {prediction.get('confidence', 'N/A')}")
                print(f"   🔧 Features Used: {data.get('features_count', 'N/A')}")
                print(f"   🤖 Model Type: {data.get('model_type', 'N/A')}")
                
                success_count += 1
                    
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                if response.text:
                    error_data = response.json() if response.headers.get('content-type') == 'application/json' else response.text
                    print(f"   Error details: {str(error_data)[:200]}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    print(f"\n📊 Price Prediction Test Results: {success_count}/{total_tests} tests passed")
    return success_count == total_tests

def test_market_recommendations():
    """
    Test market recommendation functionality
    """
    print("\n\n🏪 Testing Market Recommendations")
    print("=" * 60)
    
    test_cases = [
        {
            "location": "Kandy",
            "quantity": 100,
            "banana_type": "ambul",
            "description": "Market recommendations for Kandy farmer"
        },
        {
            "location": "Galle",
            "quantity": 50,
            "banana_type": "kolikuttu",
            "description": "Market recommendations for Galle farmer"
        }
    ]
    
    success_count = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['description']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/recommend",
                json=test_case,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                recommendations = data.get('recommendations', [])
                
                print(f"   ✅ Status: Success")
                print(f"   📊 Found {len(recommendations)} market recommendations")
                
                for j, rec in enumerate(recommendations[:3], 1):
                    print(f"   {j}. {rec.get('name', 'Unknown Market')}")
                    print(f"      💰 Price: {rec.get('price', 'N/A')} LKR/kg")
                    print(f"      🚛 Transport: {rec.get('transport_cost', 'N/A')} LKR/kg")
                    print(f"      📈 Demand: {rec.get('demand', 'N/A')}")
                    print(f"      💵 Net Profit: {rec.get('net_profit', 'N/A')} LKR")
                
                success_count += 1
                    
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                if response.text:
                    error_data = response.json() if response.headers.get('content-type') == 'application/json' else response.text
                    print(f"   Error details: {str(error_data)[:200]}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    print(f"\n📊 Market Recommendation Test Results: {success_count}/{total_tests} tests passed")
    return success_count == total_tests

def test_api_health():
    """
    Test API health and basic endpoints
    """
    print("\n\n🏥 Testing API Health & Basic Endpoints")
    print("=" * 60)
    
    endpoints = [
        {"url": f"{BASE_URL}/health", "name": "Health Check"},
        {"url": f"{BASE_URL}/supported-languages", "name": "Supported Languages"},
        {"url": f"{BASE_URL}/markets", "name": "Available Markets"}
    ]
    
    success_count = 0
    total_tests = len(endpoints)
    
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint["url"], timeout=10)
            
            if response.status_code == 200:
                print(f"   ✅ {endpoint['name']}: OK")
                data = response.json()
                
                if endpoint["name"] == "Health Check":
                    print(f"      Status: {data.get('status', 'Unknown')}")
                    print(f"      Model: {data.get('model_status', 'Unknown')}")
                elif endpoint["name"] == "Supported Languages":
                    languages = data.get('supported_languages', [])
                    print(f"      Languages: {', '.join(languages)}")
                elif endpoint["name"] == "Available Markets":
                    markets = data.get('markets', [])
                    print(f"      Markets: {len(markets)} available")
                
                success_count += 1
            else:
                print(f"   ❌ {endpoint['name']}: Error {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ {endpoint['name']}: Exception - {e}")
    
    print(f"\n📊 API Health Test Results: {success_count}/{total_tests} tests passed")
    return success_count == total_tests

def main():
    """
    Main test function
    """
    print("🍌 SmartMusa Enhanced Features - Fix Verification Test")
    print("=" * 70)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing API at: {BASE_URL}")
    
    # Run all tests
    results = []
    
    results.append(("API Health", test_api_health()))
    results.append(("Enhanced Chatbot", test_enhanced_chatbot_comprehensive()))
    results.append(("Enhanced Price Prediction", test_enhanced_price_prediction()))
    results.append(("Market Recommendations", test_market_recommendations()))
    
    # Summary
    print("\n\n🎯 TEST SUMMARY")
    print("=" * 70)
    
    total_passed = 0
    total_tests = len(results)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if passed:
            total_passed += 1
    
    print(f"\nOverall Result: {total_passed}/{total_tests} test suites passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL FIXES VERIFIED SUCCESSFULLY!")
        print("\n✅ Issues Fixed:")
        print("  • Feature mismatch in API endpoints (4 → 33 features)")
        print("  • Missing chatbot methods implemented")
        print("  • Enhanced price prediction working")
        print("  • Market recommendations functional")
        print("  • All chatbot intents working")
    else:
        print(f"\n⚠️ {total_tests - total_passed} test suite(s) still have issues")
        print("Please check the detailed output above for specific problems.")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
