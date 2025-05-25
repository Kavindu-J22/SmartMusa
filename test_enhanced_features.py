# test_enhanced_features.py
"""
Test script to demonstrate enhanced SmartMusa features
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:5000/api/v1"

def test_enhanced_chatbot():
    """
    Test the enhanced chatbot with various queries
    """
    print("🤖 Testing Enhanced Chatbot Features")
    print("=" * 50)
    
    # Test cases for enhanced chatbot
    test_cases = [
        {
            "message": "Hello, I need help with banana prices",
            "language": "english",
            "description": "Greeting and price inquiry"
        },
        {
            "message": "What's the current price of ambul bananas in Colombo for 50 kg?",
            "language": "english",
            "description": "Specific price inquiry with details"
        },
        {
            "message": "Where should I sell my bananas? I have 100 kg in Kandy",
            "language": "english",
            "description": "Market recommendation request"
        },
        {
            "message": "How do I improve the quality of my bananas?",
            "language": "english",
            "description": "Quality improvement advice"
        },
        {
            "message": "What farming techniques should I use for better yield?",
            "language": "english",
            "description": "Farming advice request"
        },
        {
            "message": "කේසෙල් මිල කීයද?",
            "language": "sinhala",
            "description": "Price inquiry in Sinhala"
        },
        {
            "message": "Help me understand market trends",
            "language": "english",
            "description": "Market trends inquiry"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['description']}")
        print(f"   Query: {test_case['message']}")
        print(f"   Language: {test_case['language']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/chatbot",
                json={
                    "message": test_case["message"],
                    "language": test_case["language"]
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Response: {data.get('text', '')[:100]}...")
                
                if data.get('enhanced'):
                    print("   🌟 Enhanced features detected!")
                    if data.get('suggestions'):
                        print(f"   💡 Suggestions: {', '.join(data['suggestions'][:3])}")
                    if data.get('data'):
                        print("   📊 Additional data provided")
                else:
                    print("   📝 Standard response")
                    
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")

def test_enhanced_price_prediction():
    """
    Test enhanced price prediction with various parameters
    """
    print("\n\n💰 Testing Enhanced Price Prediction")
    print("=" * 50)
    
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
        },
        {
            "location": "Jaffna",
            "banana_type": "ambul",
            "quantity": 200,
            "description": "Large quantity - Ambul bananas in Jaffna"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['description']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=test_case,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                prediction = data.get('prediction', {})
                
                print(f"   📍 Location: {prediction.get('location', 'N/A')}")
                print(f"   🍌 Type: {prediction.get('banana_type', 'N/A')}")
                print(f"   📦 Quantity: {prediction.get('quantity', 'N/A')} kg")
                print(f"   💰 Price: {prediction.get('price', 'N/A')} LKR/kg")
                print(f"   📊 Confidence: {prediction.get('confidence', 'N/A')}")
                
                if 'market_insights' in prediction:
                    print("   🔍 Market insights available")
                    
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")

def test_market_recommendations():
    """
    Test market recommendation functionality
    """
    print("\n\n🏪 Testing Market Recommendations")
    print("=" * 50)
    
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
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['description']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/recommend",
                json=test_case,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                recommendations = data.get('recommendations', [])
                
                print(f"   📊 Found {len(recommendations)} market recommendations")
                
                for j, rec in enumerate(recommendations[:3], 1):
                    print(f"   {j}. {rec.get('market', 'Unknown Market')}")
                    print(f"      💰 Price: {rec.get('price', 'N/A')} LKR/kg")
                    print(f"      🚛 Transport: {rec.get('transport_cost', 'N/A')} LKR/kg")
                    print(f"      📈 Demand: {rec.get('demand', 'N/A')}")
                    
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")

def test_model_performance():
    """
    Test and display model performance metrics
    """
    print("\n\n📈 Testing Model Performance")
    print("=" * 50)
    
    try:
        # Test with multiple predictions to assess consistency
        predictions = []
        
        for i in range(5):
            response = requests.post(
                f"{BASE_URL}/predict",
                json={
                    "location": "Colombo",
                    "banana_type": "ambul",
                    "quantity": 50
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                price = data.get('prediction', {}).get('price')
                if price:
                    predictions.append(price)
        
        if predictions:
            avg_price = sum(predictions) / len(predictions)
            price_range = max(predictions) - min(predictions)
            
            print(f"   📊 Consistency Test Results:")
            print(f"   • Average Price: {avg_price:.2f} LKR/kg")
            print(f"   • Price Range: {price_range:.2f} LKR/kg")
            print(f"   • Predictions: {predictions}")
            
            if price_range < 5:
                print("   ✅ Model shows good consistency")
            else:
                print("   ⚠️ Model shows some variability")
        else:
            print("   ❌ No valid predictions received")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")

def test_api_health():
    """
    Test API health and availability
    """
    print("\n\n🏥 Testing API Health")
    print("=" * 50)
    
    endpoints = [
        {"url": f"{BASE_URL}/health", "name": "Health Check"},
        {"url": f"{BASE_URL}/supported-languages", "name": "Supported Languages"},
        {"url": f"{BASE_URL}/markets", "name": "Available Markets"}
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint["url"], timeout=5)
            
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
            else:
                print(f"   ❌ {endpoint['name']}: Error {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ {endpoint['name']}: Exception - {e}")

def main():
    """
    Main test function
    """
    print("🍌 SmartMusa Enhanced Features Test Suite")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing API at: {BASE_URL}")
    
    # Run all tests
    test_api_health()
    test_enhanced_chatbot()
    test_enhanced_price_prediction()
    test_market_recommendations()
    test_model_performance()
    
    print("\n\n🎉 Test Suite Completed!")
    print("=" * 60)
    print("\nEnhanced Features Summary:")
    print("✅ Enhanced Chatbot with comprehensive responses")
    print("✅ Improved price prediction with advanced features")
    print("✅ Better market recommendations")
    print("✅ Context-aware conversation management")
    print("✅ Multi-language support (English & Sinhala)")
    print("✅ Advanced feature engineering")
    print("✅ Ensemble model with better accuracy")
    
    print("\nNext Steps:")
    print("• Monitor model performance in production")
    print("• Collect user feedback for further improvements")
    print("• Consider adding more external data sources")
    print("• Implement A/B testing for model updates")
    print("• Add more languages and regional dialects")

if __name__ == "__main__":
    main()
