# enhanced_chatbot.py
import re
import json
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
import random

class EnhancedBananaChatbot:
    """
    Enhanced chatbot with improved NLP, context management, and comprehensive responses
    """

    def __init__(self, model_path='banana_price_model.pkl'):
        """
        Initialize the enhanced chatbot
        """
        self.model = None
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}. Error: {e}")

        # Enhanced intent patterns with more variations
        self.intents = {
            'greeting': [
                r'hello', r'hi', r'hey', r'good morning', r'good afternoon', r'good evening',
                r'ආයුබෝවන්', r'හෙලෝ', r'හායි', r'සුභ උදෑසනක්', r'සුභ සන්ධ්‍යාවක්'
            ],
            'price_inquiry': [
                r'price', r'how much', r'market price', r'banana price', r'current price',
                r'cost', r'rate', r'value', r'what.*price', r'price.*today',
                r'මිල', r'කීයද', r'වෙළඳපොල මිල', r'කේසෙල් මිල', r'අද මිල'
            ],
            'market_recommendation': [
                r'where to sell', r'best market', r'recommend market', r'where should i sell',
                r'which market', r'selling location', r'market advice', r'where.*sell',
                r'විකුණන්නේ කොහේද', r'හොඳම වෙළඳපොල', r'වෙළඳපොල නිර්දේශ', r'කොහේද විකුණන්නේ'
            ],
            'weather_inquiry': [
                r'weather', r'rain', r'forecast', r'climate', r'temperature', r'humidity',
                r'කාලගුණය', r'වැස්ස', r'අනාවැකිය', r'උෂ්ණත්වය'
            ],
            'farming_advice': [
                r'how to grow', r'fertilizer', r'disease', r'pest', r'cultivation', r'planting',
                r'harvest', r'irrigation', r'soil', r'nutrients', r'organic farming',
                r'වගා කරන්නේ කෙසේද', r'පොහොර', r'රෝග', r'පළිබෝධ', r'වගාව', r'වගා කිරීම'
            ],
            'quality_grading': [
                r'quality', r'grade', r'grading', r'classification', r'standard',
                r'export quality', r'premium', r'first grade', r'second grade',
                r'ගුණත්වය', r'ශ්‍රේණිගත කිරීම', r'තත්ත්වය'
            ],
            'storage_transport': [
                r'storage', r'transport', r'packaging', r'preservation', r'shelf life',
                r'cold storage', r'ripening', r'handling',
                r'ගබඩා කිරීම', r'ප්‍රවාහනය', r'ඇසුරුම්'
            ],
            'market_trends': [
                r'trend', r'market trend', r'demand', r'supply', r'seasonal', r'forecast',
                r'future price', r'market analysis', r'price prediction',
                r'වෙළඳපොල ප්‍රවණතා', r'ඉල්ලුම', r'සැපයුම'
            ],
            'help': [
                r'help', r'guide', r'assistant', r'what can you do', r'commands', r'options',
                r'උදව්', r'මාර්ගෝපදේශය', r'සහායක', r'ඔබට කුමක් කළ හැකිද'
            ],
            'goodbye': [
                r'bye', r'goodbye', r'see you', r'thanks', r'thank you', r'exit', r'quit',
                r'ගිහින් එන්නම්', r'ස්තූතියි', r'ආයුබෝවන්'
            ]
        }

        # Enhanced responses with more comprehensive information
        self.responses = {
            'english': {
                'greeting': [
                    "Hello! I'm SmartMusa, your advanced banana farming assistant. I can help you with prices, market recommendations, farming advice, quality grading, and much more. How can I assist you today?",
                    "Welcome to SmartMusa! I'm here to provide comprehensive support for banana farmers. Ask me about current prices, best markets to sell, farming techniques, or any other banana-related questions."
                ],
                'price_inquiry': [
                    "I can provide current banana price predictions based on location, season, and market conditions. Please tell me your location and the type/quantity of bananas you have.",
                    "For accurate price information, I need details about your location, banana variety (Ambul, Kolikuttu, Anamalu, etc.), and quantity. I'll also consider current market trends."
                ],
                'market_recommendation': [
                    "I can recommend the best markets based on current prices, transportation costs, and demand. Please share your location, quantity, and preferred selling timeframe.",
                    "For optimal market recommendations, I'll analyze current prices across different markets, calculate transportation costs, and consider seasonal demand patterns. What's your location and harvest details?"
                ],
                'weather_inquiry': [
                    "Weather significantly impacts banana farming and prices. I can provide weather-related farming advice and its impact on market conditions. Which area are you interested in?",
                    "I can help you understand how weather patterns affect banana cultivation, harvesting schedules, and market prices. What specific weather information do you need?"
                ],
                'farming_advice': [
                    "I offer comprehensive farming guidance including: planting techniques, fertilizer recommendations, pest management, disease control, irrigation, and harvesting best practices. What specific area interests you?",
                    "My farming advice covers the complete banana cultivation cycle: soil preparation, planting, nutrition management, pest control, disease prevention, and post-harvest handling. What would you like to know?"
                ],
                'quality_grading': [
                    "Banana quality grading is crucial for getting the best prices. I can explain grading standards, quality parameters, and how to improve your banana quality for premium markets.",
                    "Quality determines price! I can guide you on: size classification, ripeness stages, defect identification, export standards, and packaging requirements for different market segments."
                ],
                'storage_transport': [
                    "Proper storage and transportation are vital for maintaining quality and reducing losses. I can advise on: cold storage, packaging methods, transportation best practices, and shelf-life optimization.",
                    "Post-harvest handling significantly affects your profits. I can help with: storage techniques, ripening control, packaging standards, and transportation logistics to minimize losses."
                ],
                'market_trends': [
                    "I analyze market trends including seasonal patterns, demand fluctuations, price forecasts, and supply chain dynamics. I can help you plan your farming and selling strategies accordingly.",
                    "Understanding market trends helps maximize profits. I provide insights on: seasonal price patterns, demand forecasting, supply analysis, and optimal selling timing strategies."
                ],
                'help': [
                    "I'm your comprehensive banana farming assistant! I can help with:\n• Price predictions and market analysis\n• Market recommendations and selling strategies\n• Complete farming guidance (planting to harvest)\n• Quality grading and standards\n• Storage and transportation advice\n• Weather impact analysis\n• Market trend insights\n\nWhat would you like to explore?",
                    "SmartMusa offers complete banana farming support:\n🍌 Real-time price predictions\n📊 Market analysis and recommendations\n🌱 Farming techniques and best practices\n🏆 Quality improvement guidance\n📦 Post-harvest handling advice\n🌤️ Weather-based farming tips\n📈 Market trend analysis\n\nHow can I help you succeed?"
                ],
                'goodbye': [
                    "Thank you for using SmartMusa! Remember, successful banana farming combines good agricultural practices with smart market decisions. Feel free to return anytime for guidance!",
                    "Goodbye! Keep growing and stay profitable! I'm here 24/7 whenever you need expert advice on banana farming, pricing, or market strategies."
                ],
                'default': [
                    "I didn't quite understand that. I specialize in banana farming, pricing, market analysis, and agricultural guidance. Could you please rephrase your question or ask about:\n• Current banana prices\n• Market recommendations\n• Farming techniques\n• Quality standards\n• Storage/transport advice",
                    "I'm not sure about that specific topic. As your banana farming expert, I can help with pricing, markets, cultivation, quality, and post-harvest handling. What banana-related question can I answer for you?"
                ]
            },
            'sinhala': {
                'greeting': [
                    "ආයුබෝවන්! මම SmartMusa, ඔබේ දියුණු කේසෙල් වගා සහායකයා. මට මිල, වෙළඳපොල නිර්දේශ, වගා උපදෙස්, ගුණත්ව ශ්‍රේණිගත කිරීම සහ තවත් බොහෝ දේ සමඟ උදව් කළ හැකිය. අද මට ඔබට කෙසේ සහාය විය හැකිද?",
                    "SmartMusa වෙත සාදරයෙන් පිළිගනිමු! මම කේසෙල් වගාකරුවන්ට සම්පූර්ණ සහාය ලබා දීමට මෙහි සිටිමි. වර්තමාන මිල, විකිණීමට හොඳම වෙළඳපොල, වගා ක්‍රම, හෝ වෙනත් කේසෙල් සම්බන්ධ ප්‍රශ්න ගැන මගෙන් අසන්න."
                ],
                'price_inquiry': [
                    "මට ස්ථානය, කන්නය සහ වෙළඳපොල තත්ත්වයන් මත පදනම්ව වර්තමාන කේසෙල් මිල අනාවැකි ලබා දිය හැකිය. කරුණාකර ඔබේ ස්ථානය සහ ඔබට ඇති කේසෙල් වර්ගය/ප්‍රමාණය මට කියන්න.",
                    "නිවැරදි මිල තොරතුරු සඳහා, මට ඔබේ ස්ථානය, කේසෙල් වර්ගය (අම්බුල්, කොළිකුට්ටු, අනමාලු, ආදිය), සහ ප්‍රමාණය පිළිබඳ විස්තර අවශ්‍යයි. මම වර්තමාන වෙළඳපොල ප්‍රවණතා ද සලකා බලමි."
                ],
                'market_recommendation': [
                    "මට වර්තමාන මිල, ප්‍රවාහන වියදම් සහ ඉල්ලුම මත පදනම්ව හොඳම වෙළඳපොල නිර්දේශ කළ හැකිය. කරුණාකර ඔබේ ස්ථානය, ප්‍රමාණය සහ කැමති විකිණීමේ කාල සීමාව බෙදා ගන්න.",
                    "ප්‍රශස්ත වෙළඳපොල නිර්දේශ සඳහා, මම විවිධ වෙළඳපොලවල වර්තමාන මිල විශ්ලේෂණය කරමි, ප්‍රවාහන වියදම් ගණනය කරමි, සහ කන්නාකාර ඉල්ලුම් රටා සලකා බලමි. ඔබේ ස්ථානය සහ අස්වනු විස්තර කුමක්ද?"
                ],
                'help': [
                    "මම ඔබේ සම්පූර්ණ කේසෙල් වගා සහායකයා! මට උදව් කළ හැකිය:\n• මිල අනාවැකි සහ වෙළඳපොල විශ්ලේෂණය\n• වෙළඳපොල නිර්දේශ සහ විකිණීමේ උපාය මාර්ග\n• සම්පූර්ණ වගා මාර්ගෝපදේශනය (රෝපණයේ සිට අස්වනු දක්වා)\n• ගුණත්ව ශ්‍රේණිගත කිරීම සහ ප්‍රමිතීන්\n• ගබඩා සහ ප්‍රවාහන උපදෙස්\n• කාලගුණ බලපෑම් විශ්ලේෂණය\n• වෙළඳපොල ප්‍රවණතා අවබෝධය\n\nඔබ කුමක් ගවේෂණය කිරීමට කැමතිද?",
                    "SmartMusa සම්පූර්ණ කේසෙල් වගා සහාය ලබා දෙයි:\n🍌 තත්‍ය කාලීන මිල අනාවැකි\n📊 වෙළඳපොල විශ්ලේෂණය සහ නිර්දේශ\n🌱 වගා ක්‍රම සහ හොඳම පිළිවෙත්\n🏆 ගුණත්ව වැඩිදියුණු කිරීමේ මාර්ගෝපදේශනය\n📦 අස්වනු නෙලීමෙන් පසු හැසිරවීමේ උපදෙස්\n🌤️ කාලගුණය මත පදනම් වූ වගා ඉඟි\n📈 වෙළඳපොල ප්‍රවණතා විශ්ලේෂණය\n\nමට ඔබට සාර්ථක වීමට කෙසේ උදව් කළ හැකිද?"
                ],
                'default': [
                    "මට එය හරියටම තේරුම් ගත නොහැකි විය. මම කේසෙල් වගාව, මිල ගණන්, වෙළඳපොල විශ්ලේෂණය සහ කෘෂිකාර්මික මාර්ගෝපදේශනය පිළිබඳ විශේෂඥයෙක්. කරුණාකර ඔබේ ප්‍රශ්නය නැවත සඳහන් කරන්න හෝ මේ ගැන අසන්න:\n• වර්තමාන කේසෙල් මිල\n• වෙළඳපොල නිර්දේශ\n• වගා ක්‍රම\n• ගුණත්ව ප්‍රමිතීන්\n• ගබඩා/ප්‍රවාහන උපදෙස්",
                    "මට එම නිශ්චිත මාතෘකාව ගැන විශ්වාස නැත. ඔබේ කේසෙල් වගා විශේෂඥයා ලෙස, මට මිල ගණන්, වෙළඳපොල, වගාව, ගුණත්වය සහ අස්වනු නෙලීමෙන් පසු හැසිරවීම සමඟ උදව් කළ හැකිය. මට ඔබ වෙනුවෙන් පිළිතුරු දිය හැකි කේසෙල් සම්බන්ධ ප්‍රශ්නයක් කුමක්ද?"
                ]
            }
        }

        # Enhanced context management
        self.context = {
            'user_location': None,
            'banana_type': 'ambul',
            'quantity': None,
            'conversation_history': [],
            'user_preferences': {},
            'last_intent': None,
            'session_start': datetime.now()
        }

        # Banana varieties and their characteristics
        self.banana_varieties = {
            'ambul': {'avg_price': 120, 'season': 'year-round', 'quality': 'premium'},
            'kolikuttu': {'avg_price': 100, 'season': 'year-round', 'quality': 'good'},
            'anamalu': {'avg_price': 90, 'season': 'seasonal', 'quality': 'standard'},
            'seeni': {'avg_price': 150, 'season': 'seasonal', 'quality': 'premium'},
            'rathkesel': {'avg_price': 80, 'season': 'year-round', 'quality': 'standard'}
        }

        # Market information with detailed data
        self.markets = {
            'colombo': {'base_price': 120, 'demand': 'high', 'transport_cost': 5},
            'kandy': {'base_price': 110, 'demand': 'medium', 'transport_cost': 8},
            'galle': {'base_price': 115, 'demand': 'medium', 'transport_cost': 10},
            'jaffna': {'base_price': 125, 'demand': 'high', 'transport_cost': 15},
            'anuradhapura': {'base_price': 105, 'demand': 'low', 'transport_cost': 12},
            'dambulla': {'base_price': 100, 'demand': 'high', 'transport_cost': 6}
        }

    def detect_intent(self, message, language='english'):
        """Enhanced intent detection with confidence scoring"""
        message = message.lower()
        intent_scores = {}

        for intent, patterns in self.intents.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    score += 1
            if score > 0:
                intent_scores[intent] = score

        if intent_scores:
            return max(intent_scores.keys(), key=lambda x: intent_scores[x])
        return 'default'

    def extract_entities(self, message):
        """Enhanced entity extraction"""
        entities = {}

        # Location patterns
        location_patterns = [
            r'(?:in|at|near|from)\s+([a-zA-Z\s]+?)(?:\s|$|,|\.|!|\?)',
            r'([a-zA-Z]+)\s+(?:market|area|district)'
        ]

        for pattern in location_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                entities['location'] = match.group(1).strip().lower()
                break

        # Quantity patterns
        quantity_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:kg|kilos?)',
            r'(\d+)\s*(?:bunches?|hands?)'
        ]

        for pattern in quantity_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                try:
                    entities['quantity'] = float(match.group(1))
                    break
                except ValueError:
                    continue

        # Banana type
        for banana_type in self.banana_varieties.keys():
            if banana_type in message.lower():
                entities['banana_type'] = banana_type
                break

        return entities

    def update_context(self, entities, intent):
        """Update conversation context"""
        if 'location' in entities:
            self.context['user_location'] = entities['location']
        if 'banana_type' in entities:
            self.context['banana_type'] = entities['banana_type']
        if 'quantity' in entities:
            self.context['quantity'] = entities['quantity']

        self.context['last_intent'] = intent
        self.context['conversation_history'].append({
            'timestamp': datetime.now(),
            'intent': intent,
            'entities': entities
        })

        # Keep only last 10 interactions
        if len(self.context['conversation_history']) > 10:
            self.context['conversation_history'] = self.context['conversation_history'][-10:]

    def predict_price_enhanced(self, location=None, banana_type=None, quantity=None):
        """Enhanced price prediction"""
        location = location or self.context.get('user_location', 'colombo')
        banana_type = banana_type or self.context.get('banana_type', 'ambul')
        quantity = quantity or self.context.get('quantity', 1)

        try:
            if self.model is None:
                return self._fallback_price_prediction(location, banana_type, quantity)

            # Prepare features for model
            current_date = datetime.now()
            features = [
                current_date.month,
                (current_date.day - 1) // 7 + 1,
                current_date.weekday(),
                self._get_location_code(location)
            ]

            predicted_price = float(self.model.predict([features])[0])

            # Quantity adjustments
            if quantity and quantity > 100:
                predicted_price *= 0.95

            return {
                "price": round(predicted_price, 2),
                "currency": "LKR",
                "banana_type": banana_type,
                "location": location,
                "quantity": quantity,
                "date": current_date.strftime("%Y-%m-%d"),
                "confidence": "high"
            }

        except Exception as e:
            return self._fallback_price_prediction(location, banana_type, quantity)

    def _fallback_price_prediction(self, location, banana_type, quantity):
        """Fallback price prediction"""
        base_price = self.banana_varieties.get(banana_type, {}).get('avg_price', 100)
        market_data = self.markets.get(location, {'base_price': 100})

        price_adjustment = (market_data['base_price'] - 100) / 100
        adjusted_price = base_price * (1 + price_adjustment)

        # Seasonal adjustments
        current_month = datetime.now().month
        if current_month in [12, 1, 4]:  # Festival seasons
            adjusted_price *= 1.1

        # Quantity adjustments
        if quantity and quantity > 100:
            adjusted_price *= 0.95

        return {
            "price": round(adjusted_price, 2),
            "currency": "LKR",
            "banana_type": banana_type,
            "location": location,
            "quantity": quantity or 1,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "confidence": "medium"
        }

    def _get_location_code(self, location):
        """Get numeric code for location"""
        location_mapping = {
            'colombo': 1, 'kandy': 2, 'galle': 3, 'jaffna': 4,
            'anuradhapura': 5, 'dambulla': 6
        }
        return location_mapping.get(location.lower(), 1)

    def get_comprehensive_response(self, message, language='english'):
        """Generate comprehensive response with enhanced features"""
        # Detect intent and extract entities
        intent = self.detect_intent(message, language)
        entities = self.extract_entities(message)

        # Update context
        self.update_context(entities, intent)

        # Generate response based on intent
        if intent == 'price_inquiry':
            return self._handle_price_inquiry(language)
        elif intent == 'market_recommendation':
            return self._handle_market_recommendation(language)
        elif intent == 'farming_advice':
            return self._handle_farming_advice(language)
        elif intent == 'quality_grading':
            return self._handle_quality_grading(language)
        elif intent == 'storage_transport':
            return self._handle_storage_transport(language)
        elif intent == 'market_trends':
            return self._handle_market_trends(language)
        else:
            # Default response
            response_text = random.choice(self.responses[language].get(intent, self.responses[language]['default']))
            return {
                "text": response_text,
                "data": None,
                "suggestions": self._get_suggestions(intent, language)
            }

    def _handle_price_inquiry(self, language):
        """Handle price inquiry with comprehensive information"""
        if self.context.get('user_location'):
            price_info = self.predict_price_enhanced()

            if language == 'english':
                response_text = f"""Current price prediction for {price_info['banana_type']} bananas in {price_info['location']}:

💰 **Price**: {price_info['price']} {price_info['currency']} per kg
📅 **Date**: {price_info['date']}
📊 **Confidence**: {price_info['confidence']}
📦 **Quantity**: {price_info['quantity']} kg

💡 **Tips**:
• Prices may vary based on quality and market conditions
• Consider selling in bulk for better rates
• Check multiple markets for best prices"""
            else:
                response_text = f"""{price_info['location']} හි {price_info['banana_type']} කේසෙල් සඳහා වර්තමාන මිල අනාවැකිය:

💰 **මිල**: කිලෝ එකකට {price_info['price']} {price_info['currency']}
📅 **දිනය**: {price_info['date']}
📊 **විශ්වාසනීයත්වය**: {price_info['confidence']}
📦 **ප්‍රමාණය**: කිලෝ {price_info['quantity']}

💡 **උපදෙස්**:
• ගුණත්වය සහ වෙළඳපොල තත්ත්වයන් අනුව මිල වෙනස් විය හැක
• හොඳ මිලකට තොග විකිණීම සලකා බලන්න
• හොඳම මිල සඳහා වෙළඳපොල කිහිපයක් පරීක්ෂා කරන්න"""

            return {
                "text": response_text,
                "data": price_info,
                "suggestions": ["Market recommendations", "Quality tips", "Storage advice"]
            }
        else:
            return {
                "text": self.responses[language]['price_inquiry'][0],
                "data": None,
                "suggestions": ["Tell me your location", "Specify banana type", "Mention quantity"]
            }

    def _handle_market_recommendation(self, language):
        """Handle market recommendation requests"""
        if self.context.get('user_location') and self.context.get('quantity'):
            recommendations = self._get_market_recommendations()

            if language == 'english':
                best_market = recommendations[0]
                response_text = f"""🏆 **Best Market Recommendation**:

📍 **Market**: {best_market['name']}
💰 **Expected Price**: {best_market['price']} LKR/kg
🚛 **Transport Cost**: {best_market['transport_cost']} LKR/kg
💵 **Net Profit**: {best_market['net_profit']} LKR
📊 **Demand**: {best_market['demand']}

📋 **Alternative Markets**:"""

                for i, market in enumerate(recommendations[1:3], 1):
                    response_text += f"\n{i}. {market['name']} - {market['price']} LKR/kg"
            else:
                best_market = recommendations[0]
                response_text = f"""🏆 **හොඳම වෙළඳපොල නිර්දේශය**:

📍 **වෙළඳපොල**: {best_market['name']}
💰 **අපේක්ෂිත මිල**: කිලෝ එකකට {best_market['price']} රුපියල්
🚛 **ප්‍රවාහන වියදම**: කිලෝ එකකට {best_market['transport_cost']} රුපියල්
💵 **ශුද්ධ ලාභය**: {best_market['net_profit']} රුපියල්
📊 **ඉල්ලුම**: {best_market['demand']}

📋 **විකල්ප වෙළඳපොල**:"""

                for i, market in enumerate(recommendations[1:3], 1):
                    response_text += f"\n{i}. {market['name']} - කිලෝ එකකට {market['price']} රුපියල්"

            return {
                "text": response_text,
                "data": recommendations,
                "suggestions": ["Price trends", "Transport options", "Quality requirements"]
            }
        else:
            return {
                "text": self.responses[language]['market_recommendation'][0],
                "data": None,
                "suggestions": ["Tell me your location", "Specify quantity", "Mention timeframe"]
            }

    def _handle_farming_advice(self, language):
        """Handle farming advice requests"""
        advice_topics = {
            'english': {
                'planting': "🌱 **Planting Tips**: Choose healthy suckers, plant in well-drained soil, maintain 2-3m spacing",
                'fertilizer': "🌿 **Fertilizer Guide**: Use NPK 14:14:14 monthly, add organic compost, ensure proper drainage",
                'pest_control': "🐛 **Pest Management**: Regular inspection, neem oil spray, remove affected leaves promptly",
                'irrigation': "💧 **Irrigation**: Deep watering 2-3 times weekly, avoid waterlogging, mulch around plants"
            },
            'sinhala': {
                'planting': "🌱 **රෝපණ උපදෙස්**: සෞඛ්‍ය සම්පන්න පැල තෝරන්න, ජලය බැස යන පස්වල රෝපණය කරන්න, මීටර් 2-3 ක් පරතරය තබන්න",
                'fertilizer': "🌿 **පොහොර මාර්ගෝපදේශය**: මාසිකව NPK 14:14:14 භාවිතා කරන්න, කාබනික කොම්පෝස්ට් එකතු කරන්න",
                'pest_control': "🐛 **පළිබෝධ කළමනාකරණය**: නිතිපතා පරීක්ෂා කරන්න, නීම් තෙල් ඉසින්න, බලපෑමට ලක් වූ කොළ ඉවත් කරන්න",
                'irrigation': "💧 **ජලාභිවර්ධනය**: සතියකට 2-3 වතාවක් ගැඹුරින් ජලය දෙන්න, ජලය රැඳී සිටීම වළක්වන්න"
            }
        }

        advice_text = "\n".join(advice_topics[language].values())

        return {
            "text": advice_text,
            "data": advice_topics[language],
            "suggestions": ["Specific pest problems", "Soil testing", "Harvest timing"]
        }

    def _get_market_recommendations(self):
        """Get market recommendations based on context"""
        user_location = self.context.get('user_location', 'colombo')
        quantity = self.context.get('quantity', 1)
        banana_type = self.context.get('banana_type', 'ambul')

        recommendations = []

        for market_name, market_data in self.markets.items():
            price_info = self.predict_price_enhanced(market_name, banana_type, quantity)
            transport_cost = market_data.get('transport_cost', 5)

            net_profit = (price_info['price'] - transport_cost) * quantity

            recommendations.append({
                'name': f"{market_name.title()} Market",
                'price': price_info['price'],
                'transport_cost': transport_cost,
                'net_profit': round(net_profit, 2),
                'demand': market_data.get('demand', 'medium')
            })

        # Sort by net profit
        recommendations.sort(key=lambda x: x['net_profit'], reverse=True)
        return recommendations

    def _get_suggestions(self, intent, language):
        """Get contextual suggestions"""
        suggestions = {
            'english': {
                'default': ["Check prices", "Market recommendations", "Farming tips", "Quality standards"],
                'greeting': ["Ask about prices", "Get market advice", "Learn farming techniques"],
                'help': ["Price inquiry", "Market analysis", "Farming guidance", "Quality tips"]
            },
            'sinhala': {
                'default': ["මිල පරීක්ෂා කරන්න", "වෙළඳපොල නිර්දේශ", "වගා උපදෙස්", "ගුණත්ව ප්‍රමිතීන්"],
                'greeting': ["මිල ගැන අසන්න", "වෙළඳපොල උපදෙස් ලබා ගන්න", "වගා ක්‍රම ඉගෙන ගන්න"],
                'help': ["මිල විමසීම", "වෙළඳපොල විශ්ලේෂණය", "වගා මාර්ගෝපදේශනය", "ගුණත්ව උපදෙස්"]
            }
        }

        return suggestions[language].get(intent, suggestions[language]['default'])

    def get_market_recommendation(self, location, quantity, banana_type):
        """
        Get market recommendations (for compatibility with existing API)
        """
        try:
            # Update context
            self.context['user_location'] = location
            self.context['quantity'] = quantity
            self.context['banana_type'] = banana_type

            # Get recommendations
            recommendations = self._get_market_recommendations()

            return {
                "recommendations": recommendations,
                "farmer_location": location,
                "quantity": quantity,
                "banana_type": banana_type,
                "status": "success"
            }

        except Exception as e:
            return {
                "error": f"Failed to generate recommendations: {str(e)}",
                "status": "error"
            }

    def _handle_quality_grading(self, language):
        """Handle quality grading and improvement requests"""
        quality_info = {
            'english': {
                'title': "🏆 **Banana Quality Grading & Improvement Guide**",
                'grading_standards': {
                    'Premium Grade (A)': "• Length: 15-20cm\n• Uniform yellow color\n• No blemishes or bruises\n• Firm texture\n• Sweet aroma",
                    'Standard Grade (B)': "• Length: 12-18cm\n• Slight color variations\n• Minor surface marks\n• Good texture\n• Acceptable aroma",
                    'Commercial Grade (C)': "• Length: 10-15cm\n• Color variations\n• Some surface defects\n• Softer texture\n• Processing suitable"
                },
                'improvement_tips': [
                    "🌱 **Proper Harvesting**: Harvest at 75-80% maturity for best quality",
                    "📦 **Careful Handling**: Use padded containers to prevent bruising",
                    "🌡️ **Temperature Control**: Store at 13-15°C for optimal ripening",
                    "💧 **Humidity Management**: Maintain 85-90% relative humidity",
                    "🧼 **Cleanliness**: Wash and sanitize before packaging",
                    "📏 **Size Sorting**: Group similar sizes for uniform presentation",
                    "⏰ **Timing**: Sell within 3-5 days of optimal ripeness"
                ],
                'market_premiums': {
                    'Premium Grade': "20-30% price premium",
                    'Export Quality': "40-50% price premium",
                    'Organic Certified': "25-35% price premium"
                }
            },
            'sinhala': {
                'title': "🏆 **කේසෙල් ගුණත්ව ශ්‍රේණිගත කිරීම සහ වැඩිදියුණු කිරීමේ මාර්ගෝපදේශය**",
                'grading_standards': {
                    'ප්‍රිමියම් ශ්‍රේණිය (A)': "• දිග: සෙ.මී. 15-20\n• ඒකාකාර කහ පාට\n• කැළැල් හෝ තැලීම් නැත\n• ශක්තිමත් ගතිය\n• මිහිරි සුවඳ",
                    'සම්මත ශ්‍රේණිය (B)': "• දිග: සෙ.මී. 12-18\n• සුළු වර්ණ වෙනස්කම්\n• සුළු මතුපිට සලකුණු\n• හොඳ ගතිය\n• පිළිගත හැකි සුවඳ",
                    'වාණිජ ශ්‍රේණිය (C)': "• දිග: සෙ.මී. 10-15\n• වර්ණ වෙනස්කම්\n• සමහර මතුපිට දෝෂ\n• මෘදු ගතිය\n• සැකසීම සඳහා සුදුසු"
                },
                'improvement_tips': [
                    "🌱 **නිසි අස්වනු නෙලීම**: හොඳම ගුණත්වය සඳහා 75-80% පරිණතභාවයේදී අස්වනු නෙලන්න",
                    "📦 **ප්‍රවේශමෙන් හැසිරවීම**: තැලීම් වැළැක්වීම සඳහා ගෙඩි සහිත බහාලුම් භාවිතා කරන්න",
                    "🌡️ **උෂ්ණත්ව පාලනය**: ප්‍රශස්ත ඉදීම සඳහා 13-15°C හි ගබඩා කරන්න",
                    "💧 **තෙතමනය කළමනාකරණය**: 85-90% සාපේක්ෂ තෙතමනය පවත්වන්න",
                    "🧼 **පිරිසිදුකම**: ඇසුරුම් කිරීමට පෙර සෝදා සනීපාරක්ෂක කරන්න"
                ]
            }
        }

        info = quality_info[language]

        response_text = f"{info['title']}\n\n"
        response_text += "📊 **ශ්‍රේණිගත කිරීමේ ප්‍රමිතීන්**:\n" if language == 'sinhala' else "📊 **Grading Standards**:\n"

        for grade, criteria in info['grading_standards'].items():
            response_text += f"\n**{grade}**:\n{criteria}\n"

        response_text += "\n💡 **වැඩිදියුණු කිරීමේ උපදෙස්**:\n" if language == 'sinhala' else "\n💡 **Improvement Tips**:\n"
        for tip in info['improvement_tips']:
            response_text += f"{tip}\n"

        if language == 'english' and 'market_premiums' in info:
            response_text += "\n💰 **Market Premiums**:\n"
            for grade, premium in info['market_premiums'].items():
                response_text += f"• {grade}: {premium}\n"

        return {
            "text": response_text,
            "data": info,
            "suggestions": ["Storage advice", "Market prices", "Harvesting tips"]
        }

    def _handle_storage_transport(self, language):
        """Handle storage and transportation advice"""
        storage_info = {
            'english': {
                'title': "📦 **Storage & Transportation Best Practices**",
                'storage_guidelines': {
                    'Pre-Ripening Storage': {
                        'temperature': '13-15°C (55-59°F)',
                        'humidity': '85-90%',
                        'duration': '7-14 days',
                        'tips': ['Avoid direct sunlight', 'Ensure good ventilation', 'Check daily for ripening']
                    },
                    'Ripening Storage': {
                        'temperature': '18-20°C (64-68°F)',
                        'humidity': '85-90%',
                        'duration': '3-5 days',
                        'tips': ['Use ethylene gas if needed', 'Monitor color changes', 'Separate by ripeness']
                    },
                    'Cold Storage': {
                        'temperature': '12-14°C (54-57°F)',
                        'humidity': '85-90%',
                        'duration': '2-4 weeks',
                        'tips': ['Gradual temperature changes', 'Proper air circulation', 'Regular quality checks']
                    }
                },
                'transportation_tips': [
                    "🚛 **Vehicle Preparation**: Clean and sanitize transport vehicles",
                    "📦 **Packaging**: Use ventilated boxes with cushioning material",
                    "🌡️ **Temperature Control**: Maintain cool temperatures during transport",
                    "⏰ **Timing**: Transport during cooler parts of the day",
                    "📍 **Route Planning**: Choose shortest routes to minimize handling time",
                    "🔄 **Loading**: Load carefully to prevent crushing and bruising",
                    "📋 **Documentation**: Maintain proper records for traceability"
                ]
            },
            'sinhala': {
                'title': "📦 **ගබඩා කිරීම සහ ප්‍රවාහනයේ හොඳම පිළිවෙත්**",
                'storage_guidelines': {
                    'ඉදීමට පෙර ගබඩාව': {
                        'temperature': '13-15°C (55-59°F)',
                        'humidity': '85-90%',
                        'duration': 'දින 7-14',
                        'tips': ['සෘජු හිරු එළිය වළක්වන්න', 'හොඳ වාතාශ්‍රය සහතික කරන්න', 'ඉදීම සඳහා දිනපතා පරීක්ෂා කරන්න']
                    },
                    'ඉදීමේ ගබඩාව': {
                        'temperature': '18-20°C (64-68°F)',
                        'humidity': '85-90%',
                        'duration': 'දින 3-5',
                        'tips': ['අවශ්‍ය නම් එතිලීන් වායුව භාවිතා කරන්න', 'වර්ණ වෙනස්කම් නිරීක්ෂණය කරන්න', 'ඉදීම අනුව වෙන් කරන්න']
                    }
                },
                'transportation_tips': [
                    "🚛 **වාහන සූදානම**: ප්‍රවාහන වාහන පිරිසිදු කර සනීපාරක්ෂක කරන්න",
                    "📦 **ඇසුරුම්**: කුෂන් ද්‍රව්‍ය සහිත වාතාශ්‍රිත පෙට්ටි භාවිතා කරන්න",
                    "🌡️ **උෂ්ණත්ව පාලනය**: ප්‍රවාහනය අතරතුර සිසිල් උෂ්ණත්වයන් පවත්වන්න",
                    "⏰ **කාලසටහන**: දිනයේ සිසිල් කාලවලදී ප්‍රවාහනය කරන්න"
                ]
            }
        }

        info = storage_info[language]

        response_text = f"{info['title']}\n\n"

        if 'storage_guidelines' in info:
            response_text += "🏪 **ගබඩා මාර්ගෝපදේශ**:\n\n" if language == 'sinhala' else "🏪 **Storage Guidelines**:\n\n"

            for storage_type, details in info['storage_guidelines'].items():
                response_text += f"**{storage_type}**:\n"
                response_text += f"• උෂ්ණත්වය: {details['temperature']}\n" if language == 'sinhala' else f"• Temperature: {details['temperature']}\n"
                response_text += f"• තෙතමනය: {details['humidity']}\n" if language == 'sinhala' else f"• Humidity: {details['humidity']}\n"
                response_text += f"• කාලසීමාව: {details['duration']}\n" if language == 'sinhala' else f"• Duration: {details['duration']}\n"

                for tip in details['tips']:
                    response_text += f"  - {tip}\n"
                response_text += "\n"

        response_text += "🚛 **ප්‍රවාහන උපදෙස්**:\n" if language == 'sinhala' else "🚛 **Transportation Tips**:\n"
        for tip in info['transportation_tips']:
            response_text += f"{tip}\n"

        return {
            "text": response_text,
            "data": info,
            "suggestions": ["Quality standards", "Market recommendations", "Cost optimization"]
        }

    def _handle_market_trends(self, language):
        """Handle market trends and analysis requests"""
        trends_info = {
            'english': {
                'title': "📈 **Banana Market Trends & Analysis**",
                'current_trends': {
                    'Price Trends': {
                        'direction': 'Stable with seasonal variations',
                        'factors': ['Weather conditions', 'Festival demand', 'Export opportunities', 'Supply chain efficiency']
                    },
                    'Demand Patterns': {
                        'high_demand': ['December-January (New Year)', 'April (New Year)', 'Wedding seasons'],
                        'moderate_demand': ['Regular months', 'School seasons'],
                        'low_demand': ['Monsoon periods', 'Harvest glut periods']
                    },
                    'Quality Preferences': {
                        'premium_markets': 'Uniform size, perfect ripeness, organic certification',
                        'standard_markets': 'Good quality, reasonable price, consistent supply',
                        'processing_markets': 'Volume-based, cost-effective, regular supply'
                    }
                },
                'seasonal_analysis': {
                    'January-March': 'High demand, premium prices, festival season',
                    'April-June': 'Moderate demand, stable prices, harvest season',
                    'July-September': 'Variable demand, weather-dependent prices',
                    'October-December': 'Increasing demand, rising prices, festival preparation'
                },
                'market_opportunities': [
                    "🌟 **Export Markets**: Focus on quality and certification",
                    "🏪 **Retail Chains**: Consistent supply and packaging",
                    "🏭 **Processing Industry**: Volume contracts and competitive pricing",
                    "🎯 **Direct Sales**: Farmer markets and online platforms",
                    "📱 **Digital Marketing**: Social media and e-commerce platforms"
                ]
            },
            'sinhala': {
                'title': "📈 **කේසෙල් වෙළඳපොල ප්‍රවණතා සහ විශ්ලේෂණය**",
                'current_trends': {
                    'මිල ප්‍රවණතා': {
                        'direction': 'කන්නාකාර වෙනස්කම් සමඟ ස්ථාවර',
                        'factors': ['කාලගුණික තත්ත්වයන්', 'උත්සව ඉල්ලුම', 'අපනයන අවස්ථා', 'සැපයුම් දාම කාර්යක්ෂමතාව']
                    },
                    'ඉල්ලුම් රටා': {
                        'high_demand': ['දෙසැම්බර්-ජනවාරි (අලුත් අවුරුද්ද)', 'අප්‍රේල් (අලුත් අවුරුද්ද)', 'විවාහ කන්නය'],
                        'moderate_demand': ['සාමාන්‍ය මාස', 'පාසල් කන්නය'],
                        'low_demand': ['මෝසම් කාලය', 'අස්වනු අතිරික්ත කාලය']
                    }
                },
                'market_opportunities': [
                    "🌟 **අපනයන වෙළඳපොල**: ගුණත්වය සහ සහතික කිරීම කෙරෙහි අවධානය",
                    "🏪 **සිල්ලර දාම**: ස්ථාවර සැපයුම සහ ඇසුරුම්",
                    "🏭 **සැකසුම් කර්මාන්තය**: පරිමාණ ගිණුම් සහ තරඟකාරී මිල ගණන්",
                    "🎯 **සෘජු විකිණීම**: ගොවි වෙළඳපොල සහ මාර්ගගත වේදිකා"
                ]
            }
        }

        info = trends_info[language]

        response_text = f"{info['title']}\n\n"

        # Current trends
        response_text += "📊 **වර්තමාන ප්‍රවණතා**:\n\n" if language == 'sinhala' else "📊 **Current Trends**:\n\n"

        for trend_type, details in info['current_trends'].items():
            response_text += f"**{trend_type}**:\n"
            if isinstance(details, dict):
                for key, value in details.items():
                    if isinstance(value, list):
                        response_text += f"• {key}: {', '.join(value)}\n"
                    else:
                        response_text += f"• {key}: {value}\n"
            response_text += "\n"

        # Seasonal analysis (if available)
        if 'seasonal_analysis' in info:
            response_text += "📅 **කන්නාකාර විශ්ලේෂණය**:\n" if language == 'sinhala' else "📅 **Seasonal Analysis**:\n"
            for period, analysis in info['seasonal_analysis'].items():
                response_text += f"• **{period}**: {analysis}\n"
            response_text += "\n"

        # Market opportunities
        response_text += "🚀 **වෙළඳපොල අවස්ථා**:\n" if language == 'sinhala' else "🚀 **Market Opportunities**:\n"
        for opportunity in info['market_opportunities']:
            response_text += f"{opportunity}\n"

        return {
            "text": response_text,
            "data": info,
            "suggestions": ["Price predictions", "Market recommendations", "Seasonal planning"]
        }