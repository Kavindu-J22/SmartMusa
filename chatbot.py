# chatbot.py
import re
import json
from datetime import datetime
import joblib
import numpy as np

class BananaChatbot:
    """
    A chatbot that provides banana farmers with market recommendations and price predictions
    """
    
    def __init__(self, model_path='banana_price_model.pkl'):
        """
        Initialize the chatbot
        
        Args:
            model_path (str): Path to the trained price prediction model
        """
        self.model = None
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}. Error: {e}")
        
        # Define the intents and patterns
        self.intents = {
            'greeting': [
                r'hello', r'hi', r'hey', r'good morning', r'good afternoon', 
                r'ආයුබෝවන්', r'හෙලෝ', r'හායි'
            ],
            'price_inquiry': [
                r'price', r'how much', r'market price', r'banana price', r'current price',
                r'මිල', r'කීයද', r'වෙළඳපොල මිල', r'කේසෙල් මිල'
            ],
            'market_recommendation': [
                r'where to sell', r'best market', r'recommend market', r'where should i sell',
                r'විකුණන්නේ කොහේද', r'හොඳම වෙළඳපොල', r'වෙළඳපොල නිර්දේශ'
            ],
            'weather_inquiry': [
                r'weather', r'rain', r'forecast', r'climate',
                r'කාලගුණය', r'වැස්ස', r'අනාවැකිය'
            ],
            'farming_advice': [
                r'how to grow', r'fertilizer', r'disease', r'pest', r'cultivation',
                r'වගා කරන්නේ කෙසේද', r'පොහොර', r'රෝග', r'පළිබෝධ', r'වගාව'
            ],
            'help': [
                r'help', r'guide', r'assistant', r'what can you do',
                r'උදව්', r'මාර්ගෝපදේශය', r'සහායක', r'ඔබට කුමක් කළ හැකිද'
            ],
            'goodbye': [
                r'bye', r'goodbye', r'see you', r'thanks', r'thank you',
                r'ගිහින් එන්නම්', r'ස්තූතියි'
            ]
        }
        
        # Load language-specific responses
        self.responses = {
            'english': {
                'greeting': [
                    "Hello! I'm SmartMusa, your banana farming assistant. How can I help you today?",
                    "Hi there! Welcome to SmartMusa. What information do you need about banana farming or markets?"
                ],
                'price_inquiry': [
                    "I can help you with current banana prices. Could you tell me your location and the type of bananas you grow?",
                    "To provide accurate price information, I need to know your location and the quantity of bananas you have."
                ],
                'market_recommendation': [
                    "I can recommend the best market to sell your bananas. Please share your location and the quantity you want to sell.",
                    "For market recommendations, I'll need your current location and details about your harvest."
                ],
                'weather_inquiry': [
                    "I can provide weather information relevant to banana farming. What's your location?",
                    "Weather forecasts can help with planning your farming activities. Which area are you interested in?"
                ],
                'farming_advice': [
                    "I can offer advice on banana cultivation. What specific aspect are you interested in?",
                    "For farming advice, I can help with planting, disease control, or harvesting techniques. What do you need help with?"
                ],
                'help': [
                    "I can help with market prices, recommend where to sell your bananas, provide weather information, and offer farming advice. What would you like to know?",
                    "As your banana farming assistant, I can provide market information, selling recommendations, weather updates, and cultivation advice. How can I assist you today?"
                ],
                'goodbye': [
                    "Thank you for using SmartMusa! Feel free to return whenever you need assistance with banana farming.",
                    "Goodbye! I'm here whenever you need information about banana farming or markets."
                ],
                'default': [
                    "I'm not sure I understand. Could you please rephrase your question about banana farming or markets?",
                    "I don't have information about that. I can help with banana prices, market recommendations, weather forecasts, and farming advice."
                ]
            },
            'sinhala': {
                'greeting': [
                    "ආයුබෝවන්! මම SmartMusa, ඔබේ කේසෙල් වගා සහායකයා. මට ඔබට කෙසේ උදව් කළ හැකිද?",
                    "හෙලෝ! SmartMusa වෙත සාදරයෙන් පිළිගනිමු. ඔබට කේසෙල් වගාව හෝ වෙළඳපොලවල් ගැන කුමන තොරතුරු අවශ්‍යද?"
                ],
                'price_inquiry': [
                    "මට වර්තමාන කේසෙල් මිල ගැන උදව් කළ හැකිය. ඔබේ ස්ථානය සහ ඔබ වගා කරන කේසෙල් වර්ගය මට කිව හැකිද?",
                    "නිවැරදි මිල තොරතුරු සැපයීමට, මට ඔබේ ස්ථානය සහ ඔබට ඇති කේසෙල් ප්‍රමාණය දැන ගැනීමට අවශ්‍යයි."
                ],
                'market_recommendation': [
                    "මට ඔබේ කේසෙල් විකිණීමට හොඳම වෙළඳපොල නිර්දේශ කළ හැකිය. කරුණාකර ඔබේ ස්ථානය සහ විකිණීමට අවශ්‍ය ප්‍රමාණය බෙදා ගන්න.",
                    "වෙළඳපොල නිර්දේශ සඳහා, මට ඔබේ වර්තමාන ස්ථානය සහ ඔබේ අස්වැන්න පිළිබඳ විස්තර අවශ්‍ය වනු ඇත."
                ],
                'weather_inquiry': [
                    "මට කේසෙල් වගාවට අදාළ කාලගුණ තොරතුරු සැපයිය හැකිය. ඔබේ ස්ථානය කුමක්ද?",
                    "කාලගුණ අනාවැකි ඔබේ වගා කටයුතු සැලසුම් කිරීමට උපකාර විය හැක. ඔබ උනන්දුවක් දක්වන ප්‍රදේශය කුමක්ද?"
                ],
                'farming_advice': [
                    "මට කේසෙල් වගාව පිළිබඳ උපදෙස් දිය හැකිය. ඔබ උනන්දුවක් දක්වන නිශ්චිත අංශය කුමක්ද?",
                    "වගා උපදෙස් සඳහා, මට පැළ කිරීම, රෝග පාලනය හෝ අස්වනු නෙලීමේ ක්‍රම ගැන උදව් කළ හැකිය. ඔබට කුමක් සමඟ උදව් අවශ්‍යද?"
                ],
                'help': [
                    "මට වෙළඳපොල මිල, ඔබේ කේසෙල් විකිණීමට කොතැනද යන්න, කාලගුණ තොරතුරු සපයන්න, සහ වගා උපදෙස් ලබා දිය හැකිය. ඔබට දැන ගැනීමට අවශ්‍ය කුමක්ද?",
                    "ඔබේ කේසෙල් වගා සහායක ලෙස, මට වෙළඳපොල තොරතුරු, විකිණීමේ නිර්දේශ, කාලගුණ යාවත්කාලීන කිරීම් සහ වගා උපදෙස් සැපයිය හැකිය. අද මට ඔබට කෙසේ සහාය විය හැකිද?"
                ],
                'goodbye': [
                    "SmartMusa භාවිතා කිරීම ගැන ඔබට ස්තූතියි! කේසෙල් වගාව සමඟ උදව් අවශ්‍ය වන විට ඕනෑම වේලාවක නැවත පැමිණෙන්න.",
                    "ආයුබෝවන්! ඔබට කේසෙල් වගාව හෝ වෙළඳපොලවල් ගැන තොරතුරු අවශ්‍ය වන විට මම මෙහි සිටිමි."
                ],
                'default': [
                    "මම තේරුම් ගන්නේ නැහැ. කරුණාකර කේසෙල් වගාව හෝ වෙළඳපොලවල් ගැන ඔබේ ප්‍රශ්නය යළි සඳහන් කරන්න.",
                    "මට ඒ ගැන තොරතුරු නැත. මට කේසෙල් මිල, වෙළඳපොල නිර්දේශ, කාලගුණ අනාවැකි සහ වගා උපදෙස් සමඟ උදව් කළ හැකිය."
                ]
            }
        }
        
        # Track conversation context
        self.context = {}
    
    def detect_intent(self, message, language='english'):
        """
        Detect the intent of the user's message
        
        Args:
            message (str): User's message
            language (str): Language of the message ('english' or 'sinhala')
            
        Returns:
            str: The detected intent
        """
        message = message.lower()
        
        # Check each intent's patterns
        for intent, patterns in self.intents.items():
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    return intent
        
        # If no intent is detected
        return 'default'
    
    def extract_entities(self, message):
        """
        Extract relevant entities from the user's message
        
        Args:
            message (str): User's message
            
        Returns:
            dict: Extracted entities
        """
        entities = {}
        
        # Extract location
        location_patterns = [
            r'in\s+([A-Za-z\s]+)', 
            r'near\s+([A-Za-z\s]+)', 
            r'at\s+([A-Za-z\s]+)',
            r'location\s+is\s+([A-Za-z\s]+)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                entities['location'] = match.group(1).strip()
                break
        
        # Extract quantity
        quantity_patterns = [
            r'(\d+)\s+kg', 
            r'(\d+)\s+kilos',
            r'(\d+)\s+kilograms',
            r'quantity\s+of\s+(\d+)',
            r'(\d+)\s+bananas'
        ]
        
        for pattern in quantity_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                entities['quantity'] = int(match.group(1))
                break
        
        # Extract banana type
        banana_types = ['ambul', 'kolikuttu', 'anamalu', 'seeni', 'rathkesel']
        for banana_type in banana_types:
            if banana_type in message.lower():
                entities['banana_type'] = banana_type
                break
        
        return entities
    
    def predict_price(self, location, banana_type='ambul'):
        """
        Predict the banana price based on current date and location
        
        Args:
            location (str): Farmer's location
            banana_type (str): Type of banana
            
        Returns:
            float: Predicted price
        """
        if self.model is None:
            return {"error": "Price prediction model not loaded"}
        
        try:
            # Map location to location code (this would be more sophisticated in production)
            location_code = 1  # Default
            location_mapping = {
                'colombo': 1,
                'kandy': 2,
                'galle': 3,
                'jaffna': 4,
                'anuradhapura': 5
            }
            
            location = location.lower()
            for key in location_mapping:
                if key in location:
                    location_code = location_mapping[key]
                    break
            
            # Get current date info
            now = datetime.now()
            month = now.month
            week_of_month = (now.day - 1) // 7 + 1
            
            # Create feature array for prediction
            features = np.array([month, week_of_month, location_code]).reshape(1, -1)
            
            # Make prediction
            predicted_price = float(self.model.predict(features)[0])
            
            return {
                "price": round(predicted_price, 2),
                "currency": "LKR",
                "banana_type": banana_type,
                "location": location,
                "date": now.strftime("%Y-%m-%d")
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_market_recommendation(self, location, quantity, banana_type='ambul'):
        """
        Generate market recommendations based on farmer's data
        
        Args:
            location (str): Farmer's location
            quantity (int): Quantity of bananas (in kg)
            banana_type (str): Type of banana
            
        Returns:
            dict: Market recommendations
        """
        try:
            # This is a placeholder implementation
            # In a real system, this would query nearby markets and their predicted prices
            
            # Sample nearby markets based on location
            nearby_markets = self._get_nearby_markets(location)
            
            # For each market, predict the price and calculate potential profit
            for market in nearby_markets:
                # Predict price at this market
                market_location = market["name"]
                price_info = self.predict_price(market_location, banana_type)
                
                if "error" in price_info:
                    market["predicted_price"] = 120  # Default fallback price
                else:
                    market["predicted_price"] = price_info["price"]
                
                # Calculate transportation cost (simplified)
                distance_cost_factor = 0.5  # Cost per km per kg
                transportation_cost = market["distance"] * distance_cost_factor
                
                # Calculate potential profit
                market["transportation_cost"] = transportation_cost
                market["potential_profit"] = (market["predicted_price"] * quantity) - (transportation_cost * quantity)
            
            # Sort markets by potential profit
            sorted_markets = sorted(nearby_markets, key=lambda x: x["potential_profit"], reverse=True)
            
            return {
                "best_market": sorted_markets[0],
                "alternative_markets": sorted_markets[1:],
                "banana_type": banana_type,
                "quantity": quantity
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_nearby_markets(self, location):
        """
        Get nearby markets based on farmer's location
        
        Args:
            location (str): Farmer's location
            
        Returns:
            list: List of nearby markets with distances
        """
        # This is a placeholder implementation
        # In a real system, this would query a database of markets and calculate actual distances
        
        # Sample markets in Sri Lanka
        all_markets = [
            {"name": "Colombo Manning Market", "location": "colombo", "distance": 0},
            {"name": "Dambulla Economic Center", "location": "dambulla", "distance": 0},
            {"name": "Meegoda Economic Center", "location": "meegoda", "distance": 0},
            {"name": "Kandy Market", "location": "kandy", "distance": 0},
            {"name": "Galle Central Market", "location": "galle", "distance": 0},
            {"name": "Jaffna Central Market", "location": "jaffna", "distance": 0},
            {"name": "Nuwara Eliya Market", "location": "nuwara eliya", "distance": 0}
        ]
        
        # Distance matrix (simplified)
        # In a real system, this would be calculated using maps/distance API
        distance_matrix = {
            "colombo": {
                "colombo": 5,
                "dambulla": 150,
                "meegoda": 30,
                "kandy": 120,
                "galle": 130,
                "jaffna": 380,
                "nuwara eliya": 180
            },
            "kandy": {
                 "colombo": 120,
                "dambulla": 70,
                "meegoda": 100,
                "kandy": 5,
                "galle": 230,
                "jaffna": 290,
                "nuwara eliya": 80
            },
            "galle": {
                "colombo": 130,
                "dambulla": 280,
                "meegoda": 160,
                "kandy": 230,
                "galle": 5,
                "jaffna": 480,
                "nuwara eliya": 280
            },
            "jaffna": {
                "colombo": 380,
                "dambulla": 210,
                "meegoda": 410,
                "kandy": 290,
                "galle": 480,
                "jaffna": 5,
                "nuwara eliya": 360
            },
            "anuradhapura": {
                "colombo": 200,
                "dambulla": 60,
                "meegoda": 230,
                "kandy": 110,
                "galle": 330,
                "jaffna": 190,
                "nuwara eliya": 190
            },
            "dambulla": {
                "colombo": 150,
                "dambulla": 5,
                "meegoda": 180,
                "kandy": 70,
                "galle": 280,
                "jaffna": 210,
                "nuwara eliya": 140
            }
        }
        
        # Find closest location in our matrix
        farmer_location = "colombo"  # Default
        for loc in distance_matrix.keys():
            if loc in location.lower():
                farmer_location = loc
                break
        
        # Update distances for markets
        nearby_markets = []
        for market in all_markets:
            market_loc = market["location"]
            if farmer_location in distance_matrix and market_loc in distance_matrix[farmer_location]:
                market_copy = market.copy()
                market_copy["distance"] = distance_matrix[farmer_location][market_loc]
                nearby_markets.append(market_copy)
            else:
                # If exact match not found, use a default distance
                market_copy = market.copy()
                market_copy["distance"] = 200  # Default distance
                nearby_markets.append(market_copy)
        
        return nearby_markets
    
    def get_response(self, message, language='english'):
        """
        Generate a response based on the user's message
        
        Args:
            message (str): User's message
            language (str): Language of the message ('english' or 'sinhala')
            
        Returns:
            dict: Response with text and any additional data
        """
        # Detect intent
        intent = self.detect_intent(message, language)
        
        # Extract entities
        entities = self.extract_entities(message)
        
        # Update context with new entities
        self.context.update(entities)
        
        # Generate response based on intent
        if intent == 'price_inquiry' and 'location' in self.context:
            # If we have location, provide price prediction
            price_info = self.predict_price(
                self.context['location'],
                self.context.get('banana_type', 'ambul')
            )
            
            if "error" in price_info:
                response_text = f"I'm sorry, I couldn't predict the price at this time. {price_info['error']}"
            else:
                response_text = f"The current estimated price for {price_info['banana_type']} bananas in {price_info['location']} is {price_info['price']} {price_info['currency']} per kg."
            
            return {
                "text": response_text,
                "data": price_info
            }
            
        elif intent == 'market_recommendation' and 'location' in self.context and 'quantity' in self.context:
            # If we have location and quantity, provide market recommendation
            recommendation = self.get_market_recommendation(
                self.context['location'],
                self.context['quantity'],
                self.context.get('banana_type', 'ambul')
            )
            
            if "error" in recommendation:
                response_text = f"I'm sorry, I couldn't generate a recommendation at this time. {recommendation['error']}"
            else:
                best_market = recommendation["best_market"]
                response_text = (
                    f"Based on your location and quantity ({recommendation['quantity']} kg of "
                    f"{recommendation['banana_type']} bananas), I recommend selling at "
                    f"{best_market['name']}. The estimated price is {best_market['predicted_price']} LKR/kg, "
                    f"with a potential profit of {int(best_market['potential_profit'])} LKR after "
                    f"transportation costs."
                )
            
            return {
                "text": response_text,
                "data": recommendation
            }
            
        else:
            # For other intents or if we don't have enough context, provide a general response
            if language in self.responses and intent in self.responses[language]:
                response_text = self.responses[language][intent][0]
            else:
                response_text = self.responses['english']['default'][0]
            
            return {
                "text": response_text,
                "data": None
            }