# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
import traceback

# Import configuration
import config

# Import the chatbot
from chatbot import BananaChatbot

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['DEBUG'] = config.DEBUG
CORS(app)  # Enable CORS for all routes

# Initialize the model and chatbot
model = None
chatbot = None

def load_model():
    """Load the trained model if it exists"""
    global model
    try:
        if os.path.exists(config.MODEL_PATH):
            logger.info(f"Loading model from {config.MODEL_PATH}")
            model = joblib.load(config.MODEL_PATH)
            return True
        else:
            logger.warning(f"Model file not found at {config.MODEL_PATH}")
            return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def initialize_chatbot():
    """Initialize the chatbot"""
    global chatbot
    try:
        logger.info("Initializing chatbot")
        chatbot = BananaChatbot(config.MODEL_PATH)
        return True
    except Exception as e:
        logger.error(f"Error initializing chatbot: {e}")
        return False

@app.before_first_request
def before_first_request():
    """Initialize resources before the first request"""
    load_model()
    initialize_chatbot()

@app.route('/')
def index():
    """Home page"""
    return jsonify({
        "message": "Welcome to SmartMusa API",
        "version": config.API_VERSION,
        "status": "running"
    })

@app.route(f'{config.API_PREFIX}/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running"""
    model_loaded = model is not None
    chatbot_initialized = chatbot is not None
    
    status = "healthy" if model_loaded and chatbot_initialized else "degraded"
    
    return jsonify({
        "status": status,
        "message": "SmartMusa API is running",
        "components": {
            "model": "loaded" if model_loaded else "not loaded",
            "chatbot": "initialized" if chatbot_initialized else "not initialized"
        },
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route(f'{config.API_PREFIX}/predict', methods=['POST'])
def predict_price():
    """Endpoint to predict banana prices based on input parameters"""
    try:
        # Load the model if not already loaded
        global model
        if model is None and not load_model():
            return jsonify({
                "error": "Model not trained yet. Please train the model first.",
                "instructions": "Run the train_model.py script to train the model."
            }), 400
        
        # Get data from request
        data = request.json
        logger.info(f"Received prediction request: {data}")
        
        # Extract features (would need to match your model's expected input)
        features = [
            data.get('month', datetime.now().month),
            data.get('week_of_month', (datetime.now().day - 1) // 7 + 1),
            data.get('day_of_week', datetime.now().weekday()),
            data.get('location_code', 1)  # Default to location code 1 if not provided
        ]
        
        # Convert to the format your model expects
        features = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        result = {
            "predicted_price": float(prediction),
            "currency": "LKR",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "features_used": {
                "month": features[0][0],
                "week_of_month": features[0][1],
                "day_of_week": features[0][2],
                "location_code": features[0][3]
            }
        }
        
        logger.info(f"Prediction result: {result}")
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error in price prediction: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route(f'{config.API_PREFIX}/train', methods=['POST'])
def train_model_endpoint():
    """Endpoint to train or retrain the model with data"""
    try:
        from data_processing import load_and_process_data, train_model, evaluate_model, save_model
        
        # Try to find the data file
        data_file = config.DATA_FILE
        if not os.path.exists(data_file):
            data_file = config.DATA_FILE_ALT
            if not os.path.exists(data_file):
                return jsonify({
                    "status": "error", 
                    "message": f"Data file not found at {config.DATA_FILE} or {config.DATA_FILE_ALT}"
                }), 404
        
        # Load and process data
        data = load_and_process_data(data_file)
        if data is None:
            return jsonify({
                "status": "error", 
                "message": "Failed to process data"
            }), 500
        
        # Train model
        model_result, X_test, y_test, feature_importance = train_model(
            data, 
            model_type=config.MODEL_TYPE,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE
        )
        
        if model_result is None:
            return jsonify({
                "status": "error", 
                "message": "Failed to train model"
            }), 500
        
        # Evaluate model
        eval_metrics = evaluate_model(model_result, X_test, y_test)
        if eval_metrics is None:
            return jsonify({
                "status": "error", 
                "message": "Failed to evaluate model"
            }), 500
        
        # Save model
        if not save_model(model_result, config.MODEL_PATH):
            return jsonify({
                "status": "error", 
                "message": "Failed to save model"
            }), 500
        
        # Update the global model
        global model
        model = model_result
        
        # Reinitialize the chatbot with the new model
        initialize_chatbot()
        
        # Return evaluation metrics
        return jsonify({
            "status": "success",
            "message": "Model trained and saved successfully",
            "metrics": {
                "mse": float(eval_metrics['mse']),
                "rmse": float(eval_metrics['rmse']),
                "r2": float(eval_metrics['r2'])
            },
            "feature_importance": {k: float(v) for k, v in feature_importance.items()}
        }), 200
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route(f'{config.API_PREFIX}/recommend', methods=['POST'])
def get_recommendation():
    """Endpoint to get market recommendations based on farmer's data"""
    try:
        # Initialize chatbot if not already initialized
        global chatbot
        if chatbot is None and not initialize_chatbot():
            return jsonify({"error": "Chatbot not initialized. Please try again later."}), 500
        
        # Get farmer data
        data = request.json
        logger.info(f"Received recommendation request: {data}")
        
        # Extract relevant information
        farmer_location = data.get('location', '')
        quantity = data.get('quantity', 0)
        banana_type = data.get('banana_type', 'ambul')
        
        # Get recommendation from chatbot
        recommendation = chatbot.get_market_recommendation(farmer_location, quantity, banana_type)
        
        if "error" in recommendation:
            return jsonify({"error": recommendation["error"]}), 400
        
        logger.info(f"Recommendation result: {recommendation}")
        return jsonify(recommendation), 200
    
    except Exception as e:
        logger.error(f"Error generating recommendation: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route(f'{config.API_PREFIX}/chatbot', methods=['POST'])
def chatbot_response():
    """Endpoint for the chatbot interaction"""
    try:
        # Initialize chatbot if not already initialized
        global chatbot
        if chatbot is None and not initialize_chatbot():
            return jsonify({"error": "Chatbot not initialized. Please try again later."}), 500
        
        # Get message from request
        data = request.json
        message = data.get('message', '')
        language = data.get('language', config.DEFAULT_LANGUAGE)
        
        if language not in config.SUPPORTED_LANGUAGES:
            language = config.DEFAULT_LANGUAGE
        
        logger.info(f"Received chatbot message: '{message}' (language: {language})")
        
        # Process the message using our chatbot
        response = chatbot.get_response(message, language)
        
        logger.info(f"Chatbot response: {response['text']}")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error processing chatbot message: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route(f'{config.API_PREFIX}/supported-languages', methods=['GET'])
def get_supported_languages():
    """Endpoint to get the list of supported languages"""
    return jsonify({
        "supported_languages": config.SUPPORTED_LANGUAGES,
        "default_language": config.DEFAULT_LANGUAGE
    }), 200

@app.route(f'{config.API_PREFIX}/markets', methods=['GET'])
def get_markets():
    """Endpoint to get the list of markets"""
    return jsonify({
        "markets": config.DEFAULT_MARKETS
    }), 200

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Ensure the model is loaded
    if not load_model():
        logger.warning("Model not loaded. Some functionality will be limited.")
    
    # Initialize the chatbot
    if not initialize_chatbot():
        logger.warning("Chatbot not initialized. Some functionality will be limited.")
    
    # Run the app
    app.run(debug=config.DEBUG, host='0.0.0.0', port=config.PORT)