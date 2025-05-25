# config.py
import os

# Get the directory of this script file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Flask application settings
DEBUG = True
SECRET_KEY = os.environ.get('SECRET_KEY', 'smartmusa-secret-key')
PORT = int(os.environ.get('PORT', 5000))

# File paths - using BASE_DIR ensures we have absolute paths
MODEL_PATH = os.path.join(BASE_DIR, 'banana_price_model.pkl')
DATA_FILE = os.path.join(BASE_DIR, 'weekly retail price of Ambulslit.xlsx')
# Alternative data file path if kept in a data subdirectory
DATA_FILE_ALT = os.path.join(BASE_DIR, 'data', 'weekly retail price of Ambulslit.xlsx')

# Feature column names - will be adjusted dynamically based on data
# These are just default values that will be overridden based on actual data
FEATURE_COLUMNS = ['Month', 'WeekOfMonth', 'DayOfWeek', 'LocationCode']
TARGET_COLUMN = 'Price'  # The column containing the prices

# Model settings
MODEL_TYPE = 'random_forest'  # Alternative: 'linear_regression'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Market information
DEFAULT_MARKETS = [
    {"name": "Colombo Manning Market", "location": "colombo", "base_distance": 0},
    {"name": "Dambulla Economic Center", "location": "dambulla", "base_distance": 150},
    {"name": "Meegoda Economic Center", "location": "meegoda", "base_distance": 30},
    {"name": "Kandy Market", "location": "kandy", "base_distance": 120},
    {"name": "Galle Central Market", "location": "galle", "base_distance": 130},
    {"name": "Jaffna Central Market", "location": "jaffna", "base_distance": 380},
    {"name": "Nuwara Eliya Market", "location": "nuwara eliya", "base_distance": 180}
]

# Supported languages
SUPPORTED_LANGUAGES = ['english', 'sinhala']
DEFAULT_LANGUAGE = 'english'

# API settings
API_VERSION = 'v1'
API_PREFIX = f'/api/{API_VERSION}'

# Logging configuration
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
LOG_FILE = os.path.join(BASE_DIR, 'smartmusa.log')