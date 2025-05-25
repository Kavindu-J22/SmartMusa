# enhanced_data_processing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import warnings
from datetime import datetime, timedelta
import requests
import json
import os
warnings.filterwarnings('ignore')

# Install required packages if not available
try:
    import xgboost as xgb
except ImportError:
    print("XGBoost not installed. Installing...")
    os.system("pip install xgboost")
    import xgboost as xgb

try:
    import lightgbm as lgb
except ImportError:
    print("LightGBM not installed. Installing...")
    os.system("pip install lightgbm")
    import lightgbm as lgb

class EnhancedBananaPricePredictor:
    """
    Enhanced banana price prediction system with advanced features
    """

    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.scaler = None
        self.feature_columns = []

    def load_and_process_data(self, file_path='weekly retail price of Ambulslit.xlsx'):
        """
        Enhanced data loading and processing with advanced feature engineering
        """
        try:
            # Load the Excel file
            if os.path.exists(file_path):
                data = pd.read_excel(file_path)
            elif os.path.exists(f'data/{file_path}'):
                data = pd.read_excel(f'data/{file_path}')
            else:
                print(f"Data file not found: {file_path}")
                return None

            print("Original data preview:")
            print(data.head())
            print(f"Data shape: {data.shape}")

            # Enhanced data processing
            processed_data = data.copy()

            # Handle date information
            processed_data = self._process_date_features(processed_data)

            # Handle price information
            processed_data = self._process_price_features(processed_data)

            # Handle location information
            processed_data = self._process_location_features(processed_data)

            # Create advanced features
            processed_data = self._create_advanced_features(processed_data)

            # Create lag and rolling features
            processed_data = self._create_temporal_features(processed_data)

            # Add external data features (weather, economic indicators)
            processed_data = self._add_external_features(processed_data)

            # Clean and finalize data
            processed_data = self._clean_and_finalize(processed_data)

            print(f"\nProcessed data shape: {processed_data.shape}")
            print("\nFeature columns:")
            feature_cols = [col for col in processed_data.columns if col != 'Price']
            for col in feature_cols:
                print(f"- {col}")

            return processed_data

        except Exception as e:
            print(f"Error processing data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_date_features(self, data):
        """Process and extract date features"""
        # Find date column
        date_columns = [col for col in data.columns if 'date' in col.lower()]

        if date_columns:
            date_col = date_columns[0]
            data['Date'] = pd.to_datetime(data[date_col], errors='coerce')
        elif 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        else:
            # Create sequential dates if no date column exists
            print("No date information found. Creating sequential dates.")
            start_date = datetime(2020, 1, 1)
            data['Date'] = [start_date + timedelta(days=i*7) for i in range(len(data))]

        # Extract comprehensive date features
        if 'Date' in data.columns:
            data['Year'] = data['Date'].dt.year
            data['Month'] = data['Date'].dt.month
            data['WeekOfYear'] = data['Date'].dt.isocalendar().week
            data['WeekOfMonth'] = (data['Date'].dt.day - 1) // 7 + 1
            data['DayOfWeek'] = data['Date'].dt.dayofweek
            data['DayOfMonth'] = data['Date'].dt.day
            data['DayOfYear'] = data['Date'].dt.dayofyear
            data['Quarter'] = data['Date'].dt.quarter
            data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)
            data['IsMonthStart'] = data['Date'].dt.is_month_start.astype(int)
            data['IsMonthEnd'] = data['Date'].dt.is_month_end.astype(int)
            data['IsQuarterStart'] = data['Date'].dt.is_quarter_start.astype(int)
            data['IsQuarterEnd'] = data['Date'].dt.is_quarter_end.astype(int)

            # Cyclical features for better seasonality capture
            data['MonthSin'] = np.sin(2 * np.pi * data['Month'] / 12)
            data['MonthCos'] = np.cos(2 * np.pi * data['Month'] / 12)
            data['DayOfYearSin'] = np.sin(2 * np.pi * data['DayOfYear'] / 365)
            data['DayOfYearCos'] = np.cos(2 * np.pi * data['DayOfYear'] / 365)
            data['WeekOfYearSin'] = np.sin(2 * np.pi * data['WeekOfYear'] / 52)
            data['WeekOfYearCos'] = np.cos(2 * np.pi * data['WeekOfYear'] / 52)

            # Season indicators
            data['IsSummer'] = ((data['Month'] >= 6) & (data['Month'] <= 8)).astype(int)
            data['IsWinter'] = ((data['Month'] >= 12) | (data['Month'] <= 2)).astype(int)
            data['IsMonsoon'] = ((data['Month'] >= 5) & (data['Month'] <= 9)).astype(int)

        return data

    def _process_price_features(self, data):
        """Process price information"""
        # Find price column
        price_columns = [col for col in data.columns if any(keyword in col.lower()
                        for keyword in ['price', 'cost', 'value', 'rs', 'lkr', 'rupee'])]

        if price_columns:
            price_col = price_columns[0]
            data['Price'] = pd.to_numeric(data[price_col], errors='coerce')
        else:
            # Try to identify numeric columns that might be prices
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if not any(keyword in col.lower()
                           for keyword in ['date', 'year', 'month', 'day', 'code', 'id'])]

            if numeric_cols:
                print(f"No clear price column found. Using '{numeric_cols[0]}' as price.")
                data['Price'] = data[numeric_cols[0]]
            else:
                print("No suitable price column found.")
                return None

        # Remove outliers using IQR method
        Q1 = data['Price'].quantile(0.25)
        Q3 = data['Price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_before = len(data)
        data = data[(data['Price'] >= lower_bound) & (data['Price'] <= upper_bound)]
        outliers_after = len(data)

        if outliers_before != outliers_after:
            print(f"Removed {outliers_before - outliers_after} outliers from price data")

        return data

    def _process_location_features(self, data):
        """Process location information"""
        location_columns = [col for col in data.columns if any(keyword in col.lower()
                           for keyword in ['location', 'market', 'area', 'region', 'district', 'city'])]

        if location_columns:
            location_col = location_columns[0]
            data['Location'] = data[location_col]
            # Create numeric location codes
            data['LocationCode'] = pd.factorize(data['Location'])[0] + 1

            # Create location-based features
            unique_locations = data['Location'].unique()
            for i, location in enumerate(unique_locations):
                data[f'Location_{location}'] = (data['Location'] == location).astype(int)
        else:
            data['LocationCode'] = 1
            data['Location_Default'] = 1

        return data

    def _create_advanced_features(self, data):
        """Create advanced engineered features"""
        if 'Price' in data.columns:
            # Price transformation features
            data['Price_Log'] = np.log1p(data['Price'])
            data['Price_Sqrt'] = np.sqrt(data['Price'])

            # Price statistics
            data['Price_ZScore'] = (data['Price'] - data['Price'].mean()) / data['Price'].std()

        # Interaction features
        if 'Month' in data.columns and 'LocationCode' in data.columns:
            data['Month_Location_Interaction'] = data['Month'] * data['LocationCode']

        if 'Quarter' in data.columns and 'LocationCode' in data.columns:
            data['Quarter_Location_Interaction'] = data['Quarter'] * data['LocationCode']

        # Time-based aggregations
        if 'Date' in data.columns and 'Price' in data.columns:
            # Monthly aggregations
            monthly_stats = data.groupby(data['Date'].dt.to_period('M'))['Price'].agg(['mean', 'std', 'min', 'max'])
            monthly_stats.index = monthly_stats.index.to_timestamp()

            data['Monthly_Price_Mean'] = data['Date'].map(monthly_stats['mean'])
            data['Monthly_Price_Std'] = data['Date'].map(monthly_stats['std'])
            data['Monthly_Price_Range'] = data['Date'].map(monthly_stats['max'] - monthly_stats['min'])

        return data

    def _create_temporal_features(self, data):
        """Create lag and rolling window features"""
        if 'Date' in data.columns and 'Price' in data.columns:
            # Sort by date
            data = data.sort_values('Date').reset_index(drop=True)

            # Lag features
            for lag in [1, 2, 3, 4, 8, 12]:
                data[f'Price_Lag{lag}'] = data['Price'].shift(lag)

            # Rolling statistics
            for window in [2, 4, 8, 12, 24]:
                data[f'Price_MA{window}'] = data['Price'].rolling(window=window).mean()
                data[f'Price_Std{window}'] = data['Price'].rolling(window=window).std()
                data[f'Price_Min{window}'] = data['Price'].rolling(window=window).min()
                data[f'Price_Max{window}'] = data['Price'].rolling(window=window).max()

            # Price momentum and change features
            data['Price_Momentum1'] = data['Price'] - data['Price_Lag1']
            data['Price_Momentum2'] = data['Price'] - data['Price_Lag2']
            data['Price_Momentum4'] = data['Price'] - data['Price_Lag4']

            # Price change percentages
            data['Price_PctChange1'] = data['Price'].pct_change(1)
            data['Price_PctChange2'] = data['Price'].pct_change(2)
            data['Price_PctChange4'] = data['Price'].pct_change(4)

            # Volatility features
            data['Price_Volatility4'] = data['Price'].rolling(window=4).std()
            data['Price_Volatility8'] = data['Price'].rolling(window=8).std()
            data['Price_Volatility12'] = data['Price'].rolling(window=12).std()

            # Trend features
            for window in [4, 8, 12]:
                data[f'Price_Trend{window}'] = data['Price'].rolling(window=window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
                )

        return data

    def _add_external_features(self, data):
        """Add external data features (weather, economic indicators, etc.)"""
        # Add simulated external features (in production, these would come from APIs)

        if 'Date' in data.columns:
            # Simulated weather features
            np.random.seed(42)  # For reproducibility
            data['Temperature'] = 25 + 5 * np.sin(2 * np.pi * data['DayOfYear'] / 365) + np.random.normal(0, 2, len(data))
            data['Rainfall'] = np.maximum(0, 100 + 50 * np.sin(2 * np.pi * (data['DayOfYear'] - 100) / 365) + np.random.normal(0, 30, len(data)))
            data['Humidity'] = 70 + 10 * np.sin(2 * np.pi * data['DayOfYear'] / 365) + np.random.normal(0, 5, len(data))

            # Simulated economic indicators
            data['USD_LKR_Rate'] = 200 + 50 * np.sin(2 * np.pi * data['DayOfYear'] / 365) + np.random.normal(0, 10, len(data))
            data['Fuel_Price'] = 150 + 20 * np.sin(2 * np.pi * data['DayOfYear'] / 365) + np.random.normal(0, 5, len(data))

            # Market demand indicators (simulated)
            data['Market_Demand_Index'] = 100 + 20 * np.sin(2 * np.pi * data['DayOfYear'] / 365) + np.random.normal(0, 10, len(data))

            # Festival and holiday indicators
            data['Is_Festival_Season'] = ((data['Month'] == 4) | (data['Month'] == 12) | (data['Month'] == 1)).astype(int)
            data['Is_Harvest_Season'] = ((data['Month'] >= 3) & (data['Month'] <= 5)).astype(int)

        return data

    def _clean_and_finalize(self, data):
        """Clean data and prepare final feature set"""
        # Remove rows with too many NaN values
        data = data.dropna(thresh=len(data.columns) * 0.7)

        # Fill remaining NaN values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'Price':  # Don't fill target variable
                data[col] = data[col].fillna(data[col].median())

        # Remove infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()

        # Ensure we have enough data
        if len(data) < 10:
            print("Warning: Very little data remaining after cleaning")

        return data