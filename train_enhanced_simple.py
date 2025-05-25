# train_enhanced_simple.py
"""
Simplified enhanced model training script with improved features
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
import joblib

class SimplifiedEnhancedTrainer:
    """
    Simplified enhanced trainer with better feature engineering
    """

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importance = {}

    def load_and_enhance_data(self, file_path='weekly retail price of Ambulslit.xlsx'):
        """
        Load and enhance data with better feature engineering
        """
        try:
            # Try different paths
            data_files = [
                file_path,
                f'data/{file_path}',
                f'data\\{file_path}'
            ]

            data = None
            for path in data_files:
                if os.path.exists(path):
                    print(f"Loading data from: {path}")
                    data = pd.read_excel(path)
                    break

            if data is None:
                print("❌ Data file not found. Creating synthetic data for demonstration...")
                return self._create_synthetic_data()

            print(f"Original data shape: {data.shape}")
            print("Data columns:", data.columns.tolist())

            # Enhanced data processing
            enhanced_data = self._enhance_features(data)

            print(f"Enhanced data shape: {enhanced_data.shape}")
            return enhanced_data

        except Exception as e:
            print(f"Error loading data: {e}")
            print("Creating synthetic data for demonstration...")
            return self._create_synthetic_data()

    def _create_synthetic_data(self):
        """
        Create synthetic banana price data for demonstration
        """
        print("Creating synthetic banana price data...")

        # Generate 2 years of weekly data
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(weeks=i) for i in range(104)]

        np.random.seed(42)

        # Base price with seasonal variation
        base_prices = []
        for i, date in enumerate(dates):
            # Seasonal pattern
            seasonal = 20 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
            # Trend
            trend = 0.5 * i
            # Random noise
            noise = np.random.normal(0, 10)
            # Base price
            base_price = 100 + seasonal + trend + noise
            base_prices.append(max(50, base_price))  # Minimum price of 50

        # Create DataFrame
        data = pd.DataFrame({
            'Date': dates,
            'Price': base_prices,
            'Location': np.random.choice(['Colombo', 'Kandy', 'Galle', 'Jaffna'], len(dates)),
            'Variety': np.random.choice(['Ambul', 'Kolikuttu', 'Anamalu'], len(dates))
        })

        return self._enhance_features(data)

    def _reshape_weekly_data(self, data):
        """
        Reshape weekly price data from wide format to long format
        """
        print("Reshaping weekly data format...")

        # Get week columns (W1, W2, etc.)
        week_cols = [col for col in data.columns if col.startswith('W') and col[1:].isdigit()]

        # Melt the data to long format
        reshaped_data = []

        for idx, row in data.iterrows():
            year = row.get('Year', 2022)
            market = row.get('Markets', 'Unknown')

            for week_col in week_cols:
                week_num = int(week_col[1:])
                price = row[week_col]

                # Skip if price is not numeric or is NaN
                if pd.isna(price) or not isinstance(price, (int, float)):
                    try:
                        price = float(price)
                    except (ValueError, TypeError):
                        continue

                # Create date from year and week
                date = datetime(int(year), 1, 1) + timedelta(weeks=week_num-1)

                reshaped_data.append({
                    'Date': date,
                    'Price': price,
                    'Location': market,
                    'Year': year,
                    'Week': week_num
                })

        # Convert to DataFrame
        result_df = pd.DataFrame(reshaped_data)

        # Remove rows with invalid prices
        result_df = result_df.dropna(subset=['Price'])
        result_df = result_df[result_df['Price'] > 0]  # Remove zero or negative prices

        print(f"Reshaped data: {len(result_df)} records from {len(data)} original rows")

        return result_df

    def _enhance_features(self, data):
        """
        Enhanced feature engineering
        """
        print("Processing data structure...")
        print("Data shape:", data.shape)
        print("Columns:", data.columns.tolist())

        # Handle the specific structure of the banana price data
        if 'Markets' in data.columns and any(col.startswith('W') for col in data.columns):
            # This is the weekly price data format - need to reshape
            print("Detected weekly price format - reshaping data...")
            data = self._reshape_weekly_data(data)

        # Ensure we have a Date column
        if 'Date' not in data.columns:
            # Try to find date-like columns
            date_cols = [col for col in data.columns if 'date' in col.lower()]
            if date_cols:
                data['Date'] = pd.to_datetime(data[date_cols[0]], errors='coerce')
            else:
                # Create sequential dates
                start_date = datetime(2022, 1, 1)
                data['Date'] = [start_date + timedelta(weeks=i) for i in range(len(data))]
        else:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

        # Ensure we have a Price column
        if 'Price' not in data.columns:
            # Try to find price-like columns
            price_cols = [col for col in data.columns if any(keyword in col.lower()
                         for keyword in ['price', 'cost', 'value', 'rs', 'lkr'])]
            if price_cols:
                data['Price'] = pd.to_numeric(data[price_cols[0]], errors='coerce')
            else:
                # Use first numeric column
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    data['Price'] = data[numeric_cols[0]]
                else:
                    raise ValueError("No suitable price column found")

        # Remove rows with missing dates or prices
        data = data.dropna(subset=['Date', 'Price'])

        # Sort by date
        data = data.sort_values('Date').reset_index(drop=True)

        # Enhanced date features
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['WeekOfYear'] = data['Date'].dt.isocalendar().week
        data['WeekOfMonth'] = (data['Date'].dt.day - 1) // 7 + 1
        data['DayOfWeek'] = data['Date'].dt.dayofweek
        data['DayOfYear'] = data['Date'].dt.dayofyear
        data['Quarter'] = data['Date'].dt.quarter
        data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)

        # Cyclical features for better seasonality capture
        data['MonthSin'] = np.sin(2 * np.pi * data['Month'] / 12)
        data['MonthCos'] = np.cos(2 * np.pi * data['Month'] / 12)
        data['DayOfYearSin'] = np.sin(2 * np.pi * data['DayOfYear'] / 365)
        data['DayOfYearCos'] = np.cos(2 * np.pi * data['DayOfYear'] / 365)

        # Season indicators
        data['IsSummer'] = ((data['Month'] >= 6) & (data['Month'] <= 8)).astype(int)
        data['IsWinter'] = ((data['Month'] >= 12) | (data['Month'] <= 2)).astype(int)
        data['IsMonsoon'] = ((data['Month'] >= 5) & (data['Month'] <= 9)).astype(int)
        data['IsFestivalSeason'] = ((data['Month'] == 4) | (data['Month'] == 12) | (data['Month'] == 1)).astype(int)

        # Location encoding
        if 'Location' in data.columns:
            data['LocationCode'] = pd.factorize(data['Location'])[0] + 1
        else:
            data['LocationCode'] = 1

        # Variety encoding
        if 'Variety' in data.columns:
            data['VarietyCode'] = pd.factorize(data['Variety'])[0] + 1
        else:
            data['VarietyCode'] = 1

        # Lag features (if we have enough data)
        if len(data) > 10:
            for lag in [1, 2, 4, 8]:
                if len(data) > lag:
                    data[f'Price_Lag{lag}'] = data['Price'].shift(lag)

        # Rolling statistics (if we have enough data)
        if len(data) > 10:
            for window in [4, 8, 12]:
                if len(data) > window:
                    data[f'Price_MA{window}'] = data['Price'].rolling(window=window).mean()
                    data[f'Price_Std{window}'] = data['Price'].rolling(window=window).std()

        # Price momentum
        if len(data) > 2:
            data['Price_Momentum1'] = data['Price'] - data['Price'].shift(1)
            data['Price_PctChange1'] = data['Price'].pct_change(1)

        # External factors (simulated)
        np.random.seed(42)
        data['Temperature'] = 25 + 5 * np.sin(2 * np.pi * data['DayOfYear'] / 365) + np.random.normal(0, 2, len(data))
        data['Rainfall'] = np.maximum(0, 100 + 50 * np.sin(2 * np.pi * (data['DayOfYear'] - 100) / 365) + np.random.normal(0, 30, len(data)))
        data['USD_LKR_Rate'] = 200 + 50 * np.sin(2 * np.pi * data['DayOfYear'] / 365) + np.random.normal(0, 10, len(data))

        # Clean data
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()

        print(f"Final enhanced data shape: {data.shape}")
        return data

    def train_enhanced_model(self, data, test_size=0.2):
        """
        Train enhanced model with multiple algorithms
        """
        try:
            # Prepare features and target
            feature_columns = [col for col in data.columns if col not in ['Date', 'Price', 'Location', 'Variety']]
            X = data[feature_columns]
            y = data['Price']

            print(f"Training with {len(feature_columns)} features and {len(data)} samples")
            print("Features:", feature_columns)

            # Time-series aware split
            if 'Date' in data.columns:
                data_sorted = data.sort_values('Date')
                split_idx = int(len(data_sorted) * (1 - test_size))
                X_train = data_sorted[feature_columns].iloc[:split_idx]
                X_test = data_sorted[feature_columns].iloc[split_idx:]
                y_train = data_sorted['Price'].iloc[:split_idx]
                y_test = data_sorted['Price'].iloc[split_idx:]
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

            print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

            # Feature selection
            selector = SelectKBest(score_func=f_regression, k=min(20, len(feature_columns)))
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)

            selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
            print(f"Selected {len(selected_features)} features:", selected_features)

            # Scaling
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)

            # Define models
            models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100, max_depth=10, min_samples_split=5,
                    random_state=42, n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100, learning_rate=0.1, max_depth=6,
                    random_state=42
                ),
                'extra_trees': ExtraTreesRegressor(
                    n_estimators=100, max_depth=10, min_samples_split=5,
                    random_state=42, n_jobs=-1
                ),
                'ridge': Ridge(alpha=1.0),
                'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
            }

            # Train and evaluate models
            trained_models = {}
            model_scores = {}

            print("\nTraining individual models...")
            for name, model in models.items():
                print(f"Training {name}...")

                try:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)

                    trained_models[name] = model
                    model_scores[name] = {'r2': r2, 'rmse': rmse, 'mae': mae}

                    print(f"  {name} - R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

                except Exception as e:
                    print(f"  Error training {name}: {e}")
                    continue

            # Create ensemble
            if len(trained_models) >= 2:
                print("\nCreating ensemble model...")
                ensemble_models = [(name, model) for name, model in trained_models.items()
                                 if name in ['random_forest', 'gradient_boosting', 'extra_trees']]

                if len(ensemble_models) >= 2:
                    ensemble = VotingRegressor(estimators=ensemble_models)
                    ensemble.fit(X_train_scaled, y_train)

                    y_pred_ensemble = ensemble.predict(X_test_scaled)
                    r2_ensemble = r2_score(y_test, y_pred_ensemble)
                    rmse_ensemble = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
                    mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)

                    model_scores['ensemble'] = {'r2': r2_ensemble, 'rmse': rmse_ensemble, 'mae': mae_ensemble}
                    trained_models['ensemble'] = ensemble

                    print(f"  Ensemble - R²: {r2_ensemble:.4f}, RMSE: {rmse_ensemble:.2f}, MAE: {mae_ensemble:.2f}")

            # Select best model
            best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['r2'])
            self.best_model = trained_models[best_model_name]

            print(f"\nBest model: {best_model_name} with R² = {model_scores[best_model_name]['r2']:.4f}")

            # Create pipeline
            pipeline = Pipeline([
                ('feature_selector', selector),
                ('scaler', scaler),
                ('model', self.best_model)
            ])

            # Get feature importance
            if hasattr(self.best_model, 'feature_importances_'):
                self.feature_importance = dict(zip(selected_features, self.best_model.feature_importances_))

            return pipeline, X_test, y_test, self.feature_importance, model_scores

        except Exception as e:
            print(f"Error in model training: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None, None

def main():
    """
    Main training function
    """
    print("🍌 SmartMusa Enhanced Model Training (Simplified)")
    print("=" * 60)

    trainer = SimplifiedEnhancedTrainer()

    # Load and enhance data
    print("\n📊 Loading and enhancing data...")
    data = trainer.load_and_enhance_data()

    if data is None or len(data) < 10:
        print("❌ Insufficient data for training")
        return False

    # Train model
    print("\n🤖 Training enhanced model...")
    pipeline, X_test, y_test, feature_importance, model_scores = trainer.train_enhanced_model(data)

    if pipeline is None:
        print("❌ Model training failed")
        return False

    # Evaluate model
    print("\n📈 Model Performance:")
    if model_scores:
        for model_name, scores in model_scores.items():
            print(f"  • {model_name:<15} R²: {scores['r2']:.4f}, RMSE: {scores['rmse']:.2f}")

    # Feature importance
    if feature_importance:
        print("\n🔍 Top 10 Most Important Features:")
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"  {i:2d}. {feature:<20} {importance:8.4f}")

    # Save model
    print("\n💾 Saving enhanced model...")
    try:
        model_path = 'enhanced_banana_price_model.pkl'
        joblib.dump(pipeline, model_path)
        print(f"✅ Enhanced model saved to: {model_path}")

        # Also save as default model
        default_path = 'banana_price_model.pkl'
        joblib.dump(pipeline, default_path)
        print(f"✅ Model also saved as: {default_path}")

    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False

    print("\n🎉 Enhanced model training completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Training failed")
    else:
        print("\n🚀 You can now use the enhanced model with the SmartMusa application!")
