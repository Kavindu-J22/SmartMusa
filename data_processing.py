# data_processing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime, timedelta

def load_and_process_data(file_path='weekly retail price of Ambulslit.xlsx'):
    """
    Enhanced data loading and processing with better feature engineering
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        pd.DataFrame: Processed dataframe ready for model training
    """
    try:
        # Load the Excel file
        data = pd.read_excel(file_path)
        
        # Print the first few rows to understand the structure
        print("Original data preview:")
        print(data.head())
        
        # Check for missing values
        print("\nMissing values:")
        print(data.isnull().sum())
        
        # Basic statistics
        print("\nBasic statistics:")
        print(data.describe())
        
        # Enhanced data processing
        processed_data = data.copy()
        
        # Handle date information - convert to datetime if needed
        date_columns = [col for col in processed_data.columns if 'date' in col.lower()]
        if date_columns:
            # Use the first date column found
            date_col = date_columns[0]
            processed_data['Date'] = pd.to_datetime(processed_data[date_col], errors='coerce')
        elif 'Date' in processed_data.columns:
            processed_data['Date'] = pd.to_datetime(processed_data['Date'], errors='coerce')
        else:
            # If no date column exists, try to create one from year/month/day columns
            year_col = next((col for col in processed_data.columns if 'year' in col.lower()), None)
            month_col = next((col for col in processed_data.columns if 'month' in col.lower()), None)
            day_col = next((col for col in processed_data.columns if 'day' in col.lower()), None)
            
            if year_col and month_col:
                # Create date from year and month (use 1 as default day if day_col doesn't exist)
                if day_col:
                    processed_data['Date'] = pd.to_datetime({
                        'year': processed_data[year_col],
                        'month': processed_data[month_col],
                        'day': processed_data[day_col]
                    }, errors='coerce')
                else:
                    processed_data['Date'] = pd.to_datetime({
                        'year': processed_data[year_col],
                        'month': processed_data[month_col],
                        'day': 1
                    }, errors='coerce')
            else:
                # If we can't create a date, create a sequential date
                print("No date information found. Creating sequential dates.")
                start_date = datetime(2020, 1, 1)  # Arbitrary start date
                processed_data['Date'] = [start_date + timedelta(days=i) for i in range(len(processed_data))]
        
        # Extract rich date features
        if 'Date' in processed_data.columns:
            processed_data['Year'] = processed_data['Date'].dt.year
            processed_data['Month'] = processed_data['Date'].dt.month
            processed_data['WeekOfYear'] = processed_data['Date'].dt.isocalendar().week
            processed_data['WeekOfMonth'] = (processed_data['Date'].dt.day - 1) // 7 + 1
            processed_data['DayOfWeek'] = processed_data['Date'].dt.dayofweek
            processed_data['DayOfMonth'] = processed_data['Date'].dt.day
            processed_data['DayOfYear'] = processed_data['Date'].dt.dayofyear
            processed_data['Quarter'] = processed_data['Date'].dt.quarter
            processed_data['IsWeekend'] = processed_data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
            
            # Create month-related cyclical features to capture seasonality
            processed_data['MonthSin'] = np.sin(2 * np.pi * processed_data['Month'] / 12)
            processed_data['MonthCos'] = np.cos(2 * np.pi * processed_data['Month'] / 12)
            processed_data['DayOfYearSin'] = np.sin(2 * np.pi * processed_data['DayOfYear'] / 365)
            processed_data['DayOfYearCos'] = np.cos(2 * np.pi * processed_data['DayOfYear'] / 365)
        
        # Find price column
        price_columns = [col for col in processed_data.columns if 'price' in col.lower() or 
                         'cost' in col.lower() or 'value' in col.lower() or 
                         'rs' in col.lower() or 'lkr' in col.lower()]
        
        if price_columns:
            # Use the first price column found
            price_col = price_columns[0]
            processed_data['Price'] = processed_data[price_col]
        else:
            # Try to identify numeric columns that might be prices
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if 'date' not in col.lower() and 
                          'year' not in col.lower() and 'month' not in col.lower() and 
                          'day' not in col.lower() and 'code' not in col.lower()]
            
            if numeric_cols:
                # Use the first numeric column as price
                print(f"No clear price column found. Using '{numeric_cols[0]}' as price.")
                processed_data['Price'] = processed_data[numeric_cols[0]]
            else:
                print("No suitable price column found. Cannot proceed.")
                return None
        
        # Handle location information
        location_columns = [col for col in processed_data.columns if 'location' in col.lower() or 
                           'market' in col.lower() or 'area' in col.lower() or 
                           'region' in col.lower() or 'district' in col.lower()]
        
        if location_columns:
            # Use the first location column found
            location_col = location_columns[0]
            processed_data['Location'] = processed_data[location_col]
            # Create a numeric location code
            processed_data['LocationCode'] = pd.factorize(processed_data['Location'])[0] + 1
        else:
            # If no location information, use a default location code
            processed_data['LocationCode'] = 1
        
        # Handle banana type information
        type_columns = [col for col in processed_data.columns if 'type' in col.lower() or 
                       'variety' in col.lower() or 'kind' in col.lower()]
        
        if type_columns:
            # Use the first type column found
            type_col = type_columns[0]
            processed_data['BananaType'] = processed_data[type_col]
            # Create a numeric type code
            processed_data['BananaTypeCode'] = pd.factorize(processed_data['BananaType'])[0] + 1
        
        # Create lag features if we have time-ordered data
        if 'Date' in processed_data.columns:
            # Sort by date
            processed_data = processed_data.sort_values('Date')
            
            # Create price lag features (previous periods)
            processed_data['Price_Lag1'] = processed_data['Price'].shift(1)
            processed_data['Price_Lag2'] = processed_data['Price'].shift(2)
            processed_data['Price_Lag4'] = processed_data['Price'].shift(4)
            
            # Create rolling average features
            processed_data['Price_MA2'] = processed_data['Price'].rolling(window=2).mean()
            processed_data['Price_MA4'] = processed_data['Price'].rolling(window=4).mean()
            processed_data['Price_MA8'] = processed_data['Price'].rolling(window=8).mean()
            
            # Create price momentum features
            processed_data['Price_Momentum1'] = processed_data['Price'] - processed_data['Price_Lag1']
            processed_data['Price_Momentum2'] = processed_data['Price'] - processed_data['Price_Lag2']
            
            # Create price volatility features
            processed_data['Price_Volatility4'] = processed_data['Price'].rolling(window=4).std()
            
            # Drop rows with NaN values introduced by lag/rolling features
            processed_data = processed_data.dropna()
        
        # Ensure all non-feature columns are dropped for model training
        feature_columns = [
            'Month', 'WeekOfMonth', 'DayOfWeek', 'DayOfMonth', 'Quarter',
            'IsWeekend', 'MonthSin', 'MonthCos', 'DayOfYearSin', 'DayOfYearCos',
            'LocationCode'
        ]
        
        # Add lag features if they exist
        lag_columns = [col for col in processed_data.columns if 'Lag' in col or 'MA' in col or 'Momentum' in col or 'Volatility' in col]
        feature_columns.extend(lag_columns)
        
        # Ensure all feature columns exist
        feature_columns = [col for col in feature_columns if col in processed_data.columns]
        
        # Print feature information
        print("\nUsing the following features:")
        for col in feature_columns:
            print(f"- {col}")
        
        # Ensure target column exists
        if 'Price' not in processed_data.columns:
            print("Price column not found after processing. Cannot proceed.")
            return None
        
        print("\nProcessed data shape:", processed_data.shape)
        print("\nFirst few rows of processed data:")
        print(processed_data.head())
        
        return processed_data
    
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None

def train_model(data, model_type='gradient_boosting', test_size=0.2, random_state=42):
    """
    Train an advanced price prediction model with hyperparameter tuning
    
    Args:
        data (pd.DataFrame): Processed data
        model_type (str): Type of model to train ('random_forest', 'gradient_boosting', 'ridge', 'lasso')
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (trained_model, X_test, y_test, feature_importance)
    """
    try:
        # Determine which columns to use as features
        feature_columns = [
            'Month', 'WeekOfMonth', 'DayOfWeek', 'DayOfMonth', 'Quarter',
            'IsWeekend', 'MonthSin', 'MonthCos', 'DayOfYearSin', 'DayOfYearCos',
            'LocationCode'
        ]
        
        # Add lag features if they exist
        lag_columns = [col for col in data.columns if 'Lag' in col or 'MA' in col or 'Momentum' in col or 'Volatility' in col]
        feature_columns.extend(lag_columns)
        
        # Ensure all feature columns exist in the data
        available_features = [col for col in feature_columns if col in data.columns]
        
        if not available_features:
            print("Error: None of the specified feature columns found in the data")
            print(f"Available columns: {data.columns.tolist()}")
            return None, None, None, None
        
        target_column = 'Price'
        
        if target_column not in data.columns:
            print(f"Error: Target column '{target_column}' not found in data")
            print(f"Available columns: {data.columns.tolist()}")
            return None, None, None, None
        
        # Prepare the features and target
        X = data[available_features]
        y = data[target_column]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Create a scaler for numerical features
        scaler = StandardScaler()
        
        # Train the selected model with hyperparameter tuning
        if model_type == 'random_forest':
            # Define the model and parameter grid
            model = RandomForestRegressor(random_state=random_state)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
        elif model_type == 'gradient_boosting':
            # Define the model and parameter grid
            model = GradientBoostingRegressor(random_state=random_state)
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
        elif model_type == 'ridge':
            # Define the model and parameter grid
            model = Ridge(random_state=random_state)
            param_grid = {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
            
        elif model_type == 'lasso':
            # Define the model and parameter grid
            model = Lasso(random_state=random_state)
            param_grid = {
                'alpha': [0.1, 0.5, 1.0, 5.0]
            }
            
        else:
            print(f"Unsupported model type: {model_type}")
            return None, None, None, None
        
        # Scale the features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"\nTraining {model_type} model with grid search...")
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        # Train the model with the best parameters
        grid_search.fit(X_train_scaled, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Get feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(available_features, best_model.feature_importances_))
        elif hasattr(best_model, 'coef_'):
            feature_importance = dict(zip(available_features, best_model.coef_))
        else:
            feature_importance = {}
        
        # Create a pipeline that includes scaling
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', best_model)
        ])
        
        # Return the pipeline, test data, and feature importance
        return pipeline, X_test, y_test, feature_importance
    
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model
    
    Args:
        model: Trained model or pipeline
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target values
        
    Returns:
        dict: Evaluation metrics
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Create a dataframe for actual vs predicted values
        eval_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Difference': y_test - y_pred,
            'AbsoluteError': np.abs(y_test - y_pred),
            'PercentageError': np.abs((y_test - y_pred) / y_test) * 100
        })
        
        # Calculate additional metrics
        mean_percentage_error = eval_df['PercentageError'].mean()
        
        # Print evaluation metrics
        print("\nModel Evaluation:")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Percentage Error: {mean_percentage_error:.2f}%")
        
        # Print sample predictions
        print("\nSample Predictions:")
        print(eval_df.head())
        
        # Return the metrics
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mean_percentage_error': mean_percentage_error,
            'eval_df': eval_df
        }
    
    except Exception as e:
        print(f"Error evaluating model: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_model(model, model_path='banana_price_model.pkl'):
    """
    Save the trained model to a file
    
    Args:
        model: Trained model
        model_path (str): Path to save the model
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the data processing and model training pipeline"""
    # Load and process data
    print("Loading and processing data...")
    data = load_and_process_data()
    
    if data is None:
        print("Failed to process data. Exiting.")
        return
    
    # Train the model
    print("\nTraining model...")
    model, X_test, y_test, feature_importance = train_model(
        data, model_type='gradient_boosting'
    )
    
    if model is None:
        print("Failed to train model. Exiting.")
        return
    
    # Print feature importance
    print("\nFeature Importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance}")
    
    # Evaluate the model
    print("\nEvaluating model...")
    eval_metrics = evaluate_model(model, X_test, y_test)
    
    if eval_metrics is None:
        print("Failed to evaluate model. Exiting.")
        return
    
    # Save the model
    print("\nSaving model...")
    save_model(model)

if __name__ == "__main__":
    main()