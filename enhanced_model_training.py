# enhanced_model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import joblib
import warnings
from datetime import datetime, timedelta
import os
warnings.filterwarnings('ignore')

# Install required packages if not available
try:
    import xgboost as xgb
except ImportError:
    print("Installing XGBoost...")
    os.system("pip install xgboost")
    import xgboost as xgb

try:
    import lightgbm as lgb
except ImportError:
    print("Installing LightGBM...")
    os.system("pip install lightgbm")
    import lightgbm as lgb

class EnhancedModelTrainer:
    """
    Enhanced model training with ensemble methods and advanced techniques
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importance = {}
        self.scaler = None
        self.feature_selector = None
        
    def train_ensemble_model(self, data, test_size=0.2, random_state=42):
        """
        Train multiple models and create an ensemble
        """
        try:
            # Prepare features and target
            feature_columns = [col for col in data.columns if col != 'Price']
            X = data[feature_columns]
            y = data['Price']
            
            print(f"Training with {len(feature_columns)} features and {len(data)} samples")
            
            # Time-series aware split
            if 'Date' in data.columns:
                # Sort by date and use temporal split
                data_sorted = data.sort_values('Date')
                split_idx = int(len(data_sorted) * (1 - test_size))
                X_train = data_sorted[feature_columns].iloc[:split_idx]
                X_test = data_sorted[feature_columns].iloc[split_idx:]
                y_train = data_sorted['Price'].iloc[:split_idx]
                y_test = data_sorted['Price'].iloc[split_idx:]
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
            
            # Feature selection
            print("Performing feature selection...")
            selector = SelectKBest(score_func=f_regression, k=min(50, len(feature_columns)))
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
            print(f"Selected {len(selected_features)} features")
            
            # Scaling
            scaler = RobustScaler()  # More robust to outliers
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # Store for later use
            self.scaler = scaler
            self.feature_selector = selector
            
            # Define models
            models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=200, max_depth=15, min_samples_split=5,
                    min_samples_leaf=2, random_state=random_state, n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=200, learning_rate=0.1, max_depth=6,
                    min_samples_split=5, min_samples_leaf=2, random_state=random_state
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=200, learning_rate=0.1, max_depth=6,
                    min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                    random_state=random_state, n_jobs=-1
                ),
                'lightgbm': lgb.LGBMRegressor(
                    n_estimators=200, learning_rate=0.1, max_depth=6,
                    min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
                    random_state=random_state, n_jobs=-1, verbose=-1
                ),
                'extra_trees': ExtraTreesRegressor(
                    n_estimators=200, max_depth=15, min_samples_split=5,
                    min_samples_leaf=2, random_state=random_state, n_jobs=-1
                ),
                'ridge': Ridge(alpha=1.0),
                'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=random_state)
            }
            
            # Train individual models
            trained_models = {}
            model_scores = {}
            
            print("\nTraining individual models...")
            for name, model in models.items():
                print(f"Training {name}...")
                
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test_scaled)
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    trained_models[name] = model
                    model_scores[name] = {'r2': r2, 'rmse': rmse, 'mae': mae}
                    
                    print(f"{name} - R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
                    
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    continue
            
            # Create ensemble
            print("\nCreating ensemble model...")
            ensemble_models = [(name, model) for name, model in trained_models.items() 
                             if name in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']]
            
            if len(ensemble_models) >= 2:
                ensemble = VotingRegressor(estimators=ensemble_models)
                ensemble.fit(X_train_scaled, y_train)
                
                # Evaluate ensemble
                y_pred_ensemble = ensemble.predict(X_test_scaled)
                r2_ensemble = r2_score(y_test, y_pred_ensemble)
                rmse_ensemble = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
                mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
                
                model_scores['ensemble'] = {'r2': r2_ensemble, 'rmse': rmse_ensemble, 'mae': mae_ensemble}
                trained_models['ensemble'] = ensemble
                
                print(f"Ensemble - R²: {r2_ensemble:.4f}, RMSE: {rmse_ensemble:.2f}, MAE: {mae_ensemble:.2f}")
            
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
            elif hasattr(self.best_model, 'coef_'):
                self.feature_importance = dict(zip(selected_features, self.best_model.coef_))
            
            return pipeline, X_test, y_test, self.feature_importance, model_scores
            
        except Exception as e:
            print(f"Error in ensemble training: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None, None
    
    def hyperparameter_optimization(self, model, X_train, y_train, param_grid):
        """
        Perform hyperparameter optimization using GridSearchCV
        """
        try:
            # Use TimeSeriesSplit for time series data
            cv = TimeSeriesSplit(n_splits=5)
            
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            return grid_search.best_estimator_, grid_search.best_params_
            
        except Exception as e:
            print(f"Error in hyperparameter optimization: {e}")
            return model, {}
    
    def evaluate_model_comprehensive(self, model, X_test, y_test):
        """
        Comprehensive model evaluation with multiple metrics
        """
        try:
            y_pred = model.predict(X_test)
            
            # Basic metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Additional metrics
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            # Residual analysis
            residuals = y_test - y_pred
            residual_std = np.std(residuals)
            
            # Create evaluation dataframe
            eval_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred,
                'Residual': residuals,
                'AbsoluteError': np.abs(residuals),
                'PercentageError': np.abs(residuals / y_test) * 100
            })
            
            metrics = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'residual_std': residual_std,
                'eval_df': eval_df
            }
            
            print("\nComprehensive Model Evaluation:")
            print(f"Mean Absolute Error (MAE): {mae:.2f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
            print(f"R² Score: {r2:.4f}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            print(f"Residual Standard Deviation: {residual_std:.2f}")
            
            return metrics
            
        except Exception as e:
            print(f"Error in model evaluation: {e}")
            return None
