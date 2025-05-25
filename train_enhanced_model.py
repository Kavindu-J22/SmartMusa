# train_enhanced_model.py
"""
Enhanced model training script with advanced features and ensemble methods
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced modules
from enhanced_data_processing import EnhancedBananaPricePredictor
from enhanced_model_training import EnhancedModelTrainer

def main():
    """
    Main function to train enhanced banana price prediction model
    """
    print("🍌 SmartMusa Enhanced Model Training")
    print("=" * 50)
    
    # Initialize components
    data_processor = EnhancedBananaPricePredictor()
    model_trainer = EnhancedModelTrainer()
    
    # Step 1: Load and process data
    print("\n📊 Step 1: Loading and processing data...")
    
    # Try different data file paths
    data_files = [
        'weekly retail price of Ambulslit.xlsx',
        'data/weekly retail price of Ambulslit.xlsx',
        'data\\weekly retail price of Ambulslit.xlsx'
    ]
    
    data = None
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"Found data file: {file_path}")
            data = data_processor.load_and_process_data(file_path)
            break
    
    if data is None:
        print("❌ No data file found. Please ensure the data file exists.")
        print("Expected files:")
        for file_path in data_files:
            print(f"  - {file_path}")
        return False
    
    print(f"✅ Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Step 2: Train ensemble model
    print("\n🤖 Step 2: Training ensemble model...")
    
    try:
        pipeline, X_test, y_test, feature_importance, model_scores = model_trainer.train_ensemble_model(
            data, test_size=0.2, random_state=42
        )
        
        if pipeline is None:
            print("❌ Model training failed")
            return False
        
        print("✅ Model training completed successfully!")
        
        # Step 3: Evaluate model
        print("\n📈 Step 3: Evaluating model performance...")
        
        evaluation_metrics = model_trainer.evaluate_model_comprehensive(pipeline, X_test, y_test)
        
        if evaluation_metrics:
            print("\n🎯 Model Performance Summary:")
            print(f"  • R² Score: {evaluation_metrics['r2']:.4f}")
            print(f"  • RMSE: {evaluation_metrics['rmse']:.2f}")
            print(f"  • MAE: {evaluation_metrics['mae']:.2f}")
            print(f"  • MAPE: {evaluation_metrics['mape']:.2f}%")
            
            # Performance interpretation
            if evaluation_metrics['r2'] > 0.8:
                print("🌟 Excellent model performance!")
            elif evaluation_metrics['r2'] > 0.6:
                print("👍 Good model performance!")
            elif evaluation_metrics['r2'] > 0.4:
                print("⚠️ Moderate model performance - consider more data or features")
            else:
                print("❌ Poor model performance - needs significant improvement")
        
        # Step 4: Display feature importance
        if feature_importance:
            print("\n🔍 Top 10 Most Important Features:")
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                print(f"  {i:2d}. {feature:<25} {importance:8.4f}")
        
        # Step 5: Display model comparison
        if model_scores:
            print("\n📊 Individual Model Performance:")
            for model_name, scores in model_scores.items():
                print(f"  • {model_name:<15} R²: {scores['r2']:.4f}, RMSE: {scores['rmse']:.2f}")
        
        # Step 6: Save the model
        print("\n💾 Step 4: Saving enhanced model...")
        
        try:
            import joblib
            model_path = 'enhanced_banana_price_model.pkl'
            joblib.dump(pipeline, model_path)
            print(f"✅ Enhanced model saved to: {model_path}")
            
            # Also save as the default model for the app
            default_model_path = 'banana_price_model.pkl'
            joblib.dump(pipeline, default_model_path)
            print(f"✅ Model also saved as: {default_model_path}")
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
        
        # Step 7: Generate model report
        print("\n📋 Step 5: Generating model report...")
        
        report_path = generate_model_report(
            evaluation_metrics, feature_importance, model_scores, data.shape
        )
        
        if report_path:
            print(f"✅ Model report saved to: {report_path}")
        
        print("\n🎉 Enhanced model training completed successfully!")
        print("\nNext steps:")
        print("  1. Test the model with the enhanced chatbot")
        print("  2. Monitor model performance in production")
        print("  3. Retrain periodically with new data")
        print("  4. Consider A/B testing with the previous model")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_model_report(evaluation_metrics, feature_importance, model_scores, data_shape):
    """
    Generate a comprehensive model report
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"model_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("SmartMusa Enhanced Model Training Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Shape: {data_shape[0]} rows, {data_shape[1]} columns\n\n")
            
            # Model Performance
            f.write("MODEL PERFORMANCE\n")
            f.write("-" * 20 + "\n")
            if evaluation_metrics:
                f.write(f"R² Score: {evaluation_metrics['r2']:.4f}\n")
                f.write(f"RMSE: {evaluation_metrics['rmse']:.2f}\n")
                f.write(f"MAE: {evaluation_metrics['mae']:.2f}\n")
                f.write(f"MAPE: {evaluation_metrics['mape']:.2f}%\n")
                f.write(f"Residual Std: {evaluation_metrics['residual_std']:.2f}\n\n")
            
            # Feature Importance
            f.write("FEATURE IMPORTANCE\n")
            f.write("-" * 20 + "\n")
            if feature_importance:
                sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
                for i, (feature, importance) in enumerate(sorted_features, 1):
                    f.write(f"{i:2d}. {feature:<30} {importance:8.4f}\n")
                f.write("\n")
            
            # Model Comparison
            f.write("MODEL COMPARISON\n")
            f.write("-" * 20 + "\n")
            if model_scores:
                for model_name, scores in model_scores.items():
                    f.write(f"{model_name:<20} R²: {scores['r2']:.4f}, RMSE: {scores['rmse']:.2f}, MAE: {scores['mae']:.2f}\n")
                f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            if evaluation_metrics:
                if evaluation_metrics['r2'] > 0.8:
                    f.write("• Excellent model performance - ready for production\n")
                    f.write("• Consider deploying with confidence\n")
                elif evaluation_metrics['r2'] > 0.6:
                    f.write("• Good model performance - suitable for production\n")
                    f.write("• Monitor performance and collect more data\n")
                else:
                    f.write("• Model needs improvement\n")
                    f.write("• Consider collecting more data or additional features\n")
                    f.write("• Review feature engineering strategies\n")
            
            f.write("\n")
            f.write("• Retrain model monthly with new data\n")
            f.write("• Monitor prediction accuracy in production\n")
            f.write("• Consider external data sources (weather, economic indicators)\n")
            f.write("• Implement A/B testing for model updates\n")
        
        return report_path
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return None

def test_enhanced_model():
    """
    Test the enhanced model with sample predictions
    """
    try:
        import joblib
        
        print("\n🧪 Testing Enhanced Model...")
        
        # Load the model
        model = joblib.load('enhanced_banana_price_model.pkl')
        
        # Test with sample data
        test_cases = [
            {"month": 1, "location": "colombo", "description": "January in Colombo"},
            {"month": 6, "location": "kandy", "description": "June in Kandy"},
            {"month": 12, "location": "galle", "description": "December in Galle"}
        ]
        
        print("\nSample Predictions:")
        for case in test_cases:
            # Create sample features (this would need to match your model's expected input)
            features = [case["month"], 1, 0, 1]  # month, week, day, location_code
            
            try:
                prediction = model.predict([features])[0]
                print(f"  • {case['description']}: {prediction:.2f} LKR/kg")
            except Exception as e:
                print(f"  • {case['description']}: Error - {e}")
        
        print("✅ Model testing completed")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")

if __name__ == "__main__":
    print("Starting Enhanced SmartMusa Model Training...")
    
    success = main()
    
    if success:
        # Test the model
        test_enhanced_model()
        
        print("\n🚀 Training completed successfully!")
        print("You can now use the enhanced model with the SmartMusa application.")
    else:
        print("\n❌ Training failed. Please check the errors above.")
        sys.exit(1)
