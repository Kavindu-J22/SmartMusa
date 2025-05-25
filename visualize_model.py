# visualize_model.py
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from data_processing import load_and_process_data

def create_scatter_plot(y_true, y_pred, title="Actual vs Predicted Prices", save_path=None):
    """
    Create a scatter plot comparing actual vs predicted values
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.7, color='blue')
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Add metrics text box
    plt.text(0.05, 0.95, 
             f'Mean Absolute Error: {mae:.2f} LKR\n'
             f'Root Mean Squared Error: {rmse:.2f} LKR\n'
             f'R² Score: {r2:.3f}', 
             transform=plt.gca().transAxes, 
             fontsize=12, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add labels and title
    plt.title(title, fontsize=16)
    plt.xlabel('Actual Price (LKR)', fontsize=14)
    plt.ylabel('Predicted Price (LKR)', fontsize=14)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Scatter plot saved to: {save_path}")
    
    # Display plot
    plt.show()

def create_residual_plot(y_true, y_pred, title="Residual Plot", save_path=None):
    """
    Create a residual plot to visualize prediction errors
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot of predicted values vs residuals
    plt.scatter(y_pred, residuals, alpha=0.7, color='green')
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    
    # Calculate residual statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # Add statistics text box
    plt.text(0.05, 0.95, 
             f'Mean Residual: {mean_residual:.2f} LKR\n'
             f'Std Deviation: {std_residual:.2f} LKR', 
             transform=plt.gca().transAxes, 
             fontsize=12, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add labels and title
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Price (LKR)', fontsize=14)
    plt.ylabel('Residual (Actual - Predicted) (LKR)', fontsize=14)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Residual plot saved to: {save_path}")
    
    # Display plot
    plt.show()

def create_time_series_plot(dates, y_true, y_pred, title="Banana Prices Over Time", save_path=None):
    """
    Create a time series plot comparing actual and predicted values over time
    
    Args:
        dates: Date values
        y_true: Actual values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(14, 8))
    
    # Create line plots
    plt.plot(dates, y_true, 'b-', label='Actual Price', linewidth=2)
    plt.plot(dates, y_pred, 'r--', label='Predicted Price', linewidth=2)
    
    # Add labels and title
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price (LKR)', fontsize=14)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Format date axis
    plt.gcf().autofmt_xdate()
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Ensure tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Time series plot saved to: {save_path}")
    
    # Display plot
    plt.show()

def create_prediction_error_histogram(y_true, y_pred, title="Distribution of Prediction Errors", save_path=None):
    """
    Create a histogram showing the distribution of prediction errors
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    errors = y_true - y_pred
    
    plt.figure(figsize=(12, 8))
    
    # Create histogram
    n, bins, patches = plt.hist(errors, bins=20, alpha=0.7, color='purple')
    
    # Add vertical line at x=0 (perfect prediction)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    
    # Calculate error statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)
    
    # Add statistics text box
    plt.text(0.05, 0.95, 
             f'Mean Error: {mean_error:.2f} LKR\n'
             f'Median Error: {median_error:.2f} LKR\n'
             f'Std Deviation: {std_error:.2f} LKR', 
             transform=plt.gca().transAxes, 
             fontsize=12, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add labels and title
    plt.title(title, fontsize=16)
    plt.xlabel('Prediction Error (Actual - Predicted) (LKR)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Ensure tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Histogram saved to: {save_path}")
    
    # Display plot
    plt.show()

def visualize_model_performance(data_path=None, model_path='banana_price_model.pkl'):
    """
    Main function to visualize model performance
    
    Args:
        data_path: Path to the data file (Excel)
        model_path: Path to the trained model
    """
    # Create visualizations directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vis_dir = os.path.join(script_dir, 'visualizations')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
        print(f"Created visualizations directory at {vis_dir}")
    
    # Get current timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load the model
    try:
        print("Loading the trained model...")
        model = joblib.load(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Determine data path if not provided
    if data_path is None:
        file_paths = [
            os.path.join(script_dir, 'weekly retail price of Ambulslit.xlsx'),
            os.path.join(script_dir, 'data', 'weekly retail price of Ambulslit.xlsx'),
            'weekly retail price of Ambulslit.xlsx'
        ]
        
        for path in file_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            print("Could not find data file. Please specify the path.")
            return
    
    # Load and process data
    print(f"Loading data from {data_path}...")
    data = load_and_process_data(data_path)
    
    if data is None:
        print("Failed to process data. Exiting.")
        return
    
    # Prepare features and target
    feature_columns = ['Month', 'WeekOfMonth', 'DayOfWeek', 'LocationCode']
    available_features = [col for col in feature_columns if col in data.columns]
    
    if not available_features:
        print("Error: No feature columns found in the data")
        return
    
    if 'Price' not in data.columns:
        print("Error: 'Price' column not found in the data")
        return
    
    X = data[available_features]
    y_true = data['Price']
    
    # Generate predictions
    print("Generating predictions...")
    y_pred = model.predict(X)
    
    # Create visualizations
    print("\nCreating scatter plot...")
    scatter_path = os.path.join(vis_dir, f'actual_vs_predicted_{timestamp}.png')
    create_scatter_plot(y_true, y_pred, save_path=scatter_path)
    
    print("\nCreating residual plot...")
    residual_path = os.path.join(vis_dir, f'residual_plot_{timestamp}.png')
    create_residual_plot(y_true, y_pred, save_path=residual_path)
    
    print("\nCreating prediction error histogram...")
    hist_path = os.path.join(vis_dir, f'error_histogram_{timestamp}.png')
    create_prediction_error_histogram(y_true, y_pred, save_path=hist_path)
    
    # Create time series plot if date column exists
    if 'Date' in data.columns:
        print("\nCreating time series plot...")
        time_series_path = os.path.join(vis_dir, f'time_series_{timestamp}.png')
        create_time_series_plot(data['Date'], y_true, y_pred, save_path=time_series_path)
    
    print(f"\nAll visualizations have been saved to: {vis_dir}")

if __name__ == "__main__":
    visualize_model_performance()