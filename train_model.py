# train_model.py
from data_processing import load_and_process_data, train_model, evaluate_model, save_model
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns

# Set Seaborn style for better visualizations
sns.set(style="whitegrid")

def visualize_predictions(y_test, y_pred, save_path=None, title="Actual vs Predicted Banana Prices"):
    """
    Create an improved visualization comparing actual vs predicted prices
    
    Args:
        y_test: Actual prices from test set
        y_pred: Predicted prices from model
        save_path: Path to save the figure (optional)
        title: Plot title
    """
    # Calculate metrics
    mae = np.mean(np.abs(y_test - y_pred))
    mse = np.mean((y_test - y_pred)**2)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create the scatter plot with improved styling
    scatter = plt.scatter(y_test, y_pred, alpha=0.7, s=80, c='#1f77b4', edgecolors='w', linewidths=0.5)
    
    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred)) * 0.9
    max_val = max(max(y_test), max(y_pred)) * 1.1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')
    
    # Set axis limits with some padding
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Add titles and labels with improved styling
    plt.title(title, fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('Actual Price (LKR)', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Price (LKR)', fontsize=14, fontweight='bold')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add metrics text box with better styling
    plt.text(0.05, 0.95, 
             f'MAE: {mae:.2f} LKR\nRMSE: {rmse:.2f} LKR\nR²: {r2:.3f}', 
             transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#cccccc'))
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Ensure tight layout
    plt.tight_layout()
    
    # Save if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    # Show the plot
    plt.show()

def create_residual_plot(y_test, y_pred, save_path=None):
    """
    Create a residual plot to visualize prediction errors
    
    Args:
        y_test: Actual prices from test set
        y_pred: Predicted prices from model
        save_path: Path to save the figure (optional)
    """
    # Calculate residuals
    residuals = y_test - y_pred
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create residual scatter plot
    plt.scatter(y_pred, residuals, alpha=0.7, s=80, c='#2ca02c', edgecolors='w', linewidths=0.5)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.8)
    
    # Add titles and labels
    plt.title('Residual Plot (Prediction Errors)', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('Predicted Price (LKR)', fontsize=14, fontweight='bold')
    plt.ylabel('Residual (Actual - Predicted) (LKR)', fontsize=14, fontweight='bold')
    
    # Calculate residual statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # Add statistics text box
    plt.text(0.05, 0.95, 
             f'Mean Residual: {mean_residual:.2f} LKR\nStd Deviation: {std_residual:.2f} LKR', 
             transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#cccccc'))
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure tight layout
    plt.tight_layout()
    
    # Save if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual plot saved to {save_path}")
    
    # Show the plot
    plt.show()

def create_feature_importance_plot(feature_importance, save_path=None):
    """
    Create a feature importance plot
    
    Args:
        feature_importance: Dictionary of feature names and their importance values
        save_path: Path to save the figure (optional)
    """
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features = [x[0] for x in sorted_features]
    importance = [x[1] for x in sorted_features]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart
    bars = plt.barh(features, importance, color='#ff7f0e', alpha=0.8, edgecolor='k', linewidth=0.5)
    
    # Add titles and labels
    plt.title('Feature Importance', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('Importance', fontsize=14, fontweight='bold')
    plt.ylabel('Feature', fontsize=14, fontweight='bold')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    # Ensure tight layout
    plt.tight_layout()
    
    # Save if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    # Show the plot
    plt.show()

def create_error_distribution_plot(y_test, y_pred, save_path=None):
    """
    Create a histogram of prediction errors
    
    Args:
        y_test: Actual prices from test set
        y_pred: Predicted prices from model
        save_path: Path to save the figure (optional)
    """
    # Calculate errors
    errors = y_test - y_pred
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create histogram with KDE
    sns.histplot(errors, kde=True, bins=20, color='#9467bd', alpha=0.7, edgecolor='w', linewidth=0.5)
    
    # Add vertical line at x=0
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, alpha=0.8)
    
    # Add titles and labels
    plt.title('Distribution of Prediction Errors', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('Error (Actual - Predicted) (LKR)', fontsize=14, fontweight='bold')
    plt.ylabel('Frequency', fontsize=14, fontweight='bold')
    
    # Calculate error statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)
    
    # Add statistics text box
    plt.text(0.05, 0.95, 
             f'Mean Error: {mean_error:.2f} LKR\nMedian Error: {median_error:.2f} LKR\nStd Deviation: {std_error:.2f} LKR', 
             transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#cccccc'))
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure tight layout
    plt.tight_layout()
    
    # Save if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution plot saved to {save_path}")
    
    # Show the plot
    plt.show()

def main():
    print("SmartMusa - Improved Model Training")
    print("----------------------------------")
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create a visualizations directory if it doesn't exist
    vis_dir = os.path.join(script_dir, 'visualizations')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
        print(f"Created visualizations directory at {vis_dir}")
    
    # Try different possible file paths
    file_paths = [
        os.path.join(script_dir, 'weekly retail price of Ambulslit.xlsx'),  # Same directory as script
        os.path.join(script_dir, 'data', 'weekly retail price of Ambulslit.xlsx'),  # In data subfolder
        'weekly retail price of Ambulslit.xlsx'  # Relative to working directory
    ]
    
    # Try each path until we find the file
    data = None
    for file_path in file_paths:
        print(f"\nAttempting to load data from: {file_path}")
        if os.path.exists(file_path):
            print(f"File found at: {file_path}")
            data = load_and_process_data(file_path)
            if data is not None:
                break
        else:
            print(f"File not found at: {file_path}")
    
    if data is None:
        print("\nFailed to process data. Please check the file path and format. Exiting.")
        return
    
    # Train different models and select the best one
    print("\nTraining multiple models to find the best one...")
    
    model_types = ['gradient_boosting', 'random_forest', 'ridge', 'lasso']
    best_model = None
    best_r2 = -float('inf')
    best_model_type = None
    best_test_data = None
    best_feature_importance = None
    
    for model_type in model_types:
        print(f"\nTraining {model_type} model...")
        model, X_test, y_test, feature_importance = train_model(
            data, 
            model_type=model_type,
            test_size=0.2
        )
        
        if model is None:
            print(f"Failed to train {model_type} model. Skipping.")
            continue
        
        # Evaluate the model
        print(f"Evaluating {model_type} model...")
        eval_metrics = evaluate_model(model, X_test, y_test)
        
        if eval_metrics is None:
            print(f"Failed to evaluate {model_type} model. Skipping.")
            continue
        
        # Check if this model is better than the current best
        if eval_metrics['r2'] > best_r2:
            best_r2 = eval_metrics['r2']
            best_model = model
            best_model_type = model_type
            best_test_data = (X_test, y_test)
            best_feature_importance = feature_importance
    
    if best_model is None:
        print("No successful model was trained. Exiting.")
        return
    
    print(f"\nBest model: {best_model_type} (R² = {best_r2:.4f})")
    
    # Get predictions from the best model
    X_test, y_test = best_test_data
    y_pred = best_model.predict(X_test)
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create and save visualizations
    print("\nGenerating visualizations...")
    
    # Actual vs Predicted scatter plot
    scatter_path = os.path.join(vis_dir, f'actual_vs_predicted_{timestamp}.png')
    visualize_predictions(y_test, y_pred, save_path=scatter_path)
    
    # Residual plot
    residual_path = os.path.join(vis_dir, f'residual_plot_{timestamp}.png')
    create_residual_plot(y_test, y_pred, save_path=residual_path)
    
    # Feature importance plot
    if best_feature_importance:
        importance_path = os.path.join(vis_dir, f'feature_importance_{timestamp}.png')
        create_feature_importance_plot(best_feature_importance, save_path=importance_path)
    
    # Error distribution plot
    error_path = os.path.join(vis_dir, f'error_distribution_{timestamp}.png')
    create_error_distribution_plot(y_test, y_pred, save_path=error_path)
    
    # Save the best model
    print("\nSaving the best model...")
    save_model(best_model, 'banana_price_model.pkl')
    
    print("\nTraining complete. The improved model is ready for use.")
    print(f"\nVisualizations have been saved to the '{vis_dir}' directory.")

if __name__ == "__main__":
    main()