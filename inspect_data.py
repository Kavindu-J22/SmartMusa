# inspect_data.py
import pandas as pd
import os
import numpy as np

def inspect_excel(file_path):
    """
    Inspect the structure and content of an Excel file
    
    Args:
        file_path (str): Path to the Excel file
    """
    try:
        # Try to read the Excel file
        print(f"Attempting to read: {file_path}")
        
        # First, check if file exists
        if not os.path.exists(file_path):
            print(f"File does not exist at: {file_path}")
            return False
        
        # Try to read the Excel file
        try:
            # First attempt - standard read
            data = pd.read_excel(file_path)
        except Exception as e:
            print(f"Standard read failed: {e}")
            try:
                # Second attempt - try with sheet_name=0
                data = pd.read_excel(file_path, sheet_name=0)
            except Exception as e:
                print(f"Read with sheet_name=0 failed: {e}")
                # Third attempt - try to list all sheets
                xl = pd.ExcelFile(file_path)
                sheets = xl.sheet_names
                print(f"Available sheets: {sheets}")
                if len(sheets) > 0:
                    data = pd.read_excel(file_path, sheet_name=sheets[0])
                else:
                    print("No sheets found in the Excel file.")
                    return False
        
        # Print basic information
        print("\nFile successfully read!")
        print(f"Shape: {data.shape}")
        
        # Print all column names
        print("\nColumns:")
        for i, col in enumerate(data.columns):
            print(f"{i}: {col}")
        
        # Show the first few rows
        print("\nFirst 5 rows:")
        print(data.head())
        
        # Show data types
        print("\nData types:")
        print(data.dtypes)
        
        # Check for missing values
        print("\nMissing values:")
        print(data.isnull().sum())
        
        # Show basic statistics for numeric columns
        print("\nBasic statistics:")
        print(data.describe())
        
        # Identify potential date columns
        date_columns = [col for col in data.columns if 'date' in str(col).lower() or 'day' in str(col).lower() 
                       or 'month' in str(col).lower() or 'year' in str(col).lower()]
        if date_columns:
            print("\nPotential date columns:")
            for col in date_columns:
                print(f"- {col}")
                print(data[col].head())
        
        # Identify potential price columns
        price_columns = [col for col in data.columns if 'price' in str(col).lower() or 'cost' in str(col).lower() 
                         or 'value' in str(col).lower() or 'rs' in str(col).lower()]
        if price_columns:
            print("\nPotential price columns:")
            for col in price_columns:
                print(f"- {col}")
                if pd.api.types.is_numeric_dtype(data[col]):
                    print(f"  Min: {data[col].min()}, Max: {data[col].max()}, Mean: {data[col].mean()}")
                print(data[col].head())
        
        # Identify potential location columns
        location_columns = [col for col in data.columns if 'location' in str(col).lower() or 'area' in str(col).lower() 
                           or 'region' in str(col).lower() or 'market' in str(col).lower() or 'place' in str(col).lower()]
        if location_columns:
            print("\nPotential location columns:")
            for col in location_columns:
                print(f"- {col}")
                if pd.api.types.is_string_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
                    unique_values = data[col].unique()
                    print(f"  Unique values ({len(unique_values)}): {unique_values[:10]}")
                    if len(unique_values) > 10:
                        print(f"  ... and {len(unique_values) - 10} more")
                print(data[col].head())
        
        # Analyze numeric columns more deeply
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            print("\nNumeric columns analysis:")
            for col in numeric_columns:
                print(f"\n- {col}:")
                # Basic stats
                stats = data[col].describe()
                print(f"  Count: {stats['count']}")
                print(f"  Mean: {stats['mean']}")
                print(f"  Std: {stats['std']}")
                print(f"  Min: {stats['min']}")
                print(f"  25%: {stats['25%']}")
                print(f"  50%: {stats['50%']}")
                print(f"  75%: {stats['75%']}")
                print(f"  Max: {stats['max']}")
                
                # Check for outliers
                q1 = stats['25%']
                q3 = stats['75%']
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
                outlier_pct = len(outliers) / len(data) * 100
                print(f"  Outliers: {len(outliers)} ({outlier_pct:.2f}%)")
        
        return True
    
    except Exception as e:
        print(f"Error inspecting Excel file: {e}")
        return False

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try different possible file paths
    file_paths = [
        os.path.join(script_dir, 'weekly retail price of Ambulslit.xlsx'),  # Same directory as script
        os.path.join(script_dir, 'data', 'weekly retail price of Ambulslit.xlsx'),  # In data subfolder
        'weekly retail price of Ambulslit.xlsx'  # Relative to working directory
    ]
    
    success = False
    for path in file_paths:
        print(f"\nTrying path: {path}")
        if inspect_excel(path):
            success = True
            break
    
    if not success:
        print("\nCould not find or read the Excel file. Please check the file location and format.")
        
        # Ask user for manual file path
        user_path = input("\nPlease enter the full path to your Excel file: ")
        if user_path and os.path.exists(user_path):
            inspect_excel(user_path)
        else:
            print("Invalid path or file does not exist.")

if __name__ == "__main__":
    main()