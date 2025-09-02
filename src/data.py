"""
Data loading and preprocessing module with auto-discovery and synthetic fallback.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.utils import load_config, set_random_seed


def discover_data_files(data_dir="Resources"):
    """
    Auto-discover CSV files in the data directory.
    
    Args:
        data_dir (str): Directory to search for CSV files
        
    Returns:
        list: List of CSV file paths
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    
    csv_files = list(data_path.glob("*.csv"))
    return csv_files


def load_real_data(data_dir="Resources", target_column="loan_status"):
    """
    Load real data from CSV files in the data directory.
    
    Args:
        data_dir (str): Directory containing CSV files
        target_column (str): Name of the target column
        
    Returns:
        tuple: (X, y) features and target, or (None, None) if no data found
    """
    csv_files = discover_data_files(data_dir)
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return None, None
    
    # Use the first CSV file found
    data_file = csv_files[0]
    print(f"Loading data from: {data_file}")
    
    try:
        df = pd.read_csv(data_file)
        
        if target_column not in df.columns:
            print(f"Target column '{target_column}' not found in data")
            print(f"Available columns: {list(df.columns)}")
            return None, None
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        print(f"Data loaded successfully: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
        
    except Exception as e:
        print(f"Error loading data from {data_file}: {e}")
        return None, None


def generate_synthetic_data(config):
    """
    Generate synthetic credit risk data for demonstration.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (X, y) synthetic features and target
    """
    print("Generating synthetic credit risk data...")
    
    synthetic_config = config.get('synthetic_data', {})
    n_samples = synthetic_config.get('n_samples', 10000)
    n_features = synthetic_config.get('n_features', 7)
    n_informative = synthetic_config.get('n_informative', 5)
    n_redundant = synthetic_config.get('n_redundant', 2)
    flip_y = synthetic_config.get('flip_y', 0.1)
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=1,
        flip_y=flip_y,
        random_state=config['random_seed']
    )
    
    # Create feature names similar to credit risk features
    feature_names = [
        'loan_size', 'interest_rate', 'borrower_income', 
        'debt_to_income', 'num_of_accounts', 'derogatory_marks', 
        'total_debt'
    ][:n_features]
    
    # Pad with generic names if needed
    while len(feature_names) < n_features:
        feature_names.append(f'feature_{len(feature_names)}')
    
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name='loan_status')
    
    print(f"Synthetic data generated: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y


def load_and_split_data():
    """
    Load data (real or synthetic) and split into train/val/test sets.
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    config = load_config()
    set_random_seed(config['random_seed'])
    
    # Try to load real data first
    X, y = load_real_data(
        data_dir=config['paths']['data_dir'],
        target_column=config['data']['target_column']
    )
    
    # Fall back to synthetic data if real data not available
    if X is None or y is None:
        X, y = generate_synthetic_data(config)
    
    # Split data
    test_size = config['data']['test_size']
    val_size = config['data']['val_size']
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=config['random_seed'], stratify=y
    )
    
    # Second split: separate train and validation from remaining data
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=config['random_seed'], stratify=y_temp
    )
    
    print(f"Data split completed:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples") 
    print(f"  Test: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
