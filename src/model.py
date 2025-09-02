"""
Model pipeline construction with preprocessing.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from src.utils import load_config, set_random_seed


def create_preprocessor(X):
    """
    Create preprocessing pipeline for features.
    
    Args:
        X (DataFrame): Feature data to determine preprocessing steps
        
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    # For simplicity, assume all features are numeric and apply standard scaling
    numeric_features = X.columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor


def create_model(model_type="logistic_regression", random_seed=42):
    """
    Create the specified model.
    
    Args:
        model_type (str): Type of model to create
        random_seed (int): Random seed for reproducibility
        
    Returns:
        sklearn estimator: The specified model
    """
    if model_type == "logistic_regression":
        return LogisticRegression(
            random_state=random_seed,
            max_iter=1000,
            solver='liblinear'  # Good for small datasets
        )
    elif model_type == "histogram_gradient_boosting":
        return HistGradientBoostingClassifier(
            random_state=random_seed,
            max_iter=100,
            learning_rate=0.1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_pipeline(X_train, model_type=None):
    """
    Create complete ML pipeline with preprocessing and model.
    
    Args:
        X_train (DataFrame): Training features for preprocessing setup
        model_type (str, optional): Model type, loads from config if None
        
    Returns:
        Pipeline: Complete ML pipeline
    """
    config = load_config()
    set_random_seed(config['random_seed'])
    
    if model_type is None:
        model_type = config['model']['type']
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessor(X_train)
    
    # Create model
    model = create_model(model_type, config['random_seed'])
    
    # Combine into pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    print(f"Created pipeline with {model_type} model")
    
    return pipeline


def get_feature_importance(pipeline, feature_names):
    """
    Extract feature importance from trained pipeline.
    
    Args:
        pipeline (Pipeline): Trained pipeline
        feature_names (list): List of feature names
        
    Returns:
        dict: Feature importance scores
    """
    try:
        # Get the trained classifier
        classifier = pipeline.named_steps['classifier']
        
        if hasattr(classifier, 'feature_importances_'):
            # Tree-based models
            importances = classifier.feature_importances_
        elif hasattr(classifier, 'coef_'):
            # Linear models - use absolute values of coefficients
            importances = np.abs(classifier.coef_[0])
        else:
            print("Model does not support feature importance extraction")
            return {}
        
        # Create feature importance dictionary
        feature_importance = dict(zip(feature_names, importances))
        
        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return feature_importance
        
    except Exception as e:
        print(f"Error extracting feature importance: {e}")
        return {}
