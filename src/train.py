"""
Training script with cost-aware threshold optimization.
"""

import joblib
from pathlib import Path
from src.data import load_and_split_data
from src.model import create_pipeline, get_feature_importance
from src.utils import load_config, ensure_dir, CostMatrix


def train_model():
    """Train the ML model and optimize decision threshold."""
    print("Starting training pipeline...")
    
    # Load configuration
    config = load_config()
    
    # Ensure output directory exists
    output_dir = config['paths']['output_dir']
    ensure_dir(output_dir)
    
    # Load and split data
    print("Loading and splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
    
    # Create and train pipeline
    print("Creating and training model pipeline...")
    pipeline = create_pipeline(X_train, config['model']['type'])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    print("Model training completed!")
    
    # Get predictions and probabilities on validation set
    print("Optimizing decision threshold on validation set...")
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]  # Probability of positive class
    
    # Initialize cost matrix
    cost_matrix = CostMatrix(
        fn_cost=config['cost_matrix']['false_negative_cost'],
        fp_cost=config['cost_matrix']['false_positive_cost']
    )
    
    # Find optimal threshold
    optimal_threshold = cost_matrix.optimal_threshold(y_val, y_val_proba)
    print(f"Optimal threshold found: {optimal_threshold:.4f}")
    
    # Save trained model
    model_path = Path(output_dir) / "model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save optimal threshold
    threshold_path = Path(output_dir) / "threshold.txt"
    with open(threshold_path, 'w') as f:
        f.write(str(optimal_threshold))
    print(f"Optimal threshold saved to: {threshold_path}")
    
    # Extract and display feature importance
    print("\nFeature Importance:")
    feature_importance = get_feature_importance(pipeline, X_train.columns.tolist())
    for feature, importance in list(feature_importance.items())[:10]:  # Top 10
        print(f"  {feature}: {importance:.4f}")
    
    # Validation set performance with optimal threshold
    y_val_pred_optimal = (y_val_proba >= optimal_threshold).astype(int)
    val_loss = cost_matrix.expected_loss(y_val, y_val_pred_optimal)
    
    print(f"\nValidation set performance:")
    print(f"  Expected loss with optimal threshold: ${val_loss:,.2f}")
    print(f"  Validation accuracy: {(y_val == y_val_pred_optimal).mean():.4f}")
    
    print("\nTraining pipeline completed successfully!")
    print(f"Artifacts saved in: {output_dir}")
    
    return pipeline, optimal_threshold


if __name__ == "__main__":
    train_model()
