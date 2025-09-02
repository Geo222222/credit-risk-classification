"""
Evaluation script with comprehensive metrics and visualizations.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, accuracy_score
)
from src.data import load_and_split_data
from src.utils import load_config, ensure_dir, CostMatrix


def load_trained_artifacts():
    """Load trained model and optimal threshold."""
    config = load_config()
    output_dir = config['paths']['output_dir']
    
    # Load model
    model_path = Path(output_dir) / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    
    pipeline = joblib.load(model_path)
    
    # Load threshold
    threshold_path = Path(output_dir) / "threshold.txt"
    if not threshold_path.exists():
        raise FileNotFoundError(f"Optimal threshold not found at {threshold_path}")
    
    with open(threshold_path, 'r') as f:
        optimal_threshold = float(f.read().strip())
    
    return pipeline, optimal_threshold


def evaluate_model():
    """Evaluate the trained model and generate comprehensive reports."""
    print("Starting model evaluation...")
    
    # Load configuration
    config = load_config()
    reports_dir = config['paths']['reports_dir']
    ensure_dir(reports_dir)
    
    # Load data
    print("Loading test data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
    
    # Load trained artifacts
    print("Loading trained model and threshold...")
    pipeline, optimal_threshold = load_trained_artifacts()
    
    # Make predictions
    print("Generating predictions...")
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    y_test_pred_default = pipeline.predict(X_test)  # Default threshold (0.5)
    y_test_pred_optimal = (y_test_proba >= optimal_threshold).astype(int)
    
    # Initialize cost matrix
    cost_matrix = CostMatrix(
        fn_cost=config['cost_matrix']['false_negative_cost'],
        fp_cost=config['cost_matrix']['false_positive_cost']
    )
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = {}
    
    # Basic metrics with default threshold
    metrics['default_threshold'] = {
        'threshold': 0.5,
        'accuracy': float(accuracy_score(y_test, y_test_pred_default)),
        'roc_auc': float(roc_auc_score(y_test, y_test_proba)),
        'expected_loss': float(cost_matrix.expected_loss(y_test, y_test_pred_default))
    }

    # Metrics with optimal threshold
    metrics['optimal_threshold'] = {
        'threshold': float(optimal_threshold),
        'accuracy': float(accuracy_score(y_test, y_test_pred_optimal)),
        'roc_auc': float(roc_auc_score(y_test, y_test_proba)),
        'expected_loss': float(cost_matrix.expected_loss(y_test, y_test_pred_optimal))
    }
    
    # Classification reports (convert numpy types to native Python types)
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    metrics['classification_report_default'] = convert_numpy_types(
        classification_report(y_test, y_test_pred_default, output_dict=True)
    )
    metrics['classification_report_optimal'] = convert_numpy_types(
        classification_report(y_test, y_test_pred_optimal, output_dict=True)
    )
    
    # Save metrics to JSON
    metrics_path = Path(reports_dir) / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # 1. Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Default threshold confusion matrix
    cm_default = confusion_matrix(y_test, y_test_pred_default)
    sns.heatmap(cm_default, annot=True, fmt='d', ax=axes[0], cmap='Blues')
    axes[0].set_title('Confusion Matrix (Default Threshold = 0.5)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Optimal threshold confusion matrix
    cm_optimal = confusion_matrix(y_test, y_test_pred_optimal)
    sns.heatmap(cm_optimal, annot=True, fmt='d', ax=axes[1], cmap='Blues')
    axes[1].set_title(f'Confusion Matrix (Optimal Threshold = {optimal_threshold:.3f})')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(Path(reports_dir) / "confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_test_proba):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(Path(reports_dir) / "roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Score distributions
    plt.figure(figsize=(10, 6))
    plt.hist(y_test_proba[y_test == 0], bins=50, alpha=0.7, label='Class 0 (No Default)', density=True)
    plt.hist(y_test_proba[y_test == 1], bins=50, alpha=0.7, label='Class 1 (Default)', density=True)
    plt.axvline(0.5, color='red', linestyle='--', label='Default Threshold (0.5)')
    plt.axvline(optimal_threshold, color='green', linestyle='--', 
                label=f'Optimal Threshold ({optimal_threshold:.3f})')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Score Distributions by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(Path(reports_dir) / "score_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Test set size: {len(y_test)} samples")
    print(f"Class distribution: {dict(pd.Series(y_test).value_counts())}")
    print()
    print("PERFORMANCE COMPARISON:")
    print(f"Default Threshold (0.5):")
    print(f"  Accuracy: {metrics['default_threshold']['accuracy']:.4f}")
    print(f"  Expected Loss: ${metrics['default_threshold']['expected_loss']:,.2f}")
    print()
    print(f"Optimal Threshold ({optimal_threshold:.3f}):")
    print(f"  Accuracy: {metrics['optimal_threshold']['accuracy']:.4f}")
    print(f"  Expected Loss: ${metrics['optimal_threshold']['expected_loss']:,.2f}")
    print()
    print(f"ROC AUC Score: {metrics['optimal_threshold']['roc_auc']:.4f}")
    print()
    print(f"Reports and visualizations saved in: {reports_dir}")
    print("="*60)


if __name__ == "__main__":
    evaluate_model()
