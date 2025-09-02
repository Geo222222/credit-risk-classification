"""
Basic integration smoke test for the ML pipeline.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, CostMatrix, set_random_seed
from src.data import generate_synthetic_data, load_and_split_data
from src.model import create_pipeline, get_feature_importance
from src.train import train_model
from src.eval import evaluate_model


def test_config_loading():
    """Test that configuration can be loaded."""
    config = load_config()
    assert 'random_seed' in config
    assert 'data' in config
    assert 'model' in config
    assert 'cost_matrix' in config
    print("✓ Config loading test passed")


def test_cost_matrix():
    """Test cost matrix functionality."""
    import numpy as np
    
    cost_matrix = CostMatrix(fn_cost=1000, fp_cost=100)
    
    # Test expected loss calculation
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1])  # 1 FN, 1 FP
    
    expected_loss = cost_matrix.expected_loss(y_true, y_pred)
    assert expected_loss == 1100  # 1*1000 + 1*100
    
    # Test optimal threshold finding
    y_proba = np.array([0.1, 0.9, 0.3, 0.7])
    threshold = cost_matrix.optimal_threshold(y_true, y_proba)
    assert 0 <= threshold <= 1
    
    print("✓ Cost matrix test passed")


def test_synthetic_data_generation():
    """Test synthetic data generation."""
    config = load_config()
    X, y = generate_synthetic_data(config)
    
    assert X.shape[0] > 0
    assert X.shape[1] > 0
    assert len(y) == X.shape[0]
    assert set(y.unique()) == {0, 1}
    
    print("✓ Synthetic data generation test passed")


def test_data_splitting():
    """Test data loading and splitting."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
    
    # Check that all splits have data
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0
    
    # Check that features match across splits
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    
    # Check that targets are binary
    for y in [y_train, y_val, y_test]:
        assert set(y.unique()).issubset({0, 1})
    
    print("✓ Data splitting test passed")


def test_pipeline_creation():
    """Test ML pipeline creation."""
    X_train, _, _, y_train, _, _ = load_and_split_data()
    
    # Test logistic regression pipeline
    pipeline_lr = create_pipeline(X_train, "logistic_regression")
    pipeline_lr.fit(X_train, y_train)
    
    predictions = pipeline_lr.predict(X_train)
    probabilities = pipeline_lr.predict_proba(X_train)
    
    assert len(predictions) == len(X_train)
    assert probabilities.shape == (len(X_train), 2)
    
    # Test feature importance extraction
    importance = get_feature_importance(pipeline_lr, X_train.columns.tolist())
    assert isinstance(importance, dict)
    
    print("✓ Pipeline creation test passed")


def test_full_pipeline_integration():
    """Test the complete pipeline integration."""
    # This test uses temporary directories to avoid interfering with actual artifacts
    
    # Create temporary config that uses temp directories
    import tempfile
    import yaml
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_artifacts = os.path.join(temp_dir, "artifacts")
        temp_reports = os.path.join(temp_dir, "reports")
        
        # Temporarily modify config paths
        config = load_config()
        original_output_dir = config['paths']['output_dir']
        original_reports_dir = config['paths']['reports_dir']
        
        config['paths']['output_dir'] = temp_artifacts
        config['paths']['reports_dir'] = temp_reports
        
        # Save temporary config
        temp_config_path = os.path.join(temp_dir, "temp_config.yaml")
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Override config loading for this test
        import src.utils
        original_load_config = src.utils.load_config
        src.utils.load_config = lambda: config
        
        try:
            # Test training (simplified version)
            X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
            
            # Create and train pipeline
            pipeline = create_pipeline(X_train)
            pipeline.fit(X_train, y_train)
            
            # Test predictions
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            
            assert len(y_pred) == len(X_test)
            assert len(y_proba) == len(X_test)
            assert all(pred in [0, 1] for pred in y_pred)
            assert all(0 <= prob <= 1 for prob in y_proba)
            
            print("✓ Full pipeline integration test passed")
            
        finally:
            # Restore original config loading
            src.utils.load_config = original_load_config


def run_smoke_tests():
    """Run all smoke tests."""
    print("Running smoke tests for ML pipeline...")
    print("=" * 50)
    
    try:
        test_config_loading()
        test_cost_matrix()
        test_synthetic_data_generation()
        test_data_splitting()
        test_pipeline_creation()
        test_full_pipeline_integration()
        
        print("=" * 50)
        print("🎉 All smoke tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
