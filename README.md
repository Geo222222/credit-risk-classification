# Credit Risk Classification

## 📊 Overview

This project aims to develop and evaluate a machine learning model to classify loan applicants based on their credit risk. Using historical lending data from a peer-to-peer platform, we apply classification algorithms to predict whether a borrower is likely to repay the loan.

## 🚀 Quickstart (Run in 5 Minutes)

Get the production-ready ML pipeline running in under 5 minutes:

```bash
# 1. Clone and navigate to the project
git checkout feat/ml-pipeline

# 2. Set up the environment (optional but recommended)
make setup

# 3. Train the model with cost-aware threshold optimization
make train

# 4. Evaluate and generate comprehensive reports
make eval
```

**That's it!** Check the `artifacts/` directory for the trained model and `reports/` for evaluation results.

### What You Get

- **Trained Model**: `artifacts/model.joblib` - Production-ready scikit-learn pipeline
- **Optimal Threshold**: `artifacts/threshold.txt` - Business-cost optimized decision threshold
- **Comprehensive Metrics**: `reports/metrics.json` - Including expected business loss
- **Visualizations**: `reports/*.png` - Confusion matrices, ROC curves, score distributions

### Key Features

- **Cost-Aware Thresholding**: Optimizes decision threshold to minimize expected business loss (FN cost: $1000, FP cost: $100)
- **Auto Data Discovery**: Automatically finds CSV files or generates synthetic data for demos
- **Reproducible Results**: Fixed random seed (42) ensures consistent outputs
- **Production Ready**: Modular code structure with proper preprocessing pipelines

## 🧠 Objective

Build a binary classifier to determine credit risk:
- **Target**: `loan_status` (Low Risk vs. High Risk)
- **Features**: Applicant attributes including loan amount, income, employment length, etc.

## 🛠️ Tools & Libraries

- Python
- Pandas
- Scikit-learn
- Jupyter Notebook
- PyYAML
- Matplotlib/Seaborn
- Joblib

## 📁 Project Structure

```
credit-risk-classification/
├── configs/
│   └── config.yaml              # Pipeline configuration
├── src/
│   ├── __init__.py
│   ├── utils.py                 # Utility functions and cost matrix
│   ├── data.py                  # Data loading with auto-discovery
│   ├── model.py                 # ML pipeline construction
│   ├── train.py                 # Training with threshold optimization
│   └── eval.py                  # Evaluation and reporting
├── tests/
│   └── test_smoke.py            # Integration tests
├── artifacts/                   # Generated models and thresholds
├── reports/                     # Evaluation reports and plots
├── Resources/
│   └── lending_data.csv         # Original dataset
├── requirements.txt             # Python dependencies
├── Makefile                     # Build automation
└── README.md
```


## ⚙️ Methodology

1. **Data Preparation**
   - Loaded peer-to-peer lending data
   - Cleaned and encoded categorical features
   - Scaled numerical features

2. **Model Development**
   - Baseline: Logistic Regression
   - Advanced: Random Forest, BalancedBaggingClassifier
   - Addressed class imbalance using SMOTE and undersampling

3. **Evaluation**
   - Confusion Matrix, Accuracy, Precision, Recall, F1-Score
   - Compared multiple models for robustness

## ✅ Results

| Model                  | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| Logistic Regression    | 71%      | 71%       | 70%    | 70%      |
| Random Forest          | 94%      | 95%       | 93%    | 94%      |
| Balanced Bagging Classifier | 93% | 93%       | 92%    | 92%      |

> ⚠️ These values are illustrative. Refer to the notebook for actual results.

## 📌 Future Improvements

- Cross-validation
- Feature selection & importance
- Hyperparameter tuning with GridSearchCV
- Deep learning model comparison



