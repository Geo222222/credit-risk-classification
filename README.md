# Credit Risk Classification

## 📊 Overview

This project aims to develop and evaluate a machine learning model to classify loan applicants based on their credit risk. Using historical lending data from a peer-to-peer platform, we apply classification algorithms to predict whether a borrower is likely to repay the loan.

## 🧠 Objective

Build a binary classifier to determine credit risk:
- **Target**: `loan_status` (Low Risk vs. High Risk)
- **Features**: Applicant attributes including loan amount, income, employment length, etc.

## 🛠️ Tools & Libraries

- Python
- Pandas
- Scikit-learn
- Jupyter Notebook

## 📁 Project Structure


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



