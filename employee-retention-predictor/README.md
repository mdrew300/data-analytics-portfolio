# Employee Attrition Prediction

Machine learning classification project predicting employee attrition using Random Forest and Neural Network models. Demonstrates end-to-end ML workflow: exploratory analysis, feature engineering, model selection, comparison, and interpretability analysis.

# Overview

This project builds predictive models to identify employees at risk of attrition based on demographic and employment characteristics. Compares Random Forest and Artificial Neural Network performance, with emphasis on model interpretability through feature importance and partial dependence plots.

# Key Findings:

- Random Forest outperforms ANN on employee attrition data
- Feature importance analysis identifies key drivers of attrition risk
- Partial dependence plots reveal non-linear relationships between predictors and attrition likelihood

# Features

Data Analysis & Preparation
---

- Exploratory data analysis on employee demographics and employment metrics
- Feature engineering to create predictive signals
- Train/test split with stratification to preserve class balance

Model Development
---

- Random Forest Classifier (primary model) — ensemble method for robust predictions
- Artificial Neural Network (comparison) — deep learning baseline
- Hyperparameter tuning via GridSearchCV

Model Evaluation & Interpretation
---

- Confusion matrix, precision, recall, F1, and ROC-AUC metrics
- Feature importance ranking — which factors drive attrition decisions
- Partial dependence plots — how individual features influence predictions
- Model comparison and recommendation

# Technical Stack

- Languages: Python 3.x
- ML: scikit-learn (Random Forest, preprocessing, evaluation), TensorFlow/Keras (Neural Network)
- Data: pandas, numpy
- Visualization: matplotlib, seaborn

# Dataset

- Employee data with target variable: Attrition (binary: Yes/No)
- Predictors include demographic and employment characteristics (see notebook for full feature list).

# About
Built as an applied machine learning project demonstrating practical classification modeling: from data exploration through feature engineering, model selection, and actionable interpretability analysis. Shows ability to build, compare, and explain multiple ML approaches on a real-world HR analytics problem.
