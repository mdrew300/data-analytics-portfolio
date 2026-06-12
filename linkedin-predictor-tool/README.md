# LinkedIn User Prediction Tool
Interactive Streamlit application that predicts whether an individual is a LinkedIn user based on demographic characteristics. Demonstrates end-to-end machine learning workflow: data cleaning, logistic regression modeling, model evaluation, and interactive prediction interface.

# Features
Prediction Tab

- Input demographic characteristics (income, education, parental status, marital status, gender, age)
- Get binary prediction (LinkedIn user / non-user) with probability score
- Explore how individual factors influence prediction likelihood

# Data Visualizations Tab

- Density plots comparing feature distributions across LinkedIn users vs. non-users
- Visual exploration of predictor relationships to target variable
- Helps identify key demographic patterns in the data

# Model Performance Tab

- Confusion matrix showing prediction accuracy across classes
- Precision, recall, and F1 score metrics
- ROC curve and AUC score for overall model discrimination

# Technical Stack

- Framework: Streamlit (interactive web app)
- ML: scikit-learn (logistic regression, model evaluation)
- Data: pandas, numpy
- Visualization: matplotlib, seaborn

# Model Details
- Target Variable: LinkedIn usage (binary: user / non-user)

# Predictors:

- Income (9 ordinal levels)
- Education (8 ordinal levels)
- Parental status (binary)
- Marital status (binary)
- Gender (binary)
- Age (numeric, 18-98)

# Performance:

Precision: 56.4% | Recall: 75% | F1: 0.644

Balanced class weights to handle imbalanced training data

# How to Run
Visit: [LinkedIn Predictor Tool](https://linkedin-prediction-md.streamlit.app/)

# About
Built as an applied machine learning project demonstrating practical predictive analytics: from data preparation through model deployment and interactive user interface. Shows ability to translate technical modeling work into an accessible tool for end users.
