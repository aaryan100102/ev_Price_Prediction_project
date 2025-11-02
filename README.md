⚡ Electric Vehicle (EV) Price Prediction Project
🚗 Overview

This project focuses on predicting Electric Vehicle (EV) prices using Machine Learning techniques.
The dataset contains detailed attributes of EVs such as battery capacity, range, power, charging time, brand, and body type, which are used to predict both exact prices and price categories.

The goal is to develop an efficient data-driven model that can assist consumers, manufacturers, and policymakers in understanding the key factors influencing EV pricing.

❓ Problem Statement

Electric Vehicles (EVs) are rapidly transforming the automotive industry, but their prices vary widely depending on numerous technical and brand-related factors.
Determining an EV’s price based on its specifications can help:

Consumers make better purchase decisions,

Manufacturers optimize pricing strategies, and

Researchers understand what truly drives EV costs.

The objective of this project is to predict the price of an electric vehicle based on its specifications using machine learning techniques, and to classify EVs into price categories (Low, Medium, High) for comparative analysis.

🧠 Key Objectives

Clean and preprocess the EV dataset for modeling

Perform Exploratory Data Analysis (EDA) to uncover insights and relationships

Build regression and classification models to predict EV prices and categories

Compare different machine learning algorithms such as:

Linear Regression

Random Forest Regressor

XGBoost Regressor

Logistic Regression (for category prediction)

Evaluate model performance and save trained models for reuse

📊 Dataset

The dataset used is EV_data.csv, containing multiple numerical and categorical features like:

Brand

Model

Battery Capacity (kWh)

Range (km)

Charging Time (hours)

Power (kW)

Torque (Nm)

Price (₹)

🔍 Exploratory Data Analysis (EDA)

EDA includes:

Statistical summaries of data

Missing value treatment and type conversions

Distribution plots for numerical columns

Correlation heatmap

Outlier detection

Relationship visualization between features and price

🤖 Machine Learning Models
🔹 Regression Models

Random Forest Regressor

XGBoost Regressor

Linear Regression

🔹 Classification Model

Logistic Regression — to classify EVs into price categories such as Low, Medium, High

🧩 Technologies Used

Python

NumPy, Pandas – Data processing

Matplotlib, Seaborn – Visualization

Scikit-learn – Model training & evaluation

XGBoost – Advanced gradient boosting

Joblib – Model saving/loading

🚀 Results

The Random Forest and XGBoost models achieved high accuracy in predicting EV prices.

The Logistic Regression classifier effectively categorized EVs into price ranges.

Data cleaning and feature engineering significantly improved model performance.

📈 Future Improvements

Include real-time EV datasets from APIs

Add deep learning models for better predictions

Build a web-based interface using Streamlit or Flask

👨‍💻 Author

Aaryan
B.Tech (CSE - Data Science)
