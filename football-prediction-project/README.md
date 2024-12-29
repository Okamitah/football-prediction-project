# Football prediction project
This repository contains a comprehensive pipeline for predicting football results using machine learning. The project leverages data scraped from Flashscore, followed by extensive preprocessing and feature engineering, culminating in training a neural network for predictions. Below, we outline each step of the project and provide guidance on how to replicate or build upon this work.
This repository contains a comprehensive pipeline for predicting football results using machine learning. The project leverages data scraped from Flashscore, followed by extensive preprocessing and feature engineering, culminating in training a neural network for predictions. Below, we outline each step of the project and provide guidance on how to replicate or build upon this work.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Technologies Used](#technologies-used)
3. [Data Collection](#data-collection)
4. [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Model Training](#model-training)
7. [How to Run](#how-to-run)
8. [Results](#results)
9. [Future Work](#future-work)
10. [Acknowledgments](#acknowledgments)

---

## Introduction
Predicting football match outcomes is a challenging task due to the inherent randomness and numerous factors influencing the results. This project aims to build a data-driven approach using historical match data to train a neural network capable of forecasting match outcomes. The process begins with web scraping and continues through data cleaning, feature engineering, and model training, demonstrating a complete data science pipeline.

---

## Technologies Used
- **Python**: Core programming language
- **Selenium**: For web scraping Flashscore data
- **Pandas**: For data manipulation
- **Matplotlib and Seaborn**: For visualization
- **Scikit-learn**: For preprocessing and baseline modeling
- **PyTorch**: For building and training the neural network

---

## Data Collection
Using Selenium, data from Flashscore was scraped, including details such as match results, teams, dates, and player ratings. Approximately 10,000 matches were collected, ensuring a robust dataset for analysis. The scraped data was saved in CSV format for further processing.

---

## Data Preprocessing and Feature Engineering
### Steps:
1. **Data Cleaning**:
   - Removed missing or invalid entries.
   - Standardized team names and match formats.

2. **Feature Engineering**:
   - Calculated features such as current standings, average player ratings, and recent performance (last 5 matches).
   - Encoded categorical variables (e.g., teams) using one-hot encoding or label encoding.

3. **Normalization**:
   - Scaled numerical features to ensure compatibility with the neural network.

The cleaned and engineered dataset was saved for reproducibility.

---

## Exploratory Data Analysis (EDA)
EDA was conducted to understand the data distribution and relationships between features. Key visualizations include:
- Match outcome distributions.
- Correlations between features.
- Trends in team performance over time.

---

## Model Training
### Neural Network Architecture:
The model is a feedforward neural network implemented using TensorFlow/Keras. Key parameters include:
- Input Layer: Corresponds to the number of features.
- Hidden Layers: Two dense layers with ReLU activation.
- Output Layer: Softmax for predicting match outcomes (Win/Loss/Draw).

### Training:
- Loss Function: A customised function that calculates EV.
- Optimizer: Adam.
- Metrics: Accuracy.
- Validation: Performed using a train-test split (e.g., 80/20).

Hyperparameter tuning was performed to optimize the model's performance.

---

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/okamitah/football-prediction-project.git