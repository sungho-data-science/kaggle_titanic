# kaggle_titanic

## Overview
This repository is dedicated to solving the Kaggle competition, [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview). The objective of this competition is to predict the survival status of passengers from the Titanic shipwreck.

## Repository Structure 
The repository is organized into two main sections:
* Exploratory Data Analysis (EDA):
   * Conducted visual analyses to evaluate the relevance of each feature.
   * Performed feature engineering by modifying and creating new features to improve model performance.
* Modeling:
   * Implemented two machine learning models: XGBoost and Logistic Regression.
   * Applied Bayesian Optimization to identify optimal hyperparameters for each model.
   * Trained each model using the best hyperparameters and compared their performance.
   * Submitted predictions from both models to evaluate their accuracy on the test set.

## Results
The XGBoost model achieved a slightly better accuracy score of 0.79186 compared to the Logistic Regression model's accuracy score of 0.78947.
