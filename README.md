# Bitcoin Trading Strategy - Interactive Machine Learning Application

## Description
This project is an interactive web application built with **Streamlit** that uses supervised machine learning models to analyze Bitcoin trading strategies. The application predicts buy and sell signals based on financial indicators such as moving averages and provides a comparative analysis of various machine learning algorithms.

Users can interact with the application to:
- Visualize the performance of different machine learning models.
- Adjust hyperparameters for specific models.
- Analyze metrics like accuracy and confusion matrices.
- Compare algorithms using boxplots of error distributions.

## Features
1. **Algorithms Implemented**:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Random Forest
   - AdaBoost
   - Gradient Boosting

2. **Key Visualizations**:
   - Boxplot comparing model errors.
   - Confusion matrix for detailed evaluation.
   - Interactive adjustment of hyperparameters.

3. **Metrics Analyzed**:
   - Accuracy
   - Cross-validation error distribution
   - Confusion matrix

## Dataset
The application uses the `Bitstamp_AAVEBTC_d.csv` dataset, which contains historical data on Bitcoin trading. The dataset includes columns such as `open`, `high`, `low`, `close`, and trading volumes.

Ensure the dataset is in the same directory as the application script.

## Requirements
- Python 3.9 or later
- Required libraries:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `plotly`

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repository/bitcoin-trading-strategy.git
   cd bitcoin-trading-strategy

2. Install the required libraries:
pip install -r requirements.txt

3. Place the Bitstamp_AAVEBTC_d.csv dataset in the project directory.


Author
Developed by Luiz Fernando Nunes.
