# AIDI-MLP-FinalProject

Stock Price Prediction using Bidirectional LSTM
Introduction
This repository contains the code for a stock price prediction project using advanced machine learning techniques, specifically Bidirectional Long Short-Term Memory (BiLSTM) networks. The primary goal of the project is to predict the prices of two stocks: Amazon (AMZN) and General Electric (GE). The project addresses the challenges in stock price forecasting, such as forecasting lag, using BiLSTM and compares its performance with other models like ARIMA and LSTM.

Problem Description
Predicting stock prices is crucial for investment strategies, portfolio management, and risk assessment. Accurate predictions can aid investors in making informed decisions and potentially increasing returns. The project focuses on utilizing BiLSTM, a variant of Recurrent Neural Network (RNN), to capture complex patterns and relationships in historical stock price data.

Context of the Problem
This project is not only significant for finance but also contributes to advancing the use of machine learning in real-world situations. BiLSTM, unlike traditional models, can understand complex temporal dependencies, making it suitable for unpredictable areas like stock markets.

Limitations of Other Approaches
While Long Short-Term Memory (LSTM) models have been widely used for stock price forecasting, they have limitations such as look-ahead bias, noise, uncertainty, and overfitting. The project proposes BiLSTM to address these limitations and improve model performance.

Solution: Bidirectional LSTM (BiLSTM)
BiLSTM offers several advantages over standard LSTM models for stock price forecasting:

Better handling of sequences: BiLSTM processes data in both forward and backward directions, providing information about future and past states simultaneously. This helps in understanding context and making more accurate predictions.

Reduced overfitting: BiLSTM learns input data in two ways, reducing overfitting by capturing patterns that a unidirectional LSTM might miss.

Improved model performance: By processing data bidirectionally, BiLSTM captures patterns and dependencies that may be overlooked by a unidirectional LSTM, leading to improved model performance.

Methodology
The project follows the methodology outlined in the Jupyter notebook (AIDI_1002_Final_Project_Template.ipynb). It covers data import and preparation, data visualization, technical indicators, Fourier transforms, feature engineering, and model training and evaluation using BiLSTM.

Usage
To reproduce the results, follow the steps outlined in the Jupyter notebook. Ensure that the required dependencies are installed using the provided requirements file.

bash
Copy code
pip install -r requirements.txt
Files and Directories
AIDI_1002_Final_Project_Template.ipynb: Jupyter notebook containing the complete project code and explanations.

utils.py: Utility functions used in the project.

drive/: Directory for storing the trained model and logs.

imgs/: Directory containing images generated during data visualization.

Dependencies
Python 3.7
TensorFlow 2.x
Keras
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
MXNet
yfinance
Plotly
tqdm

Contributors
Bhanu Bhakta Bhattarai
Purviben Patel
