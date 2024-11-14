# NVIDIA-Stock-Price-Prediction

This project leverages machine learning to analyze and forecast NVIDIA’s stock price movements. By examining historical data and applying predictive models, we forecast the stock's closing price trends to gain insights into potential future performance. This project includes a KPI summary in Excel and visual comparisons between actual and predicted prices for model evaluation.

## Tools Used

Python: For data preparation, technical indicator calculations, and model training.

Excel: For organizing key performance indicators (KPIs), including RMSE and MAE metrics for each model.

Matplotlib: For visualizing actual vs. predicted stock prices, comparing performance of Linear Regression and LSTM models.

## Repository Contents

nvidia_stock_prediction.py: The main code file for data extraction, feature engineering, model training, and predictions.

NVIDIA_KPI_Summary.xlsx: Excel sheet with KPI metrics, benchmarks, and business implications for both Linear Regression and LSTM models.

NVIDIA_Actual_vs_Predicted_LSTM_and_LR.png: A chart comparing actual and predicted prices, showing the performance of both models.

NVIDIA_Stock_Price_Prediction_Analysis.pdf: A comprehensive project summary document covering project goals, methods, and KPIs.

nvidia_stock_data.csv: Contains the raw stock data downloaded from Yahoo Finance, used for model training and analysis.

## How to Use This Project

This project is aimed at exploring stock price prediction using machine learning techniques. It’s ideal for finance enthusiasts, data scientists, and students interested in financial forecasting and machine learning applications in stock analysis.

Clone the Repository: Download or clone the project files to your local environment.
Install Required Libraries: pip install yfinance pandas numpy scikit-learn tensorflow matplotlib
Run the Code: Run nvidia_stock_prediction.py to see the results and visualize actual vs. predicted stock prices.
Analyze KPI Summary: Open NVIDIA_KPI_Summary.xlsx to review the performance metrics and business implications.
Read the Project Summary PDF: The PDF document provides a narrative summary of the project’s findings and model evaluations.

## Project Summary

By applying both Linear Regression and LSTM models, this project compares the accuracy and applicability of each model in predicting NVIDIA’s stock prices. The KPI Summary provides insights into model performance, with Linear Regression showing better short-term accuracy, and LSTM offering potential for long-term trend analysis despite higher volatility sensitivity.
