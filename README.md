# Forest-Flame-Predictor
Algerian Forest Fire Prediction

This project uses machine learning to predict forest fires in Algeria based on environmental and meteorological data. The goal is to analyze the dataset, perform extensive exploratory data analysis (EDA), and build an effective predictive model to assist in early detection and prevention of forest fires.

Features

Dataset: Algerian Forest Fires dataset containing meteorological data such as temperature, humidity, wind speed, rainfall, and more.

EDA: Comprehensive analysis to understand patterns, correlations, and distributions.

Feature Engineering: Extraction of meaningful features like seasonal data, derived metrics, and time-related trends.

Machine Learning Models: Implementation of algorithms like Logistic Regression, Random Forest, and Gradient Boosting for fire prediction.

Visualization: Graphical representation of trends and relationships in the dataset for better insights.

Project Structure

algerian_forest_fires.py: Main Python script for data analysis, preprocessing, and model training.

data/Algerian_forest_fires_dataset.csv: The dataset used for the project.

results/: Contains model performance metrics and visualizations.

README.md: Project documentation.

Setup Instructions

Prerequisites

Ensure you have the following installed:

Python 3.8 or later

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Installation

Clone this repository:

git clone https://github.com/Pranavtiwari30/Forest-Flame-Predictor
cd algerian-forest-fires

Install the required dependencies:

pip install -r requirements.txt

Usage

1. Data Preprocessing and EDA

Run the script to perform data cleaning, feature engineering, and visualization:

python algerian_forest_fires.py --eda

This step includes handling missing values, detecting outliers, and visualizing correlations.

2. Model Training

Train machine learning models on the processed dataset:

python algerian_forest_fires.py --train

This will train multiple models and save the best-performing one.

3. Predictions

Make predictions using the trained model:

python algerian_forest_fires.py --predict --input data/sample_input.csv

The script will output fire risk predictions for the provided data.

Example Visualizations

Correlation Heatmap:

Shows relationships between features like temperature, humidity, and wind speed.

Feature Distribution:

Displays the distribution of key variables across different fire risk categories.

Time-Series Trends:

Analyzes seasonal fire patterns.

To Do

Add advanced models like XGBoost or Neural Networks for better accuracy.

Implement real-time prediction using a web interface or API.

Enhance feature engineering by integrating external weather data.

License

This project is licensed under the MIT License. See LICENSE for details.

Acknowledgments

Dataset: Algerian Forest Fires Dataset

Libraries: Pandas, Scikit-learn, Matplotlib, Seaborn
