# Income Level Prediction Model
by: Shane C. 

## Overview
This repository contains the machine learning project 'Income Level Prediction Model' which is designed to predict whether individuals earn above or below $50k. The project uses a dataset `income.csv` and is fully developed in a Jupyter Notebook `Main.ipynb`.

## Technologies Used
- [Python 3](https://www.python.org/downloads/)
- [Jupyter Notebook](https://jupyter.org/install)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Scikit-Learn](https://scikit-learn.org/stable/)

## Description
The project's objective is to employ various machine learning techniques to predict income levels. It involves data exploration, preprocessing, model training, hyperparameter tuning with GridSearchCV, and model evaluation.

### Main.ipynb
- **Input**: `income.csv` file containing demographic and employment data.
- **Process**: Includes data cleaning, exploratory data analysis, feature engineering, model training, and evaluation.
- **Output**: A RandomForestClassifier model capable of predicting income levels, along with performance metrics to assess its accuracy.

## How It Works

### Data Preparation
The raw data from `income.csv` is cleaned and processed. Categorical features are encoded, and irrelevant features are dropped to prepare the dataset for training.

### Model Training
A RandomForestClassifier is trained with a set of hyperparameters that are optimized using GridSearchCV to ensure the model's performance is maximized.

### Model Evaluation
The model's performance is evaluated on a test dataset to understand its predictive capabilities. Metrics such as accuracy score and feature importance are used to interpret the results.

## Setup/Installation Requirements
1. Install Python 3 from the [official website](https://www.python.org/downloads/).
2. Download csv from [official website](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset). (If starting from scratch)
3. Install Jupyter Notebook, preferably through the Anaconda distribution. Instructions can be found [here](https://www.datacamp.com/community/tutorials/installing-jupyter-notebook).
4. Install the required Python libraries with `pip install pandas numpy scikit-learn`.
5. Clone this repository to your local environment.
6. Navigate to the directory containing `Main.ipynb` and launch Jupyter Notebook with `jupyter notebook`.

## Contact Information
Please reach out to me for any questions or suggestions at [your email/contact].

## License
This project is open-sourced under the [MIT License](LICENSE).
