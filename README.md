# Income Prediction Model
by: Shane C. 

## Overview
This project focuses on building a machine learning model that predicts the income level of individuals based on demographic and employment data. The project encompasses the entire machine learning workflow, including data preprocessing, model training, hyperparameter tuning, and evaluation, all within Jupyter Notebooks.

## Technologies Used
- [Python 3](https://www.python.org/downloads/)
- [Jupyter Notebook](https://jupyter.org/install)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Scikit-Learn](https://scikit-learn.org/stable/)

## Description
The aim of this project is to accurately classify individuals into two income categories: those earning above or below $50k. The Jupyter Notebooks guide the user through exploratory data analysis, feature engineering, model selection, and the interpretation of results.

### Jupyter Notebooks
#### Income_Data_Preparation.ipynb
- **Input**: Raw demographic and employment data.
- **Output**: Preprocessed dataset ready for model training.
- **Description**: This notebook handles data cleaning, feature selection, and preprocessing to structure the dataset for machine learning.

#### Income_Model_Training.ipynb
- **Input**: Preprocessed data from `Income_Data_Preparation.ipynb`.
- **Output**: A trained and tuned RandomForestClassifier model.
- **Description**: This notebook includes the model training process and hyperparameter optimization to build a robust classifier for income prediction.

## How It Works

### Data Understanding
The model starts with a comprehensive analysis of the dataset to understand the distribution of various features and their relationship with the income levels.

### Feature Engineering and Selection
The dataset is transformed through one-hot encoding to represent categorical features numerically. Features with little impact on income prediction are identified and removed to improve model performance.

### Model Training and Optimization
A RandomForestClassifier is trained on the dataset, with its hyperparameters optimized using GridSearchCV to ensure the best possible predictions.

### Model Evaluation
The trained model is evaluated on a separate testing set to estimate its real-world performance. The evaluation focuses on the accuracy and feature importance to interpret how different features affect income prediction.

## Setup/Installation Requirements
1. Install Python 3 from the [official website](https://www.python.org/downloads/).
2. Install Jupyter Notebook, preferably via Anaconda. Here's a [guide](https://www.datacamp.com/community/tutorials/installing-jupyter-notebook).
3. Install the required Python libraries by running `pip install pandas numpy scikit-learn` in your command line.
4. Clone this repository to your local machine.
5. Navigate to the local repository and run `jupyter notebook` to open the Jupyter Notebooks in your browser.

## Contact Information
For any queries or feedback, please feel free to contact me at [your contact information].

## License
This project is licensed under the [MIT License](LICENSE).
