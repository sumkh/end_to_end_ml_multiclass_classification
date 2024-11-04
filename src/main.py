import os
import pandas as pd
import sqlite3

import joblib
from joblib import dump, load
import requests
import json

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_preparation import DataPreparation
from feature_differencing import FeatureDifferencing
from feature_selection import FeatureSelector
from model_training import ModelTraining
from model_evaluation import ModelEvaluation
from model_interpretation import BestModel_Interpretation
from thresholdfinder import ThresholdFinder

import joblib
from joblib import dump, load
import json

import warnings
# Ignore all warnings
warnings.filterwarnings('ignore')


# Set the working directory to the directory of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Define the Main Object
def main():
    """
    Main function to orchestrate the data preparation, model training, and evaluation processes.

    This function performs the following steps:
    1. Loads configuration settings.
    2. Downloads and loads data from the specified URLs.
    3. Prepares the data by cleaning and splitting it.
    4. Applies feature engineering and selection techniques.
    5. Trains multiple machine learning models.
    6. Evaluates the trained models.
    7. Finds the optimal decision thresholds.
    8. Interprets the best model and plots feature importance.
    9. Saves the best model.
    """

    # Load configuration settings

    print('\n------------------------------------\n')
    print('Starting Data Preparation...')
    print('\n------------------------------------\n')

    # Open config.json and load its content
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    # Path to dataset
    url1 = config['url1']
    url2 = config['url2']

    # Load weather data from SQL database from the URL
    response = requests.get(url1)
    if response.status_code == 200:
      # Create a temporary file to store the database
      with open('weather.db', 'wb') as temp_file:
        temp_file.write(response.content)

      # Connect to the temporary database file
      conn = sqlite3.connect('weather.db')
      query = "SELECT * FROM weather"
      weather_data = pd.read_sql_query(query, conn, index_col = 'data_ref')
      conn.close()


    # Load air_quality data from SQL database from the URL
    response = requests.get(url2)
    if response.status_code == 200:
      # Create a temporary file to store the database
      with open('air_quality.db', 'wb') as temp_file:
        temp_file.write(response.content)

      # Connect to the temporary database file
      conn = sqlite3.connect('air_quality.db')
      query = "SELECT * FROM air_quality"
      air_quality_data = pd.read_sql_query(query, conn, index_col = 'data_ref')
      conn.close()

    print('Data loaded from database')
    print('\n------------------------------------\n')

    # Load data from database, clean, add features and split
    data_preparer = DataPreparation(weather_data, air_quality_data, config)
    data = data_preparer.clean_data()

    print(data.info())
    print('\n------------------------------------\n')

    X_train, X_test, y_train, y_test = data_preparer.split_data()
    print("Successfully generated Train and Test sets.")

    # Print shape of Train and Test Set
    print('Training Set:')
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}\n")

    print('Test Set:')
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}\n")

    # Proportion of Train and Test Split
    print(f"Proportion of Train Split: {len(X_train)/len(data):.2f}")
    print(f"Proportion of Test Split: {len(X_test)/len(data):.2f}")

    print('\n------------------------------------\n')

    print('X_train:')
    print(X_train.info())
    print('\n------------------------------------\n')


    print("Successfully generated Train and Test sets.")
    print('\n------------------------------------\n')

    # Feature filtering with correlation
    print('Features Filtering with Correlation')
    print('\n------------------------------------\n')

    # Instantiate Feature Selector
    corr_selector = FeatureSelector(X_train, y_train)
    corr_list = corr_selector.get_features_list()
    print('List of highly correlated features: %s' %corr_list)

    print('\n------------------------------------\n')

    # Drop features from X_train and X_test
    if corr_list is not None:
        X_train = X_train.drop(corr_list, axis=1)
        X_test = X_test.drop(corr_list, axis=1)

        print(f"Number of features after filtering: {len(X_train.columns)}")
        print(X_train.info())
    else:
        print("No highly correlated Features to drop.")

    print('\n------------------------------------\n')


    #Feature Creation with Time-Series Differencing on Train and Test Sets separately

    #Note: Ensure that the detection of non-stationary features is done using
    #the training set, and then apply the same differencing to both
    #the train and test sets based on this analysis.
    #This approach avoids data leakage.

    # Feature creation with time-series differencing
    print('Feature Creation with Time-Series Differencing')
    print('\n------------------------------------\n')

    # Instantiate the class with the training data
    fd = FeatureDifferencing(X_train)

    # Identify non-stationary features from the train set
    non_stationary_features = fd.is_stationary()

    print(f'List of Non-Stationary Features to be Differenced: {non_stationary_features}')

    # Apply differencing to both train and test sets
    X_train = fd.add_difference(X_train, non_stationary_features)
    X_test = fd.add_difference(X_test, non_stationary_features)
    print('\n------------------------------------\n')


    # Feature selection with Random Forest algorithm
    print('Feature Selection with Random Forest Algorithm')
    print('\n------------------------------------\n')

    # Fit Random Forest Selector
    rf_selector = FeatureSelector(X_train, y_train)
    rf_selector.preprocess_features()
    rf_selector.fit()

    # List of Low Importance Features to filter from X_train and X_test
    print(rf_selector.df_importance_features())
    print('\n------------------------------------\n')

    low_importance_features = rf_selector.low_importance_features(threshold=config['threshold'])

    # Drop low importance features from X_train and X_test
    if low_importance_features is not None:
        X_train = X_train.drop(low_importance_features['Original Feature'], axis=1)
        X_test = X_test.drop(low_importance_features['Original Feature'], axis=1)
        print(low_importance_features)

        print(f"Number of features after filtering: {len(X_train.columns)}")
        print(X_train.info())
    else:
        print("No low importance features to drop.")

    print('\n------------------------------------\n')

    # Plot top n_features importance
    rf_selector.plot_high_importance_features(n_features=10)

    print('\n------------------------------------\n')
    print('Feature Selection Completed')
    print('\n------------------------------------\n')

    print('Finalised X_train Dataset:')
    print(X_train.info())

    print('\n------------------------------------\n')

    # Model training
    print('Starting Models Training...')

    # Setup data preprocessing pipelines
    model_trainer = ModelTraining(config= config)
    preprocessor = model_trainer.get_preprocessor(X_train)

    # Setup Models Hyperparameters Search Space, training models and get results
    #train_models = model_trainer.train_models(X_train, y_train, preprocessor, config = config)
    train_models = model_trainer.train_models_gridsearch(X_train, y_train, preprocessor, config = config)

    # Get results dataframe
    results_df, cvfit_results_df = train_models


    print('\n------------------------------------\n')
    print('All Models trained successfully')

    # Model evaluation
    print('\n----------Models Evaluation --------------------------\n')

    # Evaluate models
    m_evaluation = ModelEvaluation(results_df, X_train, y_train, X_test, y_test)

    # Add evaluation metrics to results_df
    results_df = m_evaluation.evaluate_models()

    print('\n--------- Best Model (After Evaluation with Test Data) ---------\n')

    print (f"Best Model: {results_df.iloc[0]['model']}")
    print (f"Best Parameters: {results_df.iloc[0]['best_params']}")
    print (f"Best Validation Score: {results_df.iloc[0]['best_validation_score']}")
    print (f"ROC-AUC Train: {results_df.iloc[0]['roc_auc_train']}")
    print (f"ROC-AUC Test: {results_df.iloc[0]['roc_auc_test']}")

    print('\n------------------------------------\n')

    # Save the best model
    best_model_data = results_df.iloc[0]['model_data']
    dump(best_model_data, 'best_model.joblib')
    print('Best Model Saved')

    print('\n------------------------------------\n')

    # Plot ROC and Precision-Recall curves
    m_evaluation.plot_roc_prc()

    print('\n------------------------------------\n')

    # Classification Report and Confusion Matrix
    m_evaluation.classification_and_cm()


    print('\n------------------------------------\n')

    # Best Model under Grid SearchCV
    model_name = results_df.iloc[0]['model']


    print(f"Best Model trained with Grid SearchCV: {model_name}")
    # Save the best model
    best_model_data = results_df.iloc[0]['model_data']
    dump(best_model_data, 'best_model_CV.joblib')
    print('Best Model Saved')

    print('\n------------------------------------\n')

    # Find optimal thresholds for each class
    y_pred_prob = best_model_data.predict_proba(X_test)

    finder = ThresholdFinder(y_test, y_pred_prob)

    optimal_thresholds = finder.prc_plot_multiclass()

    print('\n------------------------------------\n')
    #optimal_thresholds = [0.39, 0.55, 0.33] # Example thresholds
    print("Best thresholds per class:", optimal_thresholds)

    print('\n------------------------------------\n')

    # Use the optimal thresholds to make final predictions
    optimal_y_pred = finder.predict_optimal_thresholds(optimal_thresholds)

    print("Classification Report with Optimized Thresholds:")

    print(classification_report(finder.y_test, optimal_y_pred))

    # Confusion Matrix
    cm = confusion_matrix(finder.y_test, optimal_y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


    print('\n------------------------------------\n')

    print('Model Evaluation Completed')

    print('\n------------------------------------\n')

    # Model interpretation
    print('Interpreting Best Model...')
    print('\n------------------------------------\n')
    # Initialize the interpreter with the entire pipeline
    interpreter = BestModel_Interpretation(best_model_data, X_test, preprocessor)

    # Extract and process importances or coefficients using the model in the pipeline
    #original_feature = interpreter.original_features()
    print(interpreter.create_df_features())
    print('\n------------------------------------\n')

    # Plotting top 10 features by importance or coefficient magnitude
    high_importance_features = interpreter.plot_high_importance_features(n_features=20)
    print('\n------------------------------------\n')

    # Save the best model
    dump(best_model_data, 'best_model.joblib')
    print('Best Model Saved')


if __name__ == "__main__":
    main()
