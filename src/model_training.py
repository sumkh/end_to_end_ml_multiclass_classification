import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from imblearn.ensemble import RUSBoostClassifier

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import time
import json

class ModelTraining:
    """
    A class for training and optimizing machine learning models.

    Attributes:
    ----------
    config : dict
        Configuration settings for model training.
    results_df : pd.DataFrame
        DataFrame of model training results.
    cv_results_df : pd.DataFrame
        DataFrame of cross-validation results.

    Methods:
    -------
    get_preprocessor(X_train)
        Returns a preprocessing pipeline.
    train_models(X_train, y_train, preprocessor)
        Trains models using RandomizedSearchCV and returns the results.
    train_models_gridsearch(X_train, y_train, preprocessor)
        Trains models using GridSearchCV and returns the results.
    """
    def __init__(self, config):
        self.config = config
        self.results_df = pd.DataFrame()
        self.cv_results_df = pd.DataFrame()
        self.best_model_data = None


    # Define the preprocessors for numerical and categorical features
    def get_preprocessor(self, X_train):
        """
        Returns a preprocessing pipeline.

        Parameters:
        ----------
        X_train : pd.DataFrame
            The training features.

        Returns:
        -------
        ColumnTransformer
            The preprocessing pipeline.
        """
        # Determine the list of numerical and categorical features
        cat_var = X_train.select_dtypes(include=['object']).columns
        num_var = X_train.select_dtypes(include=['float64', 'int64', 'int', 'bool']).columns

        # Create the preprocessing pipeline
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_var),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_var)
        ])

        return preprocessor


    # Define the function to fit the preprocessor, transform the data, and get feature names
    def get_processed_data_and_feature_names(self, X_train):
        """Fit the preprocessor, transform the data, and get feature names."""
        preprocessor = self.get_preprocessor(X_train)
        X_transformed = preprocessor.fit_transform(X_train)
        feature_names = preprocessor.get_feature_names_out()
        return X_transformed, feature_names


    # Define the model and respective hyperparameters search space and train the models
    def train_models(self, X_train, y_train, preprocessor, config):
        """
        Trains models using RandomizedSearchCV and returns the results.

        Parameters:
        ----------
        X_train : pd.DataFrame
            The training features.
        y_train : pd.Series
            The training target variable.
        preprocessor : ColumnTransformer
            The preprocessing pipeline.

        Returns:
        -------
        tuple: (pd.DataFrame, pd.DataFrame)
            The results and cross-validation results DataFrames.
        """
        model_params = {
            'random_forest': {
                'model': [RandomForestClassifier()],
                'model__n_estimators': [100, 200, 300, 500],
                'model__max_depth': [None, 10, 20, 30],
                'model__class_weight': [
                      {0: 1 / 0.2, 1: 1 / 0.5, 2: 1 / 0.3},  # weights for 3 classes
                      {0: 1 / 0.25, 1: 1 / 0.5, 2: 1 / 0.25},
                      {0: 1 / 0.3, 1: 1 / 0.4, 2: 1 / 0.3},
                      'balanced']
            },
            'xgb': {
                'model': [XGBClassifier()],
                'model__n_estimators': [100, 200, 300, 500],
                'model__eta': [0.01, 0.05, 0.1, 0.2, 0.3] # learning rate
            },
            'rusboost': {
                'model': [RUSBoostClassifier()],
                'model__n_estimators': [100, 200, 300, 500],
                'model__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3]
            },
        }

        # Setup cross validation folds with TimeSeriesSplit
        cv = TimeSeriesSplit(n_splits=config['cv_splits'])

        # Create an empty list to store the results
        results = []
        cvfit_results = []

        # Iterate over the models and hyperparameters, perform RandomizedSearchCV
        for model_name, params in model_params.items():
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', params['model'][0]) # Initialize the model (first element in the list of models)
            ])
            grid_search = RandomizedSearchCV(pipeline,
                                            param_distributions=params,
                                            cv=cv,
                                            scoring='roc_auc_ovr',
                                            verbose=1,
                                            return_train_score=True,
                                            n_iter=config['n_iter'],
                                            n_jobs=-1) # Set n_jobs to -1 to use all available cores

            start_time = time.time()
            print(f"\nTraining {model_name}...")
            grid_search.fit(X_train, y_train)
            end_time = time.time()

            cv_results = grid_search.cv_results_
            for i in range(len(cv_results['mean_train_score'])):
                cvfit_results.append({
                    'model': model_name,
                    'fit_index': i,
                    'params': grid_search.best_params_,
                    'mean_train_score': cv_results['mean_train_score'][i],
                    'mean_validation_score': cv_results['mean_test_score'][i],
                    'train_time': end_time - start_time
                })

            results.append({
                'model': model_name,
                'best_params': grid_search.best_params_,
                'best_train_score': grid_search.cv_results_['mean_train_score'][grid_search.best_index_],
                'best_validation_score': grid_search.best_score_,
                'model_data': grid_search.best_estimator_
            })

        # Convert cv_results to DataFrame
        self.cvfit_results_df = pd.DataFrame(cvfit_results)
        self.cvfit_results_df = self.cvfit_results_df.sort_values(by='mean_validation_score', ascending=False)

        # Convert results to DataFrame
        self.results_df = pd.DataFrame(results)
        self.results_df = self.results_df.sort_values(by='best_validation_score', ascending=False)

        return self.results_df, self.cvfit_results_df

    # GridSearchCV: Define the model and respective hyperparameters search space and train the models
    def train_models_gridsearch(self, X_train, y_train, preprocessor, config):
        """
        Trains models using GridSearchCV and returns the results.

        Parameters:
        ----------
        X_train : pd.DataFrame
            The training features.
        y_train : pd.Series
            The training target variable.
        preprocessor : ColumnTransformer
            The preprocessing pipeline.

        Returns:
        -------
        tuple: (pd.DataFrame, pd.DataFrame)
            The results and cross-validation results DataFrames.
        """
        model_params = {
            'random_forest': {
                'model': [RandomForestClassifier()],
                'model__n_estimators': [100, 200, 300, 500],
                'model__max_depth': [None, 10, 20, 30],
                'model__class_weight': [
                      {0: 1 / 0.2, 1: 1 / 0.5, 2: 1 / 0.3},  # weights for 3 classes
                      {0: 1 / 0.25, 1: 1 / 0.5, 2: 1 / 0.25},
                      {0: 1 / 0.3, 1: 1 / 0.4, 2: 1 / 0.3},
                      'balanced']
            },
            'xgb': {
                'model': [XGBClassifier()],
                'model__n_estimators': [100, 200, 300, 500],
                'model__eta': [0.01, 0.05, 0.1, 0.2, 0.3] # learning rate
            },
            'rusboost': {
                'model': [RUSBoostClassifier()],
                'model__n_estimators': [100, 200, 300, 500],
                'model__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3]
            },
        }

        # Setup cross validation folds with TimeSeriesSplit
        cv = TimeSeriesSplit(n_splits=config['cv_splits'])

        # Create an empty list to store the results
        results = []
        cvfit_results = []

        # Iterate over the models and hyperparameters, perform RandomizedSearchCV
        for model_name, params in model_params.items():
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', params['model'][0]) # Initialize the model (first element in the list of models)
            ])
            grid_search = GridSearchCV(pipeline,
                                       param_grid = params,
                                       cv=cv,
                                       scoring='roc_auc_ovr',
                                       verbose=1,
                                       return_train_score=True,
                                       n_jobs=-1) # Set n_jobs to -1 to use all available cores

            start_time = time.time()
            print(f"\n Training with Grid SearchCV: {model_name}...")
            grid_search.fit(X_train, y_train)
            end_time = time.time()

            cv_results = grid_search.cv_results_
            for i in range(len(cv_results['mean_train_score'])):
                cvfit_results.append({
                    'model': model_name,
                    'fit_index': i,
                    'params': grid_search.best_params_,
                    'mean_train_score': cv_results['mean_train_score'][i],
                    'mean_validation_score': cv_results['mean_test_score'][i],
                    'train_time': end_time - start_time
                })

            results.append({
                'model': model_name,
                'best_params': grid_search.best_params_,
                'best_train_score': grid_search.cv_results_['mean_train_score'][grid_search.best_index_],
                'best_validation_score': grid_search.best_score_,
                'model_data': grid_search.best_estimator_
            })

        # Convert cv_results to DataFrame
        self.cvfit_results_df = pd.DataFrame(cvfit_results)
        self.cvfit_results_df = self.cvfit_results_df.sort_values(by='mean_validation_score', ascending=False)

        # Convert results to DataFrame
        self.results_df = pd.DataFrame(results)
        self.results_df = self.results_df.sort_values(by='best_validation_score', ascending=False)

        return self.results_df, self.cvfit_results_df
