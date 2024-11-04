import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

class FeatureDifferencing:
    """
    A class for applying differencing to time series data to make it stationary.

    Attributes:
    ----------
    X : pd.DataFrame
        The input features dataframe.

    Methods:
    -------
    is_stationary(alpha=0.025)
        Tests if the features are stationary using the Augmented Dickey-Fuller test.
    add_difference(data, columns_to_difference, n_periods=30)
        Applies differencing to specified columns.
    """

    def __init__(self, X):
        """
        Feature Differencing class for time series data.
        X: Features DataFrame containing time series data.
        """
        self.X = X

    def is_stationary(self, alpha=0.025): # alpha = 5%/2
        """
        Tests if the features are stationary using the Augmented Dickey-Fuller test.
        Data is resampled monthly at the start of each month ('MS').

        Parameters:
        ----------
        alpha : float, optional
            Significance level for the test (default is 0.025).

        Returns:
        -------
        list
            Columns that are non-stationary.
        """

        columns_to_difference = []
        resampled_data = self.X.select_dtypes(include=['int64', 'float64'])
        resampled_data = resampled_data.resample('MS').mean()  # Resample data monthly at start
        resampled_data = resampled_data.rolling(window= 3 ).mean() # Window size = 3 months rolling average

        for col in resampled_data.columns:
          series = resampled_data[col].dropna()
          result = adfuller(series) # ADF Test
          p_value = result[1]
          if p_value >= alpha:  # Check if feature is non-stationary
              columns_to_difference.append(col)

        return columns_to_difference

    def add_difference(self, data, columns_to_difference, n_periods=30):
        """
        Applies differencing to specified columns and add to the input data.

        Parameters:
        ----------
        data : pd.DataFrame
            Dataframe to apply differencing.
        columns_to_difference : list
            Columns to apply differencing to.
        n_periods : int, optional
            Number of periods to difference (default is 30).

        Returns:
        -------
        pd.DataFrame
            Dataframe with differenced columns.
        """

        if not columns_to_difference:
            print('No features to apply differencing')
            return data  # Return unchanged data if no differencing needed

        for col in columns_to_difference:
            #data[col] = data[col].diff(periods=n_periods).fillna(0) # To replace the original feature
            data[col + '_diff'] = data[col].diff(periods=n_periods).fillna(0) # To keep original feature with Differenced Feature

        return data

# Example usage with a DataFrame
# Assuming 'data' is your DataFrame with a datetime index
# fd = FeatureDifferencing(data)
# non_stationary_features = fd.is_stationary()
# data_with_differences = fd.add_difference(data, non_stationary_features, n_periods=30)
