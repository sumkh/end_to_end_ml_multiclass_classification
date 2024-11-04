import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import requests
import io
import re
from datetime import datetime
import calendar
from sklearn.preprocessing import OrdinalEncoder, label_binarize
from sklearn.model_selection import train_test_split

# Define Class Object for Data Preparation
class DataPreparation:
    """
    A class to handle data preparation tasks, including data cleaning and train-test splitting.

    Attributes:
    ----------
    weather_data : pd.DataFrame
        The weather data.
    air_quality_data : pd.DataFrame
        The air quality data.
    config : dict
        Configuration settings for data preparation.

    Methods:
    -------
    clean_data()
        Cleans the data and performs feature engineering.
    split_data()
        Splits the data into training and testing sets.
    """

    def __init__(self, weather_data, air_quality_data, config):
        self.data = pd.DataFrame()
        self.weather_data = weather_data
        self.air_quality_data = air_quality_data
        self.config = config

    """
    ### Cleaning the data
    """
    # Define function to clean data
    def clean_data(self):
      """
        Cleans the data and performs feature engineering.

        - Handles missing values and outliers.
        - Converts data types as necessary.
        - Creates new features from existing ones.

        Returns:
        -------
        pd.DataFrame
            The cleaned and feature-engineered data.
        """

      # Copy data
      weather_data = self.weather_data.copy()
      air_quality_data = self.air_quality_data.copy()

      # Remove '-' and '--' from the values in weather_data and air_quality_data
      weather_data = weather_data.replace('-', np.nan)
      weather_data = weather_data.replace('--', np.nan)

      self.air_quality_data = self.air_quality_data.replace('-', np.nan)
      self.air_quality_data = self.air_quality_data.replace('--', np.nan)

      # Convert 'date' to datetime data type
      weather_data['date'] = pd.to_datetime(weather_data['date'], format = '%d/%m/%Y')
      air_quality_data['date'] = pd.to_datetime(air_quality_data['date'], format = '%d/%m/%Y')

      # Obtain column labels from weather_data, excluding the first ('date') and last 3 columns
      weather_num_labels = weather_data.columns.values[1:-3]

      # Convert the numeric columns to numeric
      weather_data.loc[:, weather_num_labels] = weather_data.loc[:, weather_num_labels].apply(pd.to_numeric, errors='coerce')

      air_quality_data.iloc[:, 1:] = air_quality_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

      # Remove duplicate rows of weather_data
      weather_data = weather_data.drop_duplicates()

      # Calculate 'max_daily_rainfall'
      weather_data['max_daily_rainfall'] = weather_data[['Daily Rainfall Total (mm)',
                                                        'Highest 60 Min Rainfall (mm)',
                                                        'Highest 120 Min Rainfall (mm)',
                                                        'Highest 30 Min Rainfall (mm)']].max(axis=1)

      # Impute missing values for 'Daily Rainfall Total (mm)' based on 'max_daily_rainfall'
      weather_data['Daily Rainfall Total (mm)']=weather_data['Daily Rainfall Total (mm)'].fillna(weather_data['max_daily_rainfall'])

      # Impute missing values for 'Highest 120 Min Rainfall (mm)' based on 'max_daily_rainfall'
      weather_data['Highest 120 Min Rainfall (mm)']=weather_data['Highest 120 Min Rainfall (mm)'].fillna(weather_data['max_daily_rainfall'])

      # Impute missing values for 'Highest 60 Min Rainfall (mm)' based on 'Highest 120 Min Rainfall (mm)'
      weather_data['Highest 60 Min Rainfall (mm)']=weather_data['Highest 60 Min Rainfall (mm)'].fillna(weather_data['Highest 120 Min Rainfall (mm)'])

      # Impute missing values for 'Highest 30 Min Rainfall (mm)' based on 'Highest 60 Min Rainfall (mm)'
      weather_data['Highest 30 Min Rainfall (mm)']=weather_data['Highest 30 Min Rainfall (mm)'].fillna(weather_data['Highest 60 Min Rainfall (mm)'])

      # Forward fill for each of the rainfall measures if missing value occurs
      weather_data['Highest 30 Min Rainfall (mm)'] = weather_data['Highest 30 Min Rainfall (mm)'].fillna(method='ffill')
      weather_data['Highest 60 Min Rainfall (mm)'] = weather_data['Highest 60 Min Rainfall (mm)'].fillna(method='ffill')
      weather_data['Highest 120 Min Rainfall (mm)'] = weather_data['Highest 120 Min Rainfall (mm)'].fillna(method='ffill')
      weather_data['Daily Rainfall Total (mm)'] = weather_data['Daily Rainfall Total (mm)'].fillna(method='ffill')
      weather_data['max_daily_rainfall'] = weather_data['max_daily_rainfall'].fillna(method='ffill')

      # Replace 'Daily Rainfall Total (mm)' with 'max_daily_rainfall' where ['max_daily_rainfall' > 'Daily Rainfall Total (mm)']
      weather_data['Daily Rainfall Total (mm)'] = weather_data[['max_daily_rainfall', 'Daily Rainfall Total (mm)']].max(axis=1)

      # New Feature: Difference of 'Highest 60 Min Rainfall (mm)' vs 'Highest 30 Min Rainfall (mm)
      weather_data['rf_60min'] = (weather_data['Highest 60 Min Rainfall (mm)'] - weather_data['Highest 30 Min Rainfall (mm)'])

      # New Feature: Difference of 'Highest 120 Min Rainfall (mm)' vs 'Highest 30 Min Rainfall (mm)'
      weather_data['rf_120min'] = (weather_data['Highest 120 Min Rainfall (mm)'] - weather_data['Highest 30 Min Rainfall (mm)'])

      # New Feature: Difference of 'max_daily_rainfall' vs 'Highest 30 Min Rainfall (mm)'
      weather_data['rf_max'] = (weather_data['max_daily_rainfall'] - weather_data['Highest 30 Min Rainfall (mm)'])

      # Forward fill 'Min Temperature (deg C)' for missing values
      weather_data['Min Temperature (deg C)'] = weather_data['Min Temperature (deg C)'].fillna(method='ffill')

      # Forward fill 'Max Temperature (deg C)' for missing values
      weather_data['Maximum Temperature (deg C)'] = weather_data['Maximum Temperature (deg C)'].fillna(method='ffill')

      # Calculate daily mean_temp and convert to numerical
      weather_data['mean_temp'] = weather_data[['Maximum Temperature (deg C)', 'Min Temperature (deg C)']].mean(axis=1)

      # When 'Min Temperature (deg C)' is higher then 'Maximum Temperature (deg C)', take the lower values
      weather_data['min_temp'] = weather_data[['Min Temperature (deg C)', 'Maximum Temperature (deg C)']].min(axis=1)

      # 'Maximum Temperature (deg C)' is lower then 'Min Temperature (deg C)', take the higher values
      weather_data['max_temp'] = weather_data[['Min Temperature (deg C)', 'Maximum Temperature (deg C)']].max(axis=1)

      # Update Feature: Difference of 'Min Temperature (deg C)' vs 'Maximum Temperature (deg C)'
      weather_data['temp_range'] = weather_data['max_temp'] - weather_data['min_temp']

      # Convert units of 'Wet Bulb Temperature' from deg F to deg C
      weather_data['wbt_c'] = (weather_data['Wet Bulb Temperature (deg F)'] - 32) * 5/9

      # Assuming that negative degress C is an error, convert all values to absolute number
      weather_data['wbt_c'] = weather_data['wbt_c'].abs()

      # Convert to absolute values for Min and Max Wind Speed (km/h)
      weather_data['Min Wind Speed (km/h)'] = weather_data['Min Wind Speed (km/h)'].abs()
      weather_data['Max Wind Speed (km/h)'] = weather_data['Max Wind Speed (km/h)'].abs()

      # Impute missing values for Min and Max Wind Speed (km/h) based on forward fill
      weather_data['Min Wind Speed (km/h)'] = weather_data['Min Wind Speed (km/h)'].fillna(method='ffill')
      weather_data['Max Wind Speed (km/h)'] = weather_data['Max Wind Speed (km/h)'].fillna(method='ffill')

      # Calculate daily mean wind speed
      weather_data['mean_wind speed'] = weather_data[['Min Wind Speed (km/h)', 'Max Wind Speed (km/h)']].mean(axis=1)

      # Convert to upper case for 'Wind Direction' and extract only characters
      weather_data['Wind Direction'] = weather_data['Wind Direction'].str.upper()
      weather_data['Wind Direction'] = weather_data['Wind Direction'].str.extract(r'([A-Z]+)', expand=False)

      # Replace values in Wind Direction,
      weather_data['Wind Direction'] = weather_data['Wind Direction'].replace({
          'SOUTHEAST': 'SE',
          'NORTHEAST': 'NE',
          'WEST': 'W',
          'NORTHWEST': 'NW',
          'SOUTH': 'S',
          'EAST': 'E',
          'NORTHWARD': 'N',
          'NORTH': 'N',
          'SOUTHWARD': 'S'
          })

      # Impute missing values for 'Sunshine Duration (hrs)' and 'Cloud Cover (%)' with forward fill
      weather_data['Sunshine Duration (hrs)'] = weather_data['Sunshine Duration (hrs)'].fillna(method='ffill')
      weather_data['Cloud Cover (%)'] = weather_data['Cloud Cover (%)'].fillna(method='ffill')

      # Convert 'Dew Point Category' to upper case
      weather_data['Dew Point Category'] = weather_data['Dew Point Category'].str.upper()

      # Replace values in 'Dew Point Category'
      weather_data['Dew Point Category'] = weather_data['Dew Point Category'].replace({
          'HIGH LEVEL': 'HIGH',
          'H': 'HIGH',
          'VH': 'VERY HIGH',
          'M': 'MODERATE',
          'NORMAL': 'MODERATE',
          'MINIMAL': 'LOW',
          'VL': 'VERY LOW',
          'BELOW AVERAGE': 'LOW',
          'L': 'LOW',
          'EXTREME': 'VERY HIGH'
      })

      # Ordinal encode for Dew Point Category
      weather_encoder = OrdinalEncoder(categories = [['VERY LOW', 'LOW', 'MODERATE', 'HIGH', 'VERY HIGH']])
      weather_data['dew_pt'] = weather_encoder.fit_transform(weather_data[['Dew Point Category']])

      # Obtain the columns labels for 'pm25_' dataframe
      pm25_labels = air_quality_data.columns.values[air_quality_data.columns.str.contains('pm25_')]
      pm25_labels_d = np.append(pm25_labels, 'date')

      # Obtain the columns labels for 'psi_' dataframe
      psi_labels = air_quality_data.columns.values[air_quality_data.columns.str.contains('psi_')]
      psi_labels_d = np.append(psi_labels, 'date')

      # Create dataframe based on the 'pm25_labels' and 'psi_labels'
      pm25_data = air_quality_data.loc[:, pm25_labels_d]
      psi_data = air_quality_data.loc[:, psi_labels_d]

      # Remove rows which all columns (exclude date) are NaN
      pm25_data = pm25_data.dropna(subset = pm25_labels ,how='all')
      psi_data = psi_data.dropna(subset = psi_labels, how='all')

      # Remove duplicate rows of data
      pm25_data = pm25_data.drop_duplicates()
      psi_data = psi_data.drop_duplicates()

      # Create 'pm25_mean' and 'psi_mean' for pm25_data and psi_data, ignore NaN
      pm25_data['pm25_mean'] = pm25_data[pm25_labels].mean(axis=1, skipna=True)
      pm25_data['pm25_mean'] = pd.to_numeric(pm25_data['pm25_mean'])
      psi_data['psi_mean'] = psi_data[psi_labels].mean(axis=1, skipna=True)
      psi_data['psi_mean'] = pd.to_numeric(psi_data['psi_mean'])

      # For each columns of the pm25_data and psi_data (excluding date),
      # fill the NaN with the mean values for each rows of the pm25_data and psi_data
      # based on 'pm25_mean' and 'psi_mean' respectively
      for col in pm25_labels:
        pm25_data[col].fillna(pm25_data['pm25_mean'], inplace=True)
        pm25_data[col] = pd.to_numeric(pm25_data[col])

      for col in psi_labels:
        psi_data[col].fillna(psi_data['psi_mean'], inplace=True)
        psi_data[col] = pd.to_numeric(psi_data[col])

      # Merge 'pm25_data' and 'psi_data' into a single dataframe based on date.
      air_data = pd.merge(pm25_data, psi_data, on='date', how='outer')

      # Merge both weather_data and air_quality_data into a single dataframe based on "date" column
      data = pd.merge(weather_data, air_data, on='date', how='left')

      # Ordinal encode Target Variable: Daily Solar Panel Efficiency
      target_encoder = OrdinalEncoder(categories = [['Low', 'Medium', 'High']])
      data['dsp_efficiency'] = target_encoder.fit_transform(data[['Daily Solar Panel Efficiency']])


      self.data = data

      return self.data

    """
    ### Preparing for Train and Test Sets
    """
    # Define a function to split the data into Train and Test sets
    def split_data(self):
        """
        Splits the data into training and testing sets based on the configuration settings.

        Returns:
        -------
        tuple: (X_train, X_test, y_train, y_test)
            The training and testing feature and target datasets.
        """

        # Copy data
        f_data = self.data.copy()

        # Drop features that are not required
        f_data = f_data.drop(columns=['Daily Solar Panel Efficiency', 'Dew Point Category',
                                      'Highest 60 Min Rainfall (mm)', 'Highest 120 Min Rainfall (mm)', 'Daily Rainfall Total (mm)',
                                      'Min Temperature (deg C)', 'Maximum Temperature (deg C)',
                                      'mean_temp',
                                      ], axis=1)

        # Splitting the data while maintaining time order
        f_data = f_data.set_index('date').sort_index()

        train_size = int(len(f_data) * self.config['train_size'])
        train, test = f_data.iloc[:train_size], f_data.iloc[train_size:]

        # Split the data into Train and Test sets
        X_train = train.drop('dsp_efficiency', axis=1)
        y_train = train['dsp_efficiency']

        X_test = test.drop('dsp_efficiency', axis=1)
        y_test = test['dsp_efficiency']

        return X_train, X_test, y_train, y_test
