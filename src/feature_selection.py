import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Define a class function to obtain list of features importance with Random Forest
# List of Highly Correlated Features
class FeatureSelector:
    """
    A class for selecting important features using a Random Forest model.

    Attributes:
    ----------
    X_train : pd.DataFrame
        The training features.
    y_train : pd.Series
        The training target variable.
    X_encoded : pd.DataFrame
        The one-hot encoded training features.
    y_encoded : pd.Series
        The encoded training target variable.
    features : list
        List of feature names.
    feature_original : dict
        Dictionary mapping encoded features to original features.
    importances : np.array
        Array of feature importances.
    df_importance : pd.DataFrame
        DataFrame of features with importances.
    low_importance : pd.DataFrame
        DataFrame of low importance features.

    Methods:
    -------
    preprocess_features()
        Encodes categorical variables using one-hot encoding.
    fit(n_estimators=100, random_state=42)
        Fits the Random Forest model and stores feature importances.
    df_importance_features()
        Returns a DataFrame of features with importance scores.
    low_importance_features(threshold=0.02)
        Returns a list of low importance features.
    plot_high_importance_features(n_features=10)
        Plots the top n_features based on importance scores.
    get_features_list()
        Returns a list of highly correlated features.
    """


    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.X_encoded = None
        self.y_encoded = None
        self.features = []
        self.feature_original = {}
        self.n_estimators = 100
        self.random_state = 42
        self.model = None
        self.df_importance = None
        self.importances = None
        self.low_importance = None

    # Function to preprocess the features
    def preprocess_features(self):
        """
        Encodes categorical variables using one-hot encoding.

        Returns:
        -------
        tuple
            The encoded training features and target variable.
        """
        X = self.X_train
        y = self.y_train
        X_y = pd.concat([X, y], axis=1).dropna()
        X = X_y.drop(y.name, axis=1)
        self.y_encoded = X_y[y.name]
        self.X_encoded = pd.get_dummies(X, drop_first=True)
        self.features = self.X_encoded.columns
        for original_feature in X.columns:
            if X[original_feature].dtype in ['object', 'category']:
                encoded_features = [col for col in self.X_encoded.columns if col.startswith(original_feature + '_')]
                for feature in encoded_features:
                    self.feature_original[feature] = original_feature
            else:
                self.feature_original[original_feature] = original_feature
        return self.X_encoded , self.y_encoded


    # Function to fit the RandomForest model
    def fit(self, n_estimators=100, random_state=42):
        """
        Fits the Random Forest model and stores feature importances.

        Parameters:
        ----------
        n_estimators : int, optional
            Number of trees in the forest (default is 100).
        random_state : int, optional
            Random seed (default is 42).

        Returns:
        -------
        FeatureSelector
            The fitted feature selector.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        self.model.fit(self.X_encoded, self.y_encoded)
        self.importances = self.model.feature_importances_
        return self # Allowing method chaining


    # Function to get DataFrame of features with importance
    def df_importance_features(self):
        """
        Returns a DataFrame of features with importance scores.

        Returns:
        -------
        pd.DataFrame
            DataFrame of features with importance scores.
        """

        if self.importances is None:
            raise ValueError("Model has not been fitted.")
        importance_data = {
            "Encoded Feature": self.features,
            "Original Feature": [self.feature_original[f] for f in self.features],
            "Importance": self.importances
        }
        self.df_importance = pd.DataFrame(importance_data)
        #self.df_importance = self.df_importance.groupby("Original Feature")['Importance'].sum().reset_index()
        self.df_importance = self.df_importance.sort_values(by="Importance", ascending=False).reset_index(drop=True)

        return self.df_importance


    # Function to get list of features with low importance
    def low_importance_features(self, threshold=0.02):
        """
        Returns a list of low importance features.

        Parameters:
        ----------
        threshold : float, optional
            Importance threshold (default is 0.02).

        Returns:
        -------
        pd.DataFrame
            DataFrame of low importance features.
        """

        if self.df_importance is None:
            raise ValueError("Feature importances have not been calculated or are empty. Ensure df_importance_features() is called and successful.")

        # Apply the threshold to the 'Importance' column to create the mask directly
        low_importance_mask = self.df_importance['Importance'] < threshold

        # Use the mask to filter the DataFrame
        self.low_importance = self.df_importance[low_importance_mask]
        self.low_importance = self.low_importance.sort_values(by='Importance', ascending=False).reset_index(drop=True)

        if self.low_importance.empty:
            print(f"No features with low importance based on the threshold <= {threshold}.")
            return None
        else:
            print(f"Low importance features based on the threshold <= {threshold}:")
            return self.low_importance


    # Function to plot top n_features importance
    def plot_high_importance_features(self, n_features=10):
        """
        Plots the top n_features based on importance scores.

        Parameters:
        ----------
        n_features : int, optional
            Number of top features to plot (default is 10).

        Returns:
        -------
        pd.DataFrame
            DataFrame of the top n_features.
        """
        high_importance = self.df_importance_features().head(n_features)
        if not high_importance.empty:
            plt.figure(figsize=(10, 6))
            sns.set_style('whitegrid')
            sns.barplot(x='Importance', y='Original Feature', data=high_importance)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Top {n_features} High Feature Importance Variables')
            plt.xlabel('Feature Importance')
            plt.ylabel('Features')
            plt.show()
            return high_importance
        else:
            print("No features with high importance.")
            return None

# Example usage:
# rf_selector = RandomForestFeatureSelector(X_train, y_train)
# rf_selector.fit()
# print("Low importance features:", rf_selector.low_importance_features())
# rf_selector.plot_high_importance_features(n_features=10)
#==========================================================================

# Define Function for Highly Correlated Features list
    def get_features_list(self):
      """
        Returns a list of highly correlated features.

        Returns:
        -------
        list
            List of highly correlated features.
      """
      self.correlated_features = None

      """
      List of Highly Correlated Features of Related Features
      """
      # Combine data
      data = pd.concat([self.X_train, self.y_train], axis=1).dropna()
      X_data = data.drop(self.y_train.name, axis=1)

      # Correlation tables for all numerical variables with 'dsp_efficiency'
      data_num = data.select_dtypes(include=['int64', 'float64'])
      corr_matrix = data_num.corr(method='spearman')
      corr_df = pd.DataFrame(corr_matrix[self.y_train.name])
      corr_df['absolute_corr'] = corr_df[self.y_train.name].abs()
      corr_df.sort_values(by='absolute_corr', ascending=False)


      # Obtain the columns labels for 'rainfall'
      rainfall_grp = [col for col in data.columns if 'rainfall' in col.lower()]

      # Obtain the columns labels for 'rainfall'
      temp_grp = [col for col in data.columns if 'temp' in col.lower()]

      # Obtain the columns labels for 'wind speed'
      wind_grp = [col for col in data.columns if 'wind speed' in col.lower()]

      # Obtain the columns labels for 'pm25_'
      pm25_grp = [col for col in data.columns if 'pm25_' in col.lower()]

      # Obtain the columns labels for 'psi_'
      psi_grp = [col for col in data.columns if 'psi_' in col.lower()]

      # Subset corr_df by respective group
      low_corr = []

      # Filter for lower correlated related features against the target variable
      for grp in [temp_grp, wind_grp, pm25_grp, psi_grp]:
        grp_df = corr_df.loc[grp].sort_values(by='absolute_corr', ascending=False)
        print(grp_df)
        print('List of lower correlated variables %s' %grp_df.index.tolist()[1:])
        print('===========\n')
        low_corr.append(grp_df.index.tolist()[1:])

      # Convert from list of list to list
      low_corr = [item for sublist in low_corr for item in sublist]

      # Remove duplicates
      low_corr = list(set(low_corr))


      self.correlated_features = low_corr
      return self.correlated_features

