import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class BestModel_Interpretation:
    def __init__(self, model, X, preprocessor):
        """
        A class for interpreting the best machine learning model by analyzing feature importances.

        Attributes:
        ----------
        model : sklearn.pipeline.Pipeline
            The trained model pipeline.
        X : pd.DataFrame
            The input features dataframe.
        preprocessor : sklearn.compose.ColumnTransformer
            The preprocessing pipeline.

        Methods:
        -------
        original_features()
            Maps encoded features back to their original names.
        extract_importances_or_coefficients()
            Extracts feature importances or coefficients based on the model type.
        create_df_features()
            Returns a DataFrame of features with their aggregated importance or coefficients.
        plot_high_importance_features(n_features=10)
            Plots the top n_features based on their importance or coefficients.
        """

        self.model = model
        self.X = X # Dataframe of Features only
        self.preprocessor = preprocessor.fit(self.X)
        self.feature_original = {}  # To map encoded names back to original names
        self.importances = None  # Store importances or coefficients
        self.df_features = None  # Store DataFrame of features with importances or coefficients


    def original_features(self):
        """
        Uses the preprocessor to transform X and maps encoded features back to their original names.

        Returns:
        -------
        dict
            Dictionary mapping encoded features to original features.
        """

        # Mapping features back to original
        feature_names_transformed = self.preprocessor.get_feature_names_out()

        for original_feature in self.X.columns:
            if self.X[original_feature].dtype == 'object' or self.X[original_feature].dtype is np.dtype('O'):
                # This is for handling categorical columns that are typically one-hot encoded
                encoded_features = [col for col in feature_names_transformed if col.startswith("cat__"+original_feature + '_')]
                for feature in encoded_features:
                    self.feature_original[feature] = original_feature
            else:
                # This is for handling numerical columns that are typically scaled or imputed
                self.feature_original["num__"+original_feature] = original_feature

        return self.feature_original


    def extract_importances_or_coefficients(self):
        """
        Extracts feature importances or coefficients based on the model type.

        Returns:
        -------
        np.array
            Array of feature importances or coefficients.
        """
        # Make sure the model is correctly set and has named_steps
        if self.model and hasattr(self.model, 'named_steps') and 'model' in self.model.named_steps:
            model_step = self.model.named_steps['model']
            if hasattr(model_step, 'feature_importances_'):
                self.importances = model_step.feature_importances_
            elif hasattr(model_step, 'coef_'):
                self.importances = np.abs(model_step.coef_.flatten()) # Flatten in case of multi-dimensional coef_
            else:
                raise ValueError("Model does not have feature_importances_ or coef_ attribute.")
        else:
            raise ValueError("The model pipeline is not correctly configured with 'named_steps'.")

        return self.importances



    def create_df_features(self):
        """
        Returns a DataFrame of features with their aggregated importance or coefficients.

        Returns:
        -------
        pd.DataFrame
            DataFrame of features with their aggregated importance or coefficients.
        """
        X_encoded = self.preprocessor.transform(self.X)
        feature_names_transformed = self.preprocessor.get_feature_names_out()
        X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names_transformed)

        try:
            # Ensure there are importances to work with
            if self.importances is None:
              self.extract_importances_or_coefficients()  # Ensure importances are extracted

            features = [col for col in X_encoded_df.columns]  # Assuming X_encoded_df is a DataFrame
            importances = self.importances

            if importances is None:
              print("No importances were extracted. Please check the model's compatibility.")
              return pd.DataFrame()  # Return an empty DataFrame if no importances

            features_data = {
                "Encoded Feature": features,
                "Original Feature": [self.original_features().get(f, f) for f in features],
                "Importance": importances
            }
            self.df_features = pd.DataFrame(features_data)
            self.df_features = self.df_features.groupby("Original Feature")['Importance'].sum().reset_index()
            self.df_features = self.df_features.sort_values(by="Importance", ascending=False).reset_index(drop=True)

            return self.df_features

        except Exception as e:
            print(f"Error in creating feature DataFrame: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on error


    def plot_high_importance_features(self, n_features=10):
        """
        Plots the top n_features based on their importance or coefficients.

        Parameters:
        ----------
        n_features : int, optional
            Number of top features to plot (default is 10).

        Returns:
        -------
        pd.DataFrame
            DataFrame of the top n_features by importance.
        """
        df_importance = self.create_df_features().head(n_features)
        if not df_importance.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Original Feature', data=df_importance)
            plt.title(f'Top {n_features} High Feature Importance or Coefficients Variables')
            plt.xlabel('Feature Importance or Coefficients')
            plt.ylabel('Features')
            plt.show()

            return df_importance
        else:
            print("No features with high importance or significant coefficients.")
            return None

    """
    Usage example:
    --------------
    1. Initialize the BestModel_Interpretation with the best model, feature dataframe, and preprocessor.
    2. Use `original_features` to map encoded features back to their original names.
    3. Use `extract_importances_or_coefficients` to get the importances or coefficients of features.
    4. Use `create_df_features` to create a DataFrame of features with their importances or coefficients.
    5. Use `plot_high_importance_features` to plot the top n features by importance.
    """

