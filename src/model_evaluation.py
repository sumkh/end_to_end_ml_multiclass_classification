import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    auc,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix
    )
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score

class ModelEvaluation:
    """
    A class for evaluating machine learning models.

    Attributes:
    ----------
    results_df : pd.DataFrame
        DataFrame of model training results.
    X_train : pd.DataFrame
        The training features.
    y_train : pd.Series
        The training target variable.
    X_test : pd.DataFrame
        The testing features.
    y_test : pd.Series
        The testing target variable.

    Methods:
    -------
    evaluate_models()
        Evaluates the models and updates the results DataFrame.
    plot_roc_prc()
        Plots ROC and Precision-Recall curves for the models.
    classification_and_cm()
        Prints the classification report and plots the confusion matrix.
    """
    def __init__(self, results_df, X_train, y_train, X_test, y_test):
        self.results_df = results_df
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    # Evaluate models
    def evaluate_models(self):
        """
        Evaluates the models and updates the results DataFrame with ROC AUC and F1 scores.

        Returns:
        -------
        pd.DataFrame
            The updated results DataFrame.
        """

        # Iterate over the models and calculate the ROC AUC score for the train and test sets
        for index, row in self.results_df.iterrows():
            model_name = row['model']
            model_data = row['model_data']
            best_params = row['best_params']

            print(f"Model {index}: {model_name}")
            print(f"Best Parameters: {best_params}\n")

            # Make predictions on the train and test set
            probas_train = model_data.predict_proba(self.X_train)
            probas_test = model_data.predict_proba(self.X_test)
            y_pred = model_data.predict(self.X_test)
            

            # Calculate evaluation metrics
            roc_auc_train = roc_auc_score(self.y_train, probas_train, multi_class='ovr')
            roc_auc_test = roc_auc_score(self.y_test, probas_test, multi_class='ovr')

            #roc_auc_train = roc_auc_score(self.y_train, model_data.predict_proba(self.X_train)[:, 1])
            #roc_auc_test = roc_auc_score(self.y_test, probas_test[:, 1])

            # Update the results dataframe
            self.results_df.loc[index, 'roc_auc_train'] = roc_auc_train
            self.results_df.loc[index, 'roc_auc_test'] = roc_auc_test

            # Finding best threshold
            # Binarize the labels for plotting
            y_test_class = label_binarize(self.y_test, classes=[0, 1, 2])

            highf1_thresholds = []
            class_optimal_f1 = []

            for i in range(probas_test.shape[1]):
                y_true_class = y_test_class[:, i]
                y_proba_class = probas_test[:, i]

                # Compute precision, recall, and thresholds
                precisions, recalls, thresholds = precision_recall_curve(y_true_class, y_proba_class)

                # Calculate F1 scores and find index of highest F1 score
                f1_scores = np.nan_to_num(2 * (precisions * recalls) / (precisions + recalls))
                #f1_scores = np.nan_to_num(f1_scores)
                highf1_idx = np.argmax(f1_scores[:-1])
                highf1_threshold = thresholds[highf1_idx]
                highf1 = f1_scores[highf1_idx]

                # Append Optimal F1 Threshold for the Class[i]
                highf1_thresholds.append(highf1_threshold)

                # Calculate F1 score for best threshold
                y_pred = (probas_test[:, i] >= highf1_threshold).astype(int)
                optimal_f1 = f1_score(y_test_class[:, i], y_pred)

                # Append Optimal F1 Threshold's F1 Score for the Class[i]
                class_optimal_f1.append(optimal_f1)

                self.results_df.loc[index, f'class{i} F1 score'] = optimal_f1
                self.results_df.loc[index, f'class{i} Threshold'] = highf1_threshold

            # Average F1 Score for all 3 classes
            self.results_df['mean_F1_score_test'] = self.results_df[[f'class{i} F1 score' for i in range(3)]].mean(axis=1)
        
        results_df_sorted = self.results_df.sort_values(by='roc_auc_test', ascending=False)
        results_df_sorted = results_df_sorted.reset_index(drop=True)
        
        print('\n--------- ROC AUC Scores ---------\n')
        print(results_df_sorted[['model', 'best_validation_score', 
                                 'roc_auc_train', 'roc_auc_test']])

        print('\n--------- F1 Scores ---------\n')
        print(results_df_sorted[['model', 
                                 'mean_F1_score_test', 
                                 'class0 F1 score', 'class1 F1 score', 'class2 F1 score']])
        
        print('\n--------- Thresholds ---------\n')
        print(results_df_sorted[['model', 
                                 'class0 Threshold', 'class1 Threshold', 'class2 Threshold']])


        # Print the best model
        print('\n--------- Best Model ---------\n')
        print (f"Best Model: {results_df_sorted.iloc[0]['model']}")
        print (f"Best Parameters: {results_df_sorted.iloc[0]['best_params']}")
        print (f"Best Validation Score: {results_df_sorted.iloc[0]['best_validation_score']}")

        return results_df_sorted

    # Plot ROC and PRC Curve
    def plot_roc_prc(self):
        """
        Plots ROC and Precision-Recall curves for the models.
        """
        # Binarize the output
        y = label_binarize(self.y_test, classes=[0, 1, 2])
        n_classes = y.shape[1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Iterate over the models and plot the ROC and Precision-Recall curves
        for index in self.results_df.index:
            model_name = self.results_df.loc[index, 'model']
            model_data = self.results_df.loc[index, 'model_data']

            # Calculate the probability scores for each class
            y_probas = model_data.predict_proba(self.X_test)

            # ROC and Precision-Recall for each class
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y[:, i], y_probas[:, i], pos_label=1)
                roc_auc = auc(fpr, tpr)
                ax1.plot(fpr, tpr, lw=2, label=f'{model_name} class {i} (AUC = {roc_auc:.2f})')

                precision, recall, _ = precision_recall_curve(y[:, i], y_probas[:, i], pos_label=1)
                average_precision = average_precision_score(y[:, i], y_probas[:, i])
                ax2.plot(recall, precision, lw=2, label=f'{model_name} class {i} (AvgPrec = {average_precision:.2f})')

        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic')
        ax1.legend(loc="lower right")

        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlim([0.0, 1.0])
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")

        plt.tight_layout()
        plt.show()

        return None


    # Define function to print Classification Report and Confusion Matrix
    def classification_and_cm(self):
        """
        Prints the classification report and plots the confusion matrix for the best model.
        """

        # Get the best model
        best_model_data = self.results_df.iloc[0]['model_data']
        #best_model_data = self.results_df.loc[index, 'model_data']

        # Get the predictions
        y_pred = best_model_data.predict(self.X_test)

        # Classification Report
        print(classification_report(self.y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

        return None

