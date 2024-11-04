import numpy as np
from sklearn.metrics import f1_score, classification_report, precision_recall_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

class ThresholdFinder:
    """
    A class to find the optimal decision thresholds for multi-class classification.

    Attributes:
    ----------
    y_test : np.array
        True labels for the test data.
    y_pred_prob : np.array
        Predicted probabilities for the test data.

    Methods:
    -------
    find_best_thresholds()
        Finds the best thresholds for each class using F1 score.
    prc_plot_multiclass()
        Plots Precision-Recall curves and finds the optimal thresholds.
    predict_optimal_thresholds(thresholds)
        Makes predictions based on the optimal thresholds.
    """
    def __init__(self, y_test, y_pred_prob):
        # Initialize with test labels and predicted probabilities
        self.y_test = y_test
        self.y_pred_prob = y_pred_prob

    # Finding best threshold using a range of possible thresholds
    def find_best_thresholds(self):
        """
        Finds the best thresholds for each class using F1 score.

        Returns:
        -------
        list
            Best thresholds for each class.
        """
        # Binarize the labels for one-vs-rest classification
        y_test_class = label_binarize(self.y_test, classes=[0, 1, 2])

        # Define a range of possible thresholds
        thresholds = np.arange(0.1, 1, 0.01)
        best_thresholds = []

        for i in range(self.y_pred_prob.shape[1]):
            best_f1 = 0
            best_thresh = 0
            for thresh in thresholds:
                # Binarize predictions based on the threshold
                y_pred = (self.y_pred_prob[:, i] >= thresh).astype(int)
                # Calculate F1 score
                f1 = f1_score(y_test_class[:, i], y_pred)
                # Update best F1 score and threshold
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            best_thresholds.append(best_thresh)

        return best_thresholds

    # Ploting PRC Plot and Finding best threshold using a range of possible thresholds (Better Method)
    def prc_plot_multiclass(self):
        """
        Plots Precision-Recall curves and finds the optimal thresholds.

        Returns:
        -------
        list
            Optimal thresholds based on highest F1 score.
        """
        # Binarize the labels for plotting
        y_test_class = label_binarize(self.y_test, classes=[0, 1, 2])
        plt.figure(figsize=(12, 8))
        colour = ['r', 'g', 'b']
        highf1_thresholds = []

        for i in range(self.y_pred_prob.shape[1]):
            y_true_class = y_test_class[:, i]
            y_proba_class = self.y_pred_prob[:, i]

            # Compute precision, recall, and thresholds
            precisions, recalls, thresholds = precision_recall_curve(y_true_class, y_proba_class)

            # Calculate F1 scores and find index of highest F1 score
            f1_scores = np.nan_to_num(2 * (precisions * recalls) / (precisions + recalls))
            #f1_scores = np.nan_to_num(f1_scores)
            highf1_idx = np.argmax(f1_scores[:-1])
            highf1_threshold = thresholds[highf1_idx]
            highf1 = f1_scores[highf1_idx]
            highf1_thresholds.append(highf1_threshold)

            # Calculate F1 score for best threshold
            y_pred = (self.y_pred_prob[:, i] >= highf1_threshold).astype(int)
            optimal_f1 = f1_score(y_test_class[:, i], y_pred)

            # Plot precision, recall, and best threshold marker
            plt.plot(thresholds, precisions[:-1], label=f"Precision - Class {i}", linewidth=1)
            plt.plot(thresholds, recalls[:-1], label=f"Recall - Class {i}", linewidth=1)
            plt.plot(thresholds, f1_scores[:-1], label=f"F1 score - Class {i}", linestyle=':', color=colour[i], linewidth=1)
            #plt.plot(best_thresholds[i], optimal_f1, 'ro', label=f'Best Threshold = {best_thresholds[i]:.2f} (Class {i})')
            plt.plot(highf1_threshold, optimal_f1, color=colour[i], marker='o', label=f'Optimal F1 Threshold = {highf1_threshold:.2f} (Class {i})')
            plt.axvline(x=highf1_threshold, color=colour[i], linestyle=':', alpha = 0.6, label=f'Optimal F1 Threshold = {highf1_threshold:.2f} (Class {i})')

        plt.title('Precision, Recall, and F1 Score vs. Threshold for Each Class')
        plt.xlabel("Threshold", fontsize=16)
        plt.ylabel("Score", fontsize=16)
        plt.legend(loc="best", fontsize=10)
        plt.grid(True)
        plt.ylim([0, 1])
        plt.show()

        # Returning the best_threshold based on best F1 score
        return highf1_thresholds


    def predict_optimal_thresholds(self, thresholds):
        """
        Makes predictions based on the optimal thresholds.

        Parameters:
        ----------
        thresholds : list
            Optimal thresholds for each class.

        Returns:
        -------
        np.array
            Predicted labels based on optimal thresholds.
        """
        # Initialize list to store predictions for each class
        y_pred = []
        for j in range(self.y_pred_prob.shape[1]):
            # Apply threshold to get predictions for class 'j'
            class_predictions = (self.y_pred_prob[:, j] >= thresholds[j]).astype(int)
            y_pred.append(class_predictions)
        # Convert list of predictions into a 2D array and find the argmax
        y_pred = np.array(y_pred).T
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred



# Example usage
# finder = ThresholdFinder(y_test, y_pred_prob)
# best_thresholds = finder.find_best_thresholds()
# print("Best thresholds per class:", best_thresholds)
# optimal_y_pred = finder.predict_optimal_thresholds(best_thresholds)
# print("Classification Report with Optimized Thresholds:")
# print(classification_report(finder.y_test, optimal_y_pred))
# highf1_thresholds = finder.prc_plot_multiclass(optimal_y_pred, best_thresholds)
