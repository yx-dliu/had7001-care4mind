import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

def get_scores(y_pred, y):
  """
    Computes evaluation metrics for a classification task.

    Parameters:
        y_pred (array-like): Predicted class labels.
        y (array-like): True class labels.

    Returns:
        tuple:
            - conf_matrix (ndarray): Confusion matrix.
            - class_report (str): Text summary of precision, recall, F1-score, and support for each class.
            - acc (float): Overall accuracy score.
  """
  conf_matrix = confusion_matrix(y, y_pred) # This line and the following two should be indented with 4 spaces.
  class_report = classification_report(y, y_pred)
  acc = accuracy_score(y, y_pred)
  return conf_matrix, class_report, acc

def print_scores(conf_matrix, class_report):
  """
    Prints the confusion matrix and classification report for a classification model.

    Parameters:
        conf_matrix (ndarray): Confusion matrix showing true vs. predicted classifications.
        class_report (str): Classification report including precision, recall, F1-score, and support.
  """
  print("Confusion Matrix:")
  print(conf_matrix)
  print("\nClassification Report:")
  print(class_report)

def plot_confusion_matrix(conf_matrix):
  """
    Plots a heatmap of the confusion matrix using Seaborn.

    Parameters:
        conf_matrix (ndarray): Confusion matrix of true vs. predicted class labels.

    Returns:
        None
  """
  sns.heatmap(conf_matrix, annot=True, fmt='d')
  plt.ylabel('Actual')
  plt.xlabel('Predicted')
  plt.show()

def get_auc_values(y_prob, y):
  """
  DOCSTRING
  """
  fpr, tpr, thresholds = roc_curve(y, y_prob)
  roc_auc = roc_auc_score(y, y_prob)
  return fpr, tpr, roc_auc

def plot_roc_curve(fpr, tpr, roc_auc, title = "ROC AUC Curve"):
  """
    Computes the false positive rate, true positive rate, and AUC score for a binary classifier.

    Parameters:
        y_prob (array-like): Predicted probabilities for the positive class.
        y (array-like): True binary class labels.

    Returns:
        tuple:
            - fpr (array): False positive rates.
            - tpr (array): True positive rates.
            - roc_auc (float): Area Under the ROC Curve (AUC) score.
  """
  plt.figure()
  plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], 'k--')  # Plot diagonal
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate (FPR)')
  plt.ylabel('True Positive Rate (TPR)')
  plt.title(title)
  plt.legend(loc='lower right')
  plt.show()

def evaluate_model(model, X_train, y_train, X_test, y_test):
  """
    Evaluates a binary classification model using training and test data.

    This function computes and displays accuracy, confusion matrix, classification report,
    and AUC-ROC for both training and test sets. It assumes the model has both
    `predict()` and `predict_proba()` methods.

    Parameters:
        model: A fitted scikit-learn-style classification model.
        X_train (array-like): Feature matrix for the training set.
        y_train (array-like): True labels for the training set.
        X_test (array-like): Feature matrix for the test set.
        y_test (array-like): True labels for the test set.

    Returns:
        tuple:
            - y_pred_train (array): Predicted class labels for the training set.
            - y_pred_test (array): Predicted class labels for the test set.
            - y_prob_train (array): Predicted probabilities for the positive class (training set).
            - y_prob_test (array): Predicted probabilities for the positive class (test set).
  """
  y_pred_train = model.predict(X_train)
  y_pred_test = model.predict(X_test)
  y_prob_train = model.predict_proba(X_train)[:, 1]
  y_prob_test = model.predict_proba(X_test)[:, 1]

  print(f"Train accuracy score: {accuracy_score(y_train, y_pred_train)}")
  print(f"Test accuracy score: {accuracy_score(y_test, y_pred_test)}")
  print('\n')

  # Calculate evaluation metrics and scores for the test and training sets
  conf_matrix_train, class_report_train, acc_train = get_scores(y_pred_train, y_train)
  conf_matrix_test, class_report_test, acc_test = get_scores(y_pred_test, y_test)

  # Print scores and evaluation metrics for the training set
  print("======== Training Set ==========")
  print_scores(conf_matrix_train, class_report_train)
  plot_confusion_matrix(conf_matrix_train) # plot the confusion matrix for the training set using the function we created above

  fpr_train, tpr_train, roc_auc_train = get_auc_values(y_prob_train, y_train)
  print(f"Training AUC: {roc_auc_train:.4f}")
  plot_roc_curve(fpr_train, tpr_train, roc_auc_train)

  # Print scores and evaluation metrics for the test set
  print('======== Test Set ==========')
  print_scores(conf_matrix_test, class_report_test)
  plot_confusion_matrix(conf_matrix_test) # plot the confusion matrix for the test set using the function we created above

  fpr_test, tpr_test, roc_auc_test = get_auc_values(y_prob_test, y_test)
  print(f"Testing AUC: {roc_auc_test:.4f}")
  plot_roc_curve(fpr_test, tpr_test, roc_auc_test)

  return y_pred_train, y_pred_test, y_prob_train, y_prob_test