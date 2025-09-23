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

def plot_confusion_matrix(conf_matrix, save_res = False, save_path = 'confusion_matrix.png'):
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

  if save_res:
    plt.savefig(save_path, bbox_inches='tight')

  plt.show()
  plt.close()

def get_auc_values(y_prob, y):
  """
  DOCSTRING
  """
  fpr, tpr, thresholds = roc_curve(y, y_prob)
  roc_auc = roc_auc_score(y, y_prob)
  return fpr, tpr, roc_auc

def plot_roc_curve(fpr, tpr, roc_auc, title = "ROC AUC Curve", save_res = False, save_path = 'roc_curve.png'):
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

  if save_res:
    plt.savefig(save_path, bbox_inches='tight')

  plt.show()
  plt.close()

def evaluate_split(y_true, y_pred, y_prob, split_name, save_res=False, plot_names=None):
    """
    Evaluate model performance on a single dataset split.

    Computes confusion matrix, classification report, and ROC AUC,
    prints metrics, and optionally saves confusion matrix and ROC curve plots.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted class labels.
        y_prob (array-like): Predicted probabilities for the positive class.
        split_name (str): Name of the split ("train" or "test") for labeling.
        save_res (bool): Whether to save plots.
        plot_names (dict, optional): Base filenames for saved plots.

    Returns:
        None
    """
    conf_matrix, class_report, acc = get_scores(y_pred, y_true)

    print(f"======== {split_name.capitalize()} Set ==========")
    print_scores(conf_matrix, class_report)

    # Confusion matrix
    if save_res and plot_names is not None:
        save_path = f"{plot_names['confusion_matrix']}_{split_name}.png"
        plot_confusion_matrix(conf_matrix, save_res, save_path)
    else:
        plot_confusion_matrix(conf_matrix, save_res)

    # ROC AUC
    fpr, tpr, roc_auc = get_auc_values(y_prob, y_true)
    print(f"{split_name.capitalize()} AUC: {roc_auc:.4f}")

    if save_res and plot_names is not None:
        save_path = f"{plot_names['roc_curve']}_{split_name}.png"
        plot_roc_curve(fpr, tpr, roc_auc, save_res, save_path)
    else:
        plot_roc_curve(fpr, tpr, roc_auc, save_res)

def evaluate_model(model, X_train, y_train, X_test, y_test, save_res=False, plot_names=None):
    """
    Evaluate a binary classification model on training and test sets.

    Generates predictions and probabilities, prints accuracy, and calls
    `evaluate_split` to compute confusion matrices, classification reports,
    and ROC curves for both train and test data. Optionally saves plots.

    Parameters:
        model: Fitted scikit-learn style classifier with `predict` and `predict_proba`.
        X_train (array-like): Training feature matrix.
        y_train (array-like): True training labels.
        X_test (array-like): Test feature matrix.
        y_test (array-like): True test labels.
        save_res (bool): Whether to save plots.
        plot_names (dict, optional): Base filenames for saved plots.

    Returns:
        tuple:
            y_pred_train (array): Predicted labels for training set.
            y_pred_test (array): Predicted labels for test set.
            y_prob_train (array): Predicted probabilities for the positive class (training set).
            y_prob_test (array): Predicted probabilities for the positive class (test set).
    """
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    print(f"Train accuracy score: {accuracy_score(y_train, y_pred_train)}")
    print(f"Test accuracy score: {accuracy_score(y_test, y_pred_test)}\n")

    evaluate_split(y_train, y_pred_train, y_prob_train, "train", save_res, plot_names)
    evaluate_split(y_test, y_pred_test, y_prob_test, "test", save_res, plot_names)

    return y_pred_train, y_pred_test, y_prob_train, y_prob_test