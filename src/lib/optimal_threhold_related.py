import numpy as np

from numpy import sqrt, argmax
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, precision_recall_curve, balanced_accuracy_score


def get_optimal_threshold_f1(y_true, y_pred_score):
    """
    find the optimal threshold to cut-off the predicted score(y_pred_score) for classification.
    y_true: true labels
    y_pred_score: predicted scores
    return the optimal threshold and best F1 score
    """
    current_best_f1 = 0
    best_threshold = 0

    for threshold in np.arange(0, 1, 0.1):
        y_pred = np.copy(y_pred_score)
        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred < threshold] = 0
        f1 = f1_score(y_true, y_pred)
        print('threshold:{}, F1:{}'.format(threshold, f1))

        if f1 > current_best_f1:
            current_best_f1 = f1
            best_threshold = threshold

    return best_threshold, current_best_f1


def get_optimal_threshold_Jvalue(y_true, y_pred_score):
    """
    find the optimal threshold to cut-off the predicted score(y_pred_score) for classification.
    y_true: true labels
    y_pred_score: predicted scores
    return the optimal threshold and best J score. This threshold is optimal for ROC curve (which measures the general
    performance of the model)
    """
    current_best_j = 0
    best_threshold = 0

    for threshold in np.arange(0, 1, 0.1):
        tpr = calculate_tpr(y_true, y_pred_score, threshold)
        fpr, fnr = calculate_fpr_fnr(y_true, y_pred_score, threshold)
        jvalue = tpr - fpr
        print('threshold:{}, J-value:{}'.format(threshold, jvalue))

        if jvalue > current_best_j:
            current_best_j = jvalue
            best_threshold = threshold

    return best_threshold, current_best_j


def get_optimal_threshold_Gmean(y_true, y_pred_score):
    """
    find the optimal threshold to cut-off the predicted score(y_pred_score) for classification.
    y_true: true labels
    y_pred_score: predicted scores
    return the optimal threshold and best geometric mean. This threshold is optimal for ROC curve (which measures the general
    performance of the model)
    """
    current_best_G = 0
    best_threshold = 0

    for threshold in np.arange(0, 1, 0.1):
        tpr = calculate_tpr(y_true, y_pred_score, threshold)
        fpr, fnr = calculate_fpr_fnr(y_true, y_pred_score, threshold)
        gmeans = sqrt(tpr * (1 - fpr))
        print('threshold:{}, G-mean:{}'.format(threshold, gmeans))

        if gmeans > current_best_G:
            current_best_G = gmeans
            best_threshold = threshold

    return best_threshold, current_best_G


def get_optimal_threshold_Fmeasure(y_true, y_pred_score):
    """
    find the optimal threshold to cut-off the predicted score(y_pred_score) for classification.
    y_true: true labels
    y_pred_score: predicted scores
    return the optimal threshold and best F measure. This threshold is optimal for PRC curve (which focuses on the
    performance of a classifier on the positive (minority class) only)
    """
    current_best_F = 0
    best_threshold = 0

    for threshold in np.arange(0, 1, 0.1):
        tpr = calculate_tpr(y_true, y_pred_score, threshold)
        fpr, fnr = calculate_fpr_fnr(y_true, y_pred_score, threshold)
        precision = tpr / (tpr + fpr)
        recall = tpr / (tpr + fnr)
        # precision, recall, thresholds = precision_recall_curve(y_true, y_pred_score)
        fscore = (2 * precision * recall) / (precision + recall)
        # ix = argmax(fscore)
        print('threshold:{}, F-measure:{}'.format(threshold, fscore))

        if fscore > current_best_F:
            current_best_F = fscore
            best_threshold = threshold

    return best_threshold, current_best_F


def calculate_balanced_accuracy (y_true, y_pred_score, threshold=0.5):
    """
    compute balanced accuracy score (1/2 (sensitivity + specificity))
    """
    y_pred = np.copy(y_pred_score)
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    ba = balanced_accuracy_score(y_true, y_pred)
    
    return ba


def calculate_precision_metrics(y_true, y_pred_score, threshold=0.5):
    tpr = calculate_tpr(y_true, y_pred_score, threshold)
    tnr = calculate_tnr(y_true, y_pred_score, threshold)
    fpr,fnr = calculate_fpr_fnr(y_true, y_pred_score, threshold)
    pd = calculate_positive_prediction(y_true, y_pred_score, threshold)

    precision = tpr / (tpr + fpr)
    recall = tpr / (tpr + fnr)

    # The true negative rate is the proportion of the individuals with a known negative condition for which the test
    # result is negative.

    return precision, recall, tpr, tnr, pd



def calculate_tpr(y_true, y_pred_score, threshold=0.5):
    """
    compute true positive rates (tpr)
    """
    y_pred = np.copy(y_pred_score)
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    cm = confusion_matrix(y_true, y_pred)
    tpr = round(cm[1, 1] / (cm[1, 1] + cm[1, 0]), 3)

    # The true positive rate is the proportion of the individuals with a known positive condition for which the test
    # result is positive. 

    return tpr


def calculate_tnr(y_true, y_pred_score, threshold=0.5):
    """
    compute true negative rate (tnr)
    """
    y_pred = np.copy(y_pred_score)
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    cm = confusion_matrix(y_true, y_pred)
    tnr = round(cm[0, 0] / (cm[0, 0] + cm[0, 1]), 3)

    # The true negative rate is the proportion of the individuals with a known negative condition for which the test
    # result is negative.

    return tnr


def calculate_positive_prediction(y_true, y_pred_score, threshold=0.5):
    """
    compute the proportion of positive predictions
    """
    y_pred = np.copy(y_pred_score)
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    cm = confusion_matrix(y_true, y_pred)
    pd = round((cm[1, 1] + cm[0, 1]) / (cm[1, 1] + cm[1, 0] + cm[0, 1] + cm[0, 0]), 3)

    return pd


def calculate_fpr_fnr(y_true, y_pred_score, threshold=0.5):
    """
    compute false positive rates (fpr) and false negative rates (fnr)
    """
    y_pred = np.copy(y_pred_score)
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # The false positive rate is the proportion of the individuals with a known negative condition for which the test
    # result is positive. This rate is sometimes called the fall-out.
    fpr = round(fp / (tn + fp), 3)
    # The false negative rate is the proportion of the individuals with a known positive condition for which the test
    # result is negative
    fnr = round(fn / (tp + fn), 3)

    return fpr, fnr


if __name__ == '__main__':
    y_true = np.array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0])
    y_pred_score = np.array([0.5, 0.1, 0.2, 0.9, 0.9, 0.5, 0, 0.1, 0.5, 0.0])
    best_threshold, current_best_f1 = get_optimal_threshold_f1(y_true, y_pred_score)
    print("best threshold", best_threshold)

    fpr, fnr = calculate_fpr_fnr(y_true, y_pred_score, threshold=best_threshold)
    tpr = calculate_tpr(y_true, y_pred_score, threshold=best_threshold)
    tnr = calculate_tnr(y_true, y_pred_score, threshold=best_threshold)
    print("true positive rate", tpr)
    print("true negative rate", tnr)
    print("false positive rate", fpr)
    print("false negative rate", fnr)
