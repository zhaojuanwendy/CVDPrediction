from os import path

import matplotlib.pyplot as plt
import numpy as np
from prepare_data import prepare_data
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MaxAbsScaler

import lib.utility_classfier as uclf
from definitions import DATA_PATH
from definitions import RESULT_PATH

#
# def set_style():
#     # This sets reasonable defaults for font size for
#     # a figure that will go in a paper
#     sns.set_context("paper")
#
#     # Set the font to be serif, rather than sans
#     sns.set(font='serif')
#
#     # Make the background white, and specify the
#     # specific font family
#     sns.set_style("white", {
#         "font.family": "serif",
#         "font.serif": ["Times", "Palatino", "serif"]
#     })

cohort = 'out_patient_5.0'
cohort_catergory = 'emerge_intersect_more'

root_data_path = DATA_PATH + '/processed'
root_data_path = path.join(path.join(root_data_path, cohort), cohort_catergory)

benchmark_folder = 'imputed_latest_benchmark'
time_folder = 'imputed_latest_time'
has_cvd = 'has_cvd'
tableau_color = ['navy', 'darkorange', 'turquoise', 'cornflowerblue','teal']

# class_name= ['framingham','LR','RF','GB','LR+time','RF+time','GB+time','MLP+time','LSTM+time']

class_name = ['LR', 'RF', 'GBF']
# class_name=['LR']

def classify(X_train, y_train, X_test, y_test, c):
    y_score = np.array([0])

    if (c == 'LR'):
        max_abs_scaler = MaxAbsScaler()
        X_train_scaled = max_abs_scaler.fit_transform(X_train)
        X_test_scaled = max_abs_scaler.fit_transform(X_test)
        y_score = uclf.logic_regression(X_train_scaled,y_train, X_test_scaled, y_test)

    elif (c == 'RF'):
        y_score = uclf.random_forest(X_train, y_train, X_test, y_test)

    elif (c == 'GBF'):
        y_score = uclf.gradiant_boosting(X_train, y_train, X_test, y_test)

    # elif (c=='dnn'):
    #

    return y_score

rskf = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)

benchmark_data_path = path.join(path.join(root_data_path, benchmark_folder), has_cvd)
X_b, y_b = prepare_data(benchmark_data_path)

time_data_path = path.join(path.join(root_data_path, time_folder), has_cvd)
X_t, y_t = prepare_data(time_data_path)

for train_index, test_index in rskf.split(X_b, y_b):
    print("iter")
    X_b_train, X_b_test = X_b[train_index], X_b[test_index]
    y_b_train, y_b_test = y_b[train_index], y_b[test_index]

    print('benchmark', X_b_train.shape)
    print('benchmark', X_b_test.shape)

    X_t_train, X_t_test = X_t[train_index], X_t[test_index]
    y_t_train, y_t_test = y_t[train_index], y_t[test_index]

    print('time series', X_t_train.shape)
    print('time series', X_t_test.shape)

    y_b_scores = []
    y_t_scores = []
    y_b_tests = []
    y_t_tests = []

    "classify"
    for c in class_name:
        y_b_score = classify(X_b_train, y_b_train, X_b_test, y_b_test, c)
        y_b_scores.append(y_b_score)
        y_b_tests.append(y_b_test)

    for c in class_name:
        y_t_score = classify(X_t_train, y_t_train, X_t_test, y_t_test, c)
        y_t_scores.append(y_t_score)
        y_t_tests.append(y_t_test)

    "plot the result"
    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(8, 6), dpi=700)
    idx = 0
    for y_b_score, y_b_test in zip(y_b_scores, y_b_tests):
        fpr, tpr, _ = roc_curve(y_b_test.ravel(), y_b_score.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=(class_name[idx] + '(AUC = %0.4f)' % roc_auc), linestyle='dashed',linewidth=2,
                 color=tableau_color[idx])
        idx += 1

    idx = 0
    for y_t_score, y_t_test in zip(y_t_scores, y_t_tests):

        fpr, tpr, _ = roc_curve(y_t_test.ravel(), y_t_score.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=(class_name[idx] + '-time' + '(AUC = %0.4f)' % roc_auc),linewidth=2,
                 color=tableau_color[idx])
        idx += 1

    plt.title(has_cvd, fontsize=14)
    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.7, 0.4), fancybox=True, ncol=1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    fig.savefig(path.join(path.join(RESULT_PATH, 'figures'),
                          cohort + '-' + cohort_catergory + '-' + has_cvd + '-comparison_of_features_auc.png'))


    fig = plt.figure(figsize=(8, 6), dpi=700)
    idx = 0
    for y_b_score, y_b_test in zip(y_b_scores, y_b_tests):
        precision, recall, _ = precision_recall_curve(y_b_test.ravel(), y_b_score.ravel())
        average_precision = average_precision_score(y_b_test, y_b_score)
        plt.plot(recall, precision, label=(class_name[idx] + '(AP = %0.4f)' % average_precision),
                 color=tableau_color[idx], linewidth=2, linestyle='-.')
        idx += 1

    idx = 0
    for y_t_score, y_t_test in zip(y_t_scores, y_t_tests):
        precision, recall, _ = precision_recall_curve(y_t_test.ravel(), y_t_score.ravel())
        average_precision = average_precision_score(y_t_test, y_t_score)
        plt.plot(recall, precision, label=(class_name[idx] + '-time' + '(AP = %0.4f)' % average_precision),
                 color=tableau_color[idx],linewidth=2)
        idx += 1


    ratio = y_t_test[y_t_test==1].shape[0]/y_t_test.shape[0]
    plt.axhline(y=ratio, color='navy', linestyle='--')
    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.7, 0.4), fancybox=True, ncol=1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve-'+ has_cvd)
    

    break;

# plot bench mark data
