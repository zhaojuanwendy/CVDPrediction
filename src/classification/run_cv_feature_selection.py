import numpy as np
import pandas as pd
import csv
from sklearn.metrics import classification_report
from sklearn import metrics, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from time import time
from os import path
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.model_selection import RepeatedStratifiedKFold
from definitions import RESULT_PATH
from definitions import DATA_PATH


def rank_features(clf, feature_list, top_feature=10, abs=False):
    """
    :param clf:
    :param feature_list:
    :param top_feature:
    :param abs: if rank with the absolute value of the coefficient
    :return:
    """
    selected_features_list = []
    if hasattr(clf, 'coef_'):
        print(clf.coef_.shape)
        if abs is True:
            abs_coef = np.abs(clf.coef_[0])
            indices = np.argsort(abs_coef)[::-1]  # ordered from largest to small
        else:
            indices = np.argsort(clf.coef_[0])[::-1]  # ordered from largest to small

        for f in range(top_feature):
            selected_features_list.append(
                feature_list[indices[f]]
            )
            print("%d. feature %s (%f)" % (f + 1, feature_list[indices[f]], clf.coef_[0][indices[f]]))

    elif hasattr(clf, 'feature_importances_'):
        indices = np.argsort(clf.feature_importances_)[::-1]  # ordered from largest to small
        for f in range(top_feature):
            selected_features_list.append(
                feature_list[indices[f]]
            )
            print("%d. feature %s (%f)" % (f + 1, feature_list[indices[f]], clf.feature_importances_[indices[f]]))

    return selected_features_list

# logic regression
def logic_regression(X_train, y_train, X_valid, y_valid, c=10, feature_list=None, top_features_num=20):
    t0 = time()
    clf = LogisticRegression(C=c)
    clf.fit(X_train, y_train)
    p_train = clf.predict_proba(X_train)
    p_valid = clf.predict_proba(X_valid)
    y_score = clf.predict_proba(X_valid)[:, 1]

    y_predict_valid = clf.predict(X_valid)
    print(metrics.log_loss(y_train, p_train))
    print(metrics.log_loss(y_valid, p_valid))
    print("Classification report")
    print(classification_report(y_valid, y_predict_valid))
    print("Confusion_matrix")
    print(confusion_matrix(y_valid, y_predict_valid))
    print("done in %fs" % (time() - t0))
    if feature_list is not None:
        selected_features = rank_features(clf, feature_list, top_features_num)
    return y_score, selected_features


# random forest tree
def random_forest(X_train, y_train, X_valid, y_valid,feature_list=None,top_features_num=20,
                  bootstrap=True,criterion='entropy', max_features=10, min_samples_split=10):
    t0 = time()
    clf = ensemble.RandomForestClassifier(n_estimators=100, bootstrap=True, criterion='entropy', max_features=10, min_samples_split=10)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    # np.savetxt("random.csv", y_pred.astype(int), fmt='%i', delimiter=",")
    print("Classification report")
    print(classification_report(y_valid, y_pred))
    print("Confusion_matrix")
    print(confusion_matrix(y_valid, y_pred))
    print("done in %fs" % (time() - t0))
    y_score = clf.predict_proba(X_valid)[:, 1]
    if feature_list is not None:
        selected_features = rank_features(clf, feature_list, top_features_num)
    return y_score,selected_features


def gradiant_boosting(X_train, y_train, X_valid, y_valid,feature_list=None,top_features_num=20):
    t0 = time()
    clf = gbc()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    # np.savetxt("random.csv", y_pred.astype(int), fmt='%i', delimiter=",")
    print("Classification report")
    print(classification_report(y_valid, y_pred))
    print("Confusion_matrix")
    print(confusion_matrix(y_valid, y_pred))
    print("done in %fs" % (time() - t0))
    y_score = clf.predict_proba(X_valid)[:, 1]
    if feature_list is not None:
        selected_features = rank_features(clf, feature_list, top_features_num)
    return y_score, selected_features


def run_cv():

    data_types =['benchmark', 'temporal']
    model_names =['LR','RF','GBT']

    for data_type in data_types:
        for model_name in model_names:
            df = pd.read_csv(path.join(DATA_PATH, data_type, 'data.csv'))
            y = df.Class.values
            df.drop(['GRID', 'Class'], axis=1, inplace=True)
            feature_list = df.iloc[:, :(df.shape[1])].columns.tolist()
            X = df.iloc[:, :(df.shape[1])].values

            rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state = 43)
            index = 1
            features_per_folds = []
            for train_index, test_index in rskf.split(X,y):
                # print("TRAIN:", train_index, "TEST:", test_index)
                print("iter", index)
                index += 1
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                print(X_train.shape[0])
                print(X_test.shape[0])

                if model_name == 'LR':
                    standardscaler = preprocessing.MaxAbsScaler()
                    X_train_scaled = standardscaler.fit_transform(X_train)
                    X_test_scaled = standardscaler.transform(X_test)
                    y_score, selected_features_lr = logic_regression(X_train_scaled, y_train, X_test_scaled, y_test, 10, feature_list, 10)
                elif model_name == 'RF':
                    y_score, selected_features_lr = random_forest(X_train, y_train, X_test, y_test,feature_list,
                                                                     10)
                else:
                    y_score, selected_features_lr = gradiant_boosting(X_train, y_train, X_test, y_test, feature_list,
                                                                     10)

                print("############iter end #############")
                features_per_folds.append(selected_features_lr)


            with open(path.join(RESULT_PATH, "overlapped_features_{}_{}.csv".format(data_type,model_name)),"w+", newline='') as my_csv:
                fieldnames = ['iter_s','iter_t','overlapped_features']
                csvWriter = csv.DictWriter(my_csv,fieldnames=fieldnames)
                csvWriter.writeheader()
                for i in range(0, index-2):
                    for j in range(i+1, index-1):
                        s = features_per_folds[i]
                        t = features_per_folds[j]
                        num_overlapped_features = len(list(set(s) & set(t)))
                        csvWriter.writerow({'iter_s': i, 'iter_t': j, 'overlapped_features': num_overlapped_features})



            features_per_folds = zip(*features_per_folds)
            with open(path.join(RESULT_PATH, "top_features_{}_{}.csv".format(data_type, model_name)),"w+") as my_csv:
                csvWriter = csv.writer(my_csv, delimiter=',')
                csvWriter.writerow(['iter{}'.format(i) for i in range(10)])
                csvWriter.writerows(features_per_folds)

run_cv()