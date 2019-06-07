import logging
import time
from os import path

import numpy as np  # linear algebra
import pandas as pd  #
import utility as util
from prepare_data import prepare_data
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

import lib.utility_classfier as uclf

# from definitions import LOGS_PATH
# from definitions import RESULT_PATH
# from definitions import DATA_PATH

LOGS_PATH = '/legodata/zhaoj/cvd_risk_time/logs'
RESULT_PATH = '/legodata/zhaoj/cvd_risk_time/results'
DATA_PATH = '/legodata/zhaoj/cvd_risk_time/data'

root_data_path = path.join(DATA_PATH,'processed')

time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

feature_list =[]

def split_data_with_time_win(X, time_feature_beg,timesteps,feature_size):
    """

    :param X:
    :param time_feature_beg:
    :param timesteps:
    :param feature_size: distinct of time-series features
    :return:
    """
    X_time = X[:, time_feature_beg:(X.shape[1])]
    X_static = X[:, 0:time_feature_beg]
    multi_task_X = []
    for i in range(timesteps):
        X_time_t = np.zeros(shape=(X.shape[0], feature_size))
        for j in range(feature_size):
            X_time_t[:, j] = X_time[:, i + j * timesteps]  #all variables at time window t=i
        X_mix_t = np.concatenate([X_static, X_time_t], axis=1) #concat the time variable with static variables as dataset1
        print("X mix_t", X_mix_t.shape)
        multi_task_X.append(X_mix_t)
    return multi_task_X

def get_features_with_time_win(feature_list,time_feature_beg,timesteps, feature_size):
    f_time = feature_list[time_feature_beg:(len(feature_list))]
    f_static = feature_list[0:time_feature_beg]
    multi_task_featues = []
    for i in range(timesteps):
        f_time_t = [] # features for each time step
        for j in range(feature_size):
            f_time_t.append(f_time[i + j * timesteps])  # all variables at time window t=i
        f_mix_t = f_static + f_time_t # concat the time variable with static variables as dataset1
        multi_task_featues.append(f_mix_t)
    return multi_task_featues


def train_multiLR(multi_task_X_train, y_train, n_samples,multi_task_features,timesteps):
    clfs = []
    y_train_scores = np.zeros(shape=(n_samples, timesteps))

    for idx, X_train in enumerate(multi_task_X_train):
        print("200%d year" % idx)
        print('train shape', X_train.shape)
        clf = ensemble.GradientBoostingClassifier()
        clf.fit(X_train, y_train)
        predict_score = clf.predict_proba(X_train)[:, 1]
        y_train_scores[:, idx] = predict_score
        clfs.append(clf)
        uclf.rank_features(clf, multi_task_features[idx], 20)
        idx += 1
    return clfs, y_train_scores


def predict_with_multiLR(clfs, multi_task_X_test, n_samples):
    y_scores = np.zeros(shape=(n_samples, len(clfs)))
    idx = 0
    for clf, X_test in zip(clfs, multi_task_X_test):
        y_scores[:, idx] = clf.predict_proba(X_test)[:, 1]
        idx += 1
    return y_scores

def cal_avg(y_scores):
    return np.mean(y_scores, axis=1)


def multitaks(X_train, y_train, X_test, y_test, feature_list):
    print(X_train.shape)
    scaler = preprocessing.MaxAbsScaler().fit(X_train)
    # Scale the train set
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    multi_task_X_train = split_data_with_time_win(X_train, time_feature_beg=13, timesteps=7, feature_size=74)
    multi_task_X_test = split_data_with_time_win(X_test, time_feature_beg=13, timesteps=7, feature_size=74)

    multi_task_features = get_features_with_time_win(feature_list,time_feature_beg=13,timesteps=7, feature_size=74)

    clfs, y_train_scores = train_multiLR(multi_task_X_train, y_train, X_train.shape[0], multi_task_features,timesteps=7)
    lr = LogisticRegression(C=10)
    lr.fit(y_train_scores, y_train)

    y_scores = predict_with_multiLR(clfs, multi_task_X_test, X_test.shape[0])
    y_score = lr.predict_proba(y_scores)[:, 1]

    if hasattr(lr, 'coef_'):
        print(lr.coef_.shape)
        indices = np.argsort(lr.coef_[0])[::-1]  # ordered from largest to small
        feature_list = [str(i) for i in range(2000, 2007)]
        for f in range(y_scores.shape[1]):
            print("%d. feature % s (%f)" % (f + 1, feature_list[indices[f]], lr.coef_[0][indices[f]]))


    auc = roc_auc_score(y_test, y_score)
    ap = average_precision_score(y_test, y_score)
    print(y_score.shape)
    print('AUC', auc)
    print('ap', ap)
    return auc, ap

def run_multimask( data_type = 'benchmark'):
    # records = []
    cv_fold = 10
    records = []


    data_path = path.join(DATA_PATH, data_type)
    if phemed_value_is_binary:
        data_path = path.join(data_path, 'phemed_binary_value')
        print('loading phemed binary')
    # logging.info(data_path)
    df = pd.read_csv(path.join(data_path, 'cohort_preprocseed.csv'))
    df.drop(['GRID', 'Class'], axis=1, inplace=True)
    print(df.shape)
    feature_list = df.columns.tolist()
    X, y = prepare_data(data_path)

    random_state = 12883823
    rskf = RepeatedStratifiedKFold(n_splits=cv_fold, n_repeats=1, random_state=random_state)

    index = 0

    cvs_aucs = []
    cvs_aps = []

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        index += 1
        print("iter", index)
        print(X_train.shape[0])
        print(X_test.shape[0])
        logging.debug("train: %d" % X_train.shape[0])
        logging.debug("test: %d" % X_test.shape[0])
        logging.debug("feature_size: %d" % X_train.shape[1])
        auc, ap = multitaks(X_train, y_train, X_test, y_test, feature_list)
        cvs_aucs.append(auc)
        cvs_aps.append(ap)

        records.append({
        'a_data_type': data_type,
        'a_phemed_value_is_binary': str(phemed_value_is_binary),
        'ab_model': 'multitask',
        'auc': '%.4f (+/- %.4f)' % (np.mean(cvs_aucs), np.std(cvs_aucs)),
        's_average_prec': '%.4f (+/- %.4f)' % (np.mean(cvs_aps), np.std(cvs_aps))
        })
    result_csv = 'multitask-nested-cv-{}-fold-WB-25y-bp.csv'.format(cv_fold)
    pd.DataFrame(records).to_csv(path.join(RESULT_PATH, result_csv), index=False)


if __name__ == '__main__':
    from sys import argv

    snps_size = 204
    myargs = util.getopts(argv)

    if '-d' in myargs:  # the cohort category, large_cohort or emerge_intersect or emerge_intersect_more
        print(myargs['-d'])
        data_type = myargs['-d']

    run_multimask(data_type)