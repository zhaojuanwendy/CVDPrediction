
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# from sklearn.preprocessing import Normalizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from os import path
import logging
import time
#
from definitions import LOGS_PATH

time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

log_files = path.join(LOGS_PATH, 'log_benchmark_cv.txt')
logging.basicConfig(filename=log_files+str(time_stamp), level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug('This is a log message.')


# def feature_selection(X_train, y_train, X_test):
#
#     lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
#     model = SelectFromModel(lsvc, prefit=True)
#     X_train_new = model.transform(X_train)
#     print(X_train_new.shape)
#     X_test_new = model.transform(X_test)
#     print(X_test_new.shape)
#     return X_train_new, X_test_new


def simple_benchmark(model_class, X_train, X_test,y_train, y_test):
    scores = []
    times = []
    model = model_class
    start = time.time()
    preds = model.fit(X_train, y_train).predict(X_test)
    end = time.time()
    scores.append(accuracy_score(y_test, preds))
    times.append(end - start)
    return scores, times

def benchmark_nested_cross_val(X, y, model, model_arams, model_name, standardscaler=None, n_split=5, cv_fold=5):
    """benchmarks a given model on a given dataset
    Instantiates the model with given parameters.
    :param model_class: class of the model to instantiate
    :param X: input feature data
    :param y: input label
    :param standardscaler: input standardization scaler or normalizer
    :param n_split: number of folds for split traning and testing,must be 2
    :param cv_fold: number of for tune hyper-parameters on training
    :param return_time: if true, returns list of running times in addition to scores
    :return: tuple array list (accuracy scores, prediction scores, recall scores, auc scores running times)
    """
    cvs_accs = []
    cvs_preds = []
    cvs_recalls = []
    cvs_aucs = []
    cvs_ap=[]
    #newly added for grid search
    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
    times = []
    random_state = 12883823
    # rkf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=random_state)
    rskf = RepeatedStratifiedKFold(n_splits=n_split, n_repeats=1, random_state=random_state)
    index = 0

    for train_index, test_index in rskf.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        index += 1
        print("iter", index)
        logging.debug("iter: %d" % index)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(X_train.shape[0])
        print(X_test.shape[0])
        logging.debug("train: %d" % X_train.shape[0])
        logging.debug("test: %d" % X_test.shape[0])
        logging.debug("feature_size: %d" % X_train.shape[1])

        start = time.time()
        if standardscaler is not None:
            standardscaler.fit(X_train)
            X_train = standardscaler.transform(X_train)
            X_test = standardscaler.transform(X_test)
            # print("standardized")
        # print(X_train.shape[0])

        if model_name.startswith('Logistic_reg') or model_name =='RandomForest':
        #grid search nested cross validation
            clf = GridSearchCV(model, model_arams, scoring=scoring, cv=cv_fold, refit='AUC',n_jobs=-1)
            grid_result = clf.fit(X_train, y_train)

            logging.debug("train model: %s" % model_name)
            print('train model', model_name)

            # for mean, stdev, param in zip(means, stds, params):
            #     logging.debug("%f (%f) AUC with: %r" % (mean, stdev, param))
            #     print("%f (%f) with: %r" % (mean, stdev, param))
            best_params_ = grid_result.best_params_
            logging.debug("best parameters: %s" % best_params_)
            print('best parameters', best_params_)

            preds = grid_result.predict(X_test);
            #preds = model.fit(X_train, y_train).predict(X_test)
            y_score = grid_result.predict_proba(X_test)[:, 1]

        else:
            preds = model.fit(X_train, y_train).predict(X_test)
            y_score = model.predict_proba(X_test)[:, 1]

        end = time.time()
        cvs_accs.append(accuracy_score(y_test, preds))
        cvs_preds.append(precision_score(y_test, preds))
        cvs_recalls.append(recall_score(y_test, preds))
        cvs_aucs.append(roc_auc_score(y_test, y_score))
        cvs_ap.append(average_precision_score(y_test, y_score))
        print(roc_auc_score(y_test, y_score))
        times.append(end - start)

    return cvs_accs, cvs_preds, cvs_recalls, cvs_aucs, cvs_ap, times

def benchmark_cross_val_k_fold(X, y, model, standardscaler=None, n_iters=5, cv_fold=10):
    """benchmarks a given model on a given dataset
    Instantiates the model with given parameters.
    :param model_class: class of the model to instantiate
    :param X: input feature data
    :param y: input label
    :param standardscaler: input standardization scaler or normalizer
    :param iters: how many times to benchmark
    :param return_time: if true, returns list of running times in addition to scores
    :return: tuple array list (accuracy scores, prediction scores, recall scores, auc scores running times)
    """
    cvs_accs = []
    cvs_preds = []
    cvs_recalls = []
    cvs_aucs = []
    cvs_ap = [] # average precision


    times = []
    random_state = 12883823
    # rkf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=random_state)
    rskf = RepeatedStratifiedKFold(n_splits=cv_fold, n_repeats=n_iters, random_state = random_state)
    index = 0

    for train_index, test_index in rskf.split(X,y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        print("iter", index)
        index += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(X_train.shape[0])
        print(X_test.shape[0])

        start = time.time()
        if standardscaler is not None:
            standardscaler.fit(X_train)
            X_train = standardscaler.transform(X_train)
            X_test = standardscaler.transform(X_test)
            # print("standardized")
        # print(X_train.shape[0])

        preds = model.fit(X_train, y_train).predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        end = time.time()
        cvs_accs.append(accuracy_score(y_test, preds))
        cvs_preds.append(precision_score(y_test, preds))
        cvs_recalls.append(recall_score(y_test, preds))
        cvs_aucs.append(roc_auc_score(y_test, y_score))
        cvs_ap.append(average_precision_score(y_test, y_score))
        print(roc_auc_score(y_test, y_score))
        times.append(end - start)

    return cvs_accs, cvs_preds, cvs_recalls, cvs_aucs,cvs_ap, times




