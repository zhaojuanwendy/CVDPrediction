import time
import numpy as np  # linear algebra
import pandas as pd  #
import logging
from os import path
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import ensemble
from sklearn import preprocessing


from benchmarks import benchmark_nested_cross_val
from prepare_data import prepare_data
import utility as util

from sys import argv

myargs = util.getopts(argv)

from definitions import LOGS_PATH
from definitions import RESULT_PATH
from definitions import DATA_PATH



time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

log_files = path.join(LOGS_PATH, 'log_benchmark_time.txt')
logging.basicConfig(filename=log_files+str(time_stamp), level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug('This is a log message.')


models = [
    # #alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
    #    eta0=0.0, fit_intercept=True, max_iter=1000, tol=None,l1_ratio=0.15,
    #    learning_rate='optimal', loss='hinge'
    (LogisticRegression(C=2), {'C': [1, 10, 100]}, 'Logistic_reg_scale'),
    (ensemble.RandomForestClassifier(n_estimators=100), {"max_depth": [3, None],
              "max_features": ['auto', 3, 10],
              "min_samples_split": [2, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}, 'RandomForest'),
    (ensemble.GradientBoostingClassifier(), {}, 'Gradient boosting')
    #(LateFusion(model2=svm.SVC(kernel='linear', C=1, probability=True),standardscaler=preprocessing.Normalizer()), {}, 'late_fusion(GB,SVM) fusion5:5')
    # (LateFusion(model2=LogisticRegression(C=2, class_weight={0:0.1, 1:0.9})),{},'late_fusion(GB,LR(0.1,0.9)'),
    # (LateFusion(model2=ensemble.GradientBoostingClassifier()),{},'late_fusion(GB, GB')
]

def run_nested_cv_fold(data_type='benchmark',n_split = 5, cv_fold = 5):
    """
    a nested cross validation method
    :param data_type: benchmark or temporal
    :param n_split: number of folds for split traning and testing,must be 2
    :param cv_fold: number of fold in the cross-validation on training test for determining the best hyper-parameters with grid-search
    :return:
    """

    data_path = path.join(DATA_PATH, data_type)

    if path.exists(data_path):
        X, y= prepare_data(data_path)
    else:
        return 0

    print(X.shape)
    logging.debug('total feature %d'%  X.shape[1])
    logging.debug('total control %d'% y[y == 0].shape[0])
    logging.debug('total cases %d'% y[y==1].shape[0])
    print('total control', y[y == 0].shape[0])
    print('total cases', y[y == 1].shape[0])
    print('control/case ratio ', y[y==0].shape[0]/y[y==1].shape[0])

    # for each model compute the performance, using k-fold cross validation with grid search in the training set
    records = []
    for model_class, params, model_name in models:
        # for test_size in [0.3, 0.2, 0.1]:
        standardscaler = None
        if model_name == 'Logistic_reg_scale':
            standardscaler = preprocessing.MaxAbsScaler()
        if model_name == 'SVM':
            standardscaler = preprocessing.Normalizer()

        cvs_accs, cvs_preds, cvs_recalls, cvs_aucs, cvs_ap, times = \
            benchmark_nested_cross_val(X, y, model_class, params, model_name, standardscaler, n_split=n_split,
                                       cv_fold=cv_fold)

        df_results_cv = pd.DataFrame({
            'a_iter': range(n_split),
            'acc': cvs_accs,
            'auc': cvs_aucs,
            'preds': cvs_preds,
            'recall': cvs_recalls,
            'zaverage_prec': cvs_ap
        })

        df_results_cv.to_csv(path.join(RESULT_PATH, 'detailed-{}.result'.format(data_type)), index=False)


        records.append({
            'a_data_type': data_type,
            'ab_model': model_name,
            'accuracy': '%.4f (+/- %.4f)' % (np.mean(cvs_accs), np.std(cvs_accs)),
            'auc': '%.4f (+/- %.4f)' % (np.mean(cvs_aucs), np.std(cvs_aucs)),
            'prec': '%.4f (+/- %.4f)' % (np.mean(cvs_preds), np.std(cvs_preds)),
            'recall': '%.4f (+/- %.4f)' % (np.mean(cvs_recalls), np.std(cvs_recalls)),
            's_average_prec': '%.4f (+/- %.4f)' % (np.mean(cvs_ap), np.std(cvs_ap)),
            'time': '%.4f (+/- %.4f)' % (np.mean(times), np.std(times)),
        })

        # result_csv = 'clf_results_emerge/' + data_file_keywds[index] + str(time_stamp)+'.csv'
    result_csv = 'clf-nested-cv-average-{}_fold-{}_times.csv'.format(cv_fold, n_split)
    pd.DataFrame(records).to_csv(path.join(RESULT_PATH, result_csv), index=False)
    logging.debug("iterations %d" % 10)



if __name__ == '__main__':

    data_type ='benchmark'

    if '-d' in myargs:  #
        print(myargs['-d'])
        data_type = myargs['-d']

    run_nested_cv_fold(data_type=data_type, n_split=2, cv_fold =2)
