import pandas as pd
import seaborn as sns
import numpy as np
from os import path
import os
import sys
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, average_precision_score
import sklearn.preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn import ensemble
from sys import argv
import utility as util
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
print(module_path)
import src.lib.utility_classfier as uclf
from definitions import LOGS_PATH
from definitions import RESULT_PATH
from definitions import DATA_PATH

import logging
import time
time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

log_files = path.join(LOGS_PATH, 'log_tune_late_fusion_pretrain.txt')
logging.basicConfig(filename=log_files+str(time_stamp), level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug('This is a log message.')
root_data_path = path.join(DATA_PATH,'processed')
results_path = RESULT_PATH

big_category='out_patient_10.0'
cohort_category= 'large_cohort'

records =[]
time_steps = 7
feature_size = 74
time_features_start = 13
use_snps_features = False
snps_auto_encoder = False
snps_size = 204
usesnps = False
myargs = util.getopts(argv)
phemed_value_is_binary = False

if '-o' in myargs:  # Example usage.
    print(myargs['-o'])
    big_category = myargs['-o']
if '-c' in myargs:  # the cohort category, large_cohort or emerge_intersect or emerge_intersect_more
    print(myargs['-c'])
    cohort_category = myargs['-c']
if '-s' in myargs:  # if use snps
    print('use snps', myargs['-s']) #if use snps features
    if myargs['-s'] == 'y':
        use_snps_features = True
if '-snpsz' in myargs:  # if use snps size
    print(myargs['-snpsz']) #if use snps features
    snps_size = eval(myargs['-snpsz'])

if '-encode' in myargs:  # if use snps size
    print(myargs['-encode']) #if use snps features
    if myargs['-encode'] == 'y':
        snps_auto_encoder = True

if '-phemed_binary' in myargs:  # if use snps size
    print(myargs['-phemed_binary'])  # if use snps features
    if myargs['-phemed_binary'] == 'y':
        phemed_value_is_binary = True

# load dataset

# data_types = ['imputed_latest_benchmark','imputed_latest_time']
data_types =['imputed_latest_time']
cvd_history_types= ['no_cvd']

cohort_category1 = cohort_category
cohort_category2 = 'large_cohort'


def train_demo_lab(df_lab):
    y=df_lab.Class.values
    df_lab.drop(['GRID','Class'], axis=1, inplace=True)
    X = df_lab.values
    clf = ensemble.GradientBoostingClassifier()
    clf.fit(X, y)
    return clf


from sklearn.linear_model import LogisticRegression


def train_genetic_pretrain(df_gene):
    y = df_gene.Class.values
    df_gene.drop(['GRID', 'Class', 'CVD_event_year', 'CVD_first_age'], axis=1, inplace=True)
    X = df_gene.values
    X = preprocessing.scale(X)
    # clf = LogisticRegression(C=10, class_weight={0: 0.4, 1: 0.6})
    clf = LogisticRegression(C=10)
    clf.fit(X, y)
    uclf.rank_features(clf, df_gene.columns.tolist(), top_feature=20, abs=True)
    return clf

# def train_genetic(X, y,feature_list):
#     X = preprocessing.scale(X)
#     clf = LogisticRegression(C=10, class_weight={0: 0.4, 1: 0.6})
#     clf.fit(X, y)
#     uclf.rank_features(clf, feature_list)
#     return clf

def train_late_fusion(X,y):
    # fusion_clf = LogisticRegression(C=10,class_weight={0:0.4, 1:0.6})
    fusion_clf = LogisticRegression(C=10)
    fusion_clf.fit(X,y)
    return fusion_clf

import os
import errno
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

for data_type in data_types:
    ##loading the big genotyped cohort for pretrain
    df = pd.read_csv(path.join(root_data_path, 'large_emerge_for_pretrain.csv'))
    print('loading big genotyped cohort', df.shape)

    ##loading the emerge_intersect cohort
    data_folder = path.join(path.join(big_category, cohort_category1), path.join(data_type, iscvd))
    data_path = path.join(root_data_path, data_folder)
    if phemed_value_is_binary:
        data_path = path.join(data_path, 'phemed_binary_value')
        print('loading phemed binary', data_path)

    df_emerge_ins = pd.read_csv(path.join(data_path, 'cohort_preprocseed.csv'))
    print(df_emerge_ins.shape)
    print(len(df_emerge_ins.GRID.unique().tolist()))
    cvs_aucs1 =[]
    cvs_ap1=[]
    cvs_aucs2 = []
    cvs_ap2 = []
    cvs_aucs_final = []
    cvs_ap_final =[]
    from sklearn.model_selection import RepeatedStratifiedKFold

    n_splits = 5
    n_repeats = 10
    kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    # for i in range(10):
    #     np.random.seed(np.random.randint(100000*(i+1)))
    #     msk = np.random.rand(len(df_emerge_ins)) < 0.7
    #     train = df_emerge_ins[msk]
    #     test = df_emerge_ins[~msk]
    y_label = df_emerge_ins['Class'].values
    iter = 0
    for train_index, test_index in kf.split(df_emerge_ins, y_label):
        print('starting iter:', iter)
        train = df_emerge_ins.iloc[train_index,:]
        test = df_emerge_ins.iloc[test_index,:]

        print('train shape', train.shape)
        print('test shape', test.shape)

        data_folder = path.join(DATA_PATH, data_type)

        df_large = pd.read_csv(path.join(data_path, 'data.csv'))
        print('loading the large EHR cohort', df_large.shape)
        print(len(df_large.GRID.unique().tolist()))

        # remove the test cohort from the large lab cohort
        df_large_selected = df_large[~df_large.GRID.isin(test.GRID)]
        print('df_large_selected', df_large_selected.shape)

        # remove the test cohort from large genetic cohort
        df_genetic_selected = df[~df.GRID.isin(test.GRID)]
        print('df_genetic_selected', df_genetic_selected.shape)

        #pretrain the model for demo+lab and genetic respectively
        clf_model1 = train_demo_lab(df_large_selected)
        clf_model2 = train_genetic_pretrain(df_genetic_selected)

        y_valid = train.Class
        train.drop(['GRID', 'Class'], axis=1, inplace=True)
        # print(train.head())

        df_valid_lab = train.iloc[:, :train.shape[1] - snps_size]

        X_valid_demo_lab = df_valid_lab.values
        print(X_valid_demo_lab.shape)
        y_valid_1 = clf_model1.predict_proba(X_valid_demo_lab)[:, 1]
        cols = train.columns.tolist()
        demo_gene_cols = cols[:5] + cols[-snps_size:]

        df_valid_gene = train[demo_gene_cols]
        print("valid gene", df_valid_gene.shape)
        # print(df_valid_gene.head())
        X_valid_gene = df_valid_gene.values
        print(X_valid_gene.shape)

        # clf_model2 = train_genetic(X_valid_gene,y_valid, df_valid_gene.columns.tolist())

        y_valid_2 = clf_model2.predict_proba(preprocessing.scale(X_valid_gene))[:, 1]
        X_fusion_scores = np.stack((y_valid_1, y_valid_2), axis=-1)

        model3 = train_late_fusion(X_fusion_scores, y_valid)

        #test the fusion
        y_test = test.Class

        test.drop(['GRID', 'Class'], axis=1, inplace=True)
        y_scores_bl = []  # predict score list
        df_test_lab = test.iloc[:, :test.shape[1] - snps_size]
        print(test.shape[1] - snps_size)
        print(df_test_lab.shape[1])

        X_test_demo_lab = df_test_lab.values
        print('test lab', X_test_demo_lab.shape)

        y_predict_1 = clf_model1.predict_proba(X_test_demo_lab)[:, 1]
        y_scores_bl.append(y_predict_1)

        cvs_aucs1.append(roc_auc_score(y_test, y_predict_1))
        cvs_ap1.append(average_precision_score(y_test, y_predict_1))

        cols = test.columns.tolist()
        selected_cols = cols[:5] + cols[-snps_size:]

        df_test_gene = test[selected_cols]
        print('test gene', df_test_gene.shape)
        X_test_gene = df_test_gene.values

        y_predict_2 = clf_model2.predict_proba(preprocessing.scale(X_test_gene))[:, 1]
        y_scores_bl.append(y_predict_2)

        cvs_aucs2.append(roc_auc_score(y_test, y_predict_2))
        cvs_ap2.append(average_precision_score(y_test, y_predict_2))

        X_fusion_scores_test = np.stack((y_predict_1, y_predict_2), axis=-1)
        print('fusion input shape', X_fusion_scores_test.shape)

        y_final_score = model3.predict_proba(X_fusion_scores_test)[:, 1]  # fusion score
        y_scores_bl.append(y_final_score)

        cvs_aucs_final.append(roc_auc_score(y_test, y_final_score))
        cvs_ap_final.append(average_precision_score(y_test, y_final_score))

        #save the predict result


        out_path = path.join(RESULT_PATH, 'late_fusion')
        cv_out_path = path.join(out_path, 'iteration_'+ str(iter))
        mkdir_p(cv_out_path)
        np.savetxt(path.join(cv_out_path,'y_true.out'), y_test, fmt='%.1e')

        np.savetxt(path.join(cv_out_path, 'y_predict1.out'), y_predict_1, fmt='%.5e')
        np.savetxt(path.join(cv_out_path, 'y_predict2.out'), y_predict_2, fmt='%.5e')
        np.savetxt(path.join(cv_out_path, 'y_predict_final.out'), y_final_score,fmt='%.5e')
        np.savetxt(path.join(cv_out_path, 'test_index.txt'), test_index, fmt='%d') #save the index of the test data (index of the df_emerge_ins file)
        iter += 1
        # save the result for each iteration
    df_results_cv = pd.DataFrame({
        'a_iter': range(n_splits*n_repeats),
        'auc_1': cvs_aucs1,
        'auc_2': cvs_aucs2,
        'auc_fusion': cvs_aucs_final,
        'average_prec_1': cvs_ap1,
        'average_prec_2': cvs_ap2,
        'average_prec_fusion': cvs_ap_final
    })
    detailed_result_csv = 'late-fusion-pretrain-cv-detailed-' + str(n_splits) + '-fold-' + str(n_repeats) + '-iter-' + big_category + '-' + cohort_category + \
             '-usesnps-' + str(use_snps_features) + '-'+ data_type  + str(time_stamp) +'.csv'
    df_results_cv.to_csv(path.join(results_path, detailed_result_csv), index=False)

    # end for cv
    records.append({
        'a_data_type': data_type,
        'a_phemed_value_is_binary': str(phemed_value_is_binary),
        'auc1': '%.4f (+/- %.4f)' % (np.mean(cvs_aucs1), np.std(cvs_aucs1)),
        'average_precision1': '%.4f (+/- %.4f)' % (np.mean(cvs_ap1), np.std(cvs_ap1)),
        'auc2': '%.4f (+/- %.4f)' % (np.mean(cvs_aucs2), np.std(cvs_aucs2)),
        'average_precision2': '%.4f (+/- %.4f)' % (np.mean(cvs_ap2), np.std(cvs_ap2)),
        'auc_final': '%.4f (+/- %.4f)' % (np.mean(cvs_aucs_final), np.std(cvs_aucs_final)),
        'average_precision_final': '%.4f (+/- %.4f)' % (np.mean(cvs_ap_final), np.std(cvs_ap_final))
    })
    #save the average result
    result_csv = 'late-fusion-pretrain-cv-average-'+str(n_splits)+'-fold-'+str(n_repeats)+'-iter-' + big_category + '-' + cohort_category + \
                 '-usesnps-' + str(use_snps_features) + '-' + str(time_stamp) + '.csv'
    pd.DataFrame(records).to_csv(path.join(results_path, result_csv), index=False)