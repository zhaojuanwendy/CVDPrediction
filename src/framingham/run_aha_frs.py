import logging
import time
from os import path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from definitions import DATA_PATH
from definitions import LOGS_PATH
import aha_frs_cvd

time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

log_files = path.join(LOGS_PATH, 'log_benchmark.txt')
logging.basicConfig(filename=log_files+str(time_stamp), level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug('This is a log message.')


#Please identify the framingham features
#framing_features= ['Gender','AGE','MEDIAN_VALUE_Chol','MEDIAN_VALUE_HDL-C','MEDIAN_SYSTOLIC','Smoking','T2DM_CNT','HTN_DRUG_CNT','Race']
framing_features = ['Gender', 'AGE', 'RECENT_VALUE_Chol', 'RECENT_VALUE_HDL-C', 'RECENT_SYSTOLIC', 'Smoking',
                    'T2DM_CNT', 'HTN_DRUG_CNT', 'Race']


def compute_ind_frs(df):
    score_list = []
    for index, row in df.iterrows():
        X = row[framing_features].values
        score = aha_frs_cvd.frs(*X)
        score_list.append(score)
    df['frs'] = pd.Series(score_list)
    return df


def predict_by_framingham(df: object, reset_index: object = False) -> object:
    # if the input df is cross validation split, need to reset index
    if reset_index:
        df.reset_index(drop=True)
    y = df.Class.values
    df = compute_ind_frs(df)
    print("after compute", df.shape)
    x = df[(df['Gender'] == 'F') & (df['Race'] == 'W')]['frs']

    mean_frs_women_w = np.mean(df[(df['Gender']=='F') & (df['Race']=='W')]['frs'])
    mean_frs_women_b = np.mean(df[(df['Gender']=='F') & (df['Race']=='B')]['frs'])
    mean_frs_men_w = np.mean(df[(df['Gender']=='M') & (df['Race']=='W')]['frs'])
    mean_frs_men_b = np.mean(df[(df['Gender']=='M') & (df['Race']=='B')]['frs'])
    # mean_frs_women_w = -29.18
    # mean_frs_women_b = 86.61
    # mean_frs_men_w = 61.18
    # mean_frs_men_b = 19.54
    risk_list = []
    for index, row in df.iterrows():
        gender = row['Gender']
        race = row['Race']
        risk = 0

        if gender == 'F':
            if race == 'W':
                risk = aha_frs_cvd.estimiate_risk(ind_frs=row['frs'], mean_frs=mean_frs_women_w, gender='F', race='W')
            elif race == 'B':
                risk = aha_frs_cvd.estimiate_risk(ind_frs=row['frs'], mean_frs=mean_frs_women_b, gender='F', race='B')
            else:
                print('1',race)
        elif gender == 'M':
            if race == 'W':
                risk = aha_frs_cvd.estimiate_risk(ind_frs=row['frs'], mean_frs=mean_frs_men_w, gender='M', race='W')
            elif race == 'B':
                risk = aha_frs_cvd.estimiate_risk(ind_frs=row['frs'], mean_frs=mean_frs_men_b, gender='M', race='B')
            else:
                print('2', race)
        else:
            print('3', gender)
        # if np.isnan(risk):
        #     print(index)

        risk_list.append(risk)
    df['risk'] = pd.Series(risk_list)
    print(df.risk.unique())
    print(len(risk_list))
    df.loc[df['risk'] > 0.075, 'predict'] = 1
    df.loc[df['risk'] <= 0.075, 'predict'] = 0
    print(df.predict.unique())
    #save the interim output
    df.to_csv(path.join(DATA_PATH, 'framingham_result.csv'))

    print('accuracy', accuracy_score(y, df['predict'].values))
    print('roc AUC', roc_auc_score(y, df['risk'].values))
    print('precision', precision_score(y, df['predict'].values))
    print('recall', recall_score(y, df['predict'].values))

    return accuracy_score(y, df['predict'].values), roc_auc_score(y, df['risk'].values), \
           precision_score(y, df['predict'].values), recall_score(y, df['predict'].values), \
           average_precision_score(y, df['risk'].values)


def run():
    df = pd.read_csv(path.join(DATA_PATH, 'framingham_data.csv'))
    acc, roc, precision, recall, ap = predict_by_framingham(df)
    print("accuracy", acc)
    print("roc", roc)
    print("precision", precision)
    print("recall", recall)
    print("ap", ap)


if __name__ == '__main__':
    run()
