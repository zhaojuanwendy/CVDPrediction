import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from os import path
def prepare_data_from_csv(path):
    df = pd.read_csv(path)
    # assgin the Class to the class label
    y = df.Class
    # drop GRID, Dob_year, Class for the feature
    # fill the missing value
    result_new = df.drop(['GRID', 'Class'], axis=1)
    # result_new.fillna(0, inplace=True)

    X = result_new.values
    features = result_new.columns

    return X, y, features

def prepare_data(data_path):
    X = np.load(path.join(data_path, 'X.npy'))
    y = np.load(path.join(data_path, 'y.npy'))

    return X, y


def prepare_data_with_all_gene(path):
    df = pd.read_csv(path)
    num_of_pos = sum(df.Class)
    print(num_of_pos)
    num_of_neg = df.shape[0] - num_of_pos
    print(num_of_neg)
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.Class.values

    df.drop(['GRID','Class'], axis=1, inplace=True)
    X = df.values
    return X, y

def reshape_time_series(X, time_steps, feature_size):
    X_new = np.zeros(shape=(X.shape[0],X.shape[1]))
    idx=0
    for i in range(time_steps):
        for j in range(feature_size):
            X_new[:,idx] = X[:,i+j*time_steps]
            idx+=1
    return X_new

def prepare_features(X,time_feature_beg, timesteps, feature_size, if_use_snps=False, snps_features=204):
    if if_use_snps == True:
        X_time = X[:, time_feature_beg:(X.shape[1] - snps_features)]
        X_static = X[:, 0:time_feature_beg]
        X_snps = X[:, (X.shape[1] - snps_features):]
        X_aux = np.concatenate([X_static, X_snps], axis=1)
    else:
        X_time = X[:, time_feature_beg:]
        X_aux = X[:, 0:time_feature_beg]

    X_time_new = reshape_time_series(X_time,timesteps,feature_size)
    X_time = X_time_new.reshape((X_time_new.shape[0], timesteps, int(X_time_new.shape[1] / timesteps)))
    print(X_time.shape)
    print(X_aux.shape)
    return X_time, X_aux
