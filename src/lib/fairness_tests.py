import pandas as pd
import numpy as np

import imblearn
from imblearn.over_sampling import RandomOverSampler, SMOTE

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer

from .optimal_threhold_related import get_optimal_threshold_Jvalue, calculate_tpr, calculate_positive_prediction


def get_EOD(y_test_1, y_score_1,threshold_1, y_test_2, y_score_2, threshold_2):
    """
    calculate equal opportunity difference (difference in true positive rate) across two groups
    """    
    tpr_1 = calculate_tpr(y_test_1, y_score_1, threshold=threshold_1)
    print ("True positive rate of class 1 is " , tpr_1)

    tpr_2 = calculate_tpr(y_test_2, y_score_2, threshold=threshold_2)
    print("True positive rate of class 2 is " , tpr_2)

    eod = tpr_1 - tpr_2
    return eod


def get_SP(y_test_1, y_score_1, threshold_1, y_test_2, y_score_2, threshold_2):
    """
    calculate equal opportunity difference (difference in true positive rate) across two groups
    """
    pd_1 = calculate_positive_prediction(y_test_1, y_score_1, threshold=threshold_1)
    print("Positive prediction rate of class 1 is " , pd_1)

    pd_2 = calculate_positive_prediction(y_test_2, y_score_2, threshold=threshold_2)
    print("Positive prediction rate of class 2 is " , pd_2)

    sp = pd_1/pd_2
    return sp


def split_by_trait(X, y, attribute, random_state):
    """get test set"""
    df_train_val, df_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    """get train and evaluation set"""
    df_train, df_val, y_train, y_val = train_test_split(df_train_val, y_train_val, test_size=0.25, random_state=random_state)
    
    """separate the test sets by the trait (attribute) we want"""
    y_test_1 = y_test[df_test[attribute]==1]
    y_test_2 = y_test[df_test[attribute]==0]
    
    df_test_1 = df_test[df_test[attribute]==1]
    X_test_1 = df_test_1.drop([attribute], axis=1).values
    df_test_2 = df_test[df_test[attribute]==0]
    X_test_2 = df_test_2.drop([attribute], axis=1).values
    
    """separate the validation sets by the trait (attribute) we want"""
    y_val_1 = y_val[df_val[attribute]==1]
    y_val_2 = y_val[df_val[attribute]==0]
       
    df_val_1 = df_val[df_val[attribute]==1]
    X_val_1 = df_val_1.drop([attribute], axis=1).values
    df_val_2 = df_val[df_val[attribute]==0]
    X_val_2 = df_val_2.drop([attribute], axis=1).values

    """The overall X set should be protected (exclude the attribute)"""
    X_train = df_train.drop([attribute], axis=1).values
    X_val = df_val.drop([attribute], axis=1).values
    X_test = df_test.drop([attribute], axis=1).values
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_val_1, X_val_2, y_val_1, y_val_2, X_test_1, X_test_2, y_test_1, y_test_2


def split_by_trait_no_protected_trait (X, y, attribute, random_state):
    """get test set"""
    df_train_val, df_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    """get train and evaluation set"""
    df_train, df_val, y_train, y_val = train_test_split(df_train_val, y_train_val, test_size=0.25, random_state=random_state)
    
    """separate the test sets by the trait (attribute) we want"""
    y_test_1 = y_test[df_test[attribute]==1]
    y_test_2 = y_test[df_test[attribute]==0]
    
    df_test_1 = df_test[df_test[attribute]==1]
    
    df_test_2 = df_test[df_test[attribute]==0]
    
    
    """separate the validation sets by the trait (attribute) we want"""
    y_val_1 = y_val[df_val[attribute]==1]
    y_val_2 = y_val[df_val[attribute]==0]
       
    df_val_1 = df_val[df_val[attribute]==1]
    
    df_val_2 = df_val[df_val[attribute]==0]
    

    """The overall X set should be protected (exclude the attribute)"""
    X_train = df_train.drop([attribute], axis=1).values
    X_val = df_val.drop([attribute], axis=1).values
    X_test = df_test.drop([attribute], axis=1).values
    
    return df_train, y_train, df_val, y_val, df_test, y_test, df_val_1, df_val_2, y_val_1, y_val_2, df_test_1, df_test_2, y_test_1, y_test_2


def split_by_trait_balance_size(X, y, attribute, random_state):
    """get test set"""
    df_train_val, df_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    """get train and evaluation set"""
    df_train, df_val, y_train, y_val = train_test_split(df_train_val, y_train_val, test_size=0.25, random_state=random_state)
    
    df_train ['Class'] = y_train
    y = df_train[attribute]
    X = df_train.drop([attribute], axis=1)
    X_over, y_over = SMOTE().fit_resample(X,y)
    y_1 = y_over[y_over == 1]
    y_0 = y_over[y_over == 0]
    print (y_1.shape)
    print (y_0.shape)    
    X_over [attribute] = y_over    
    resampled_y_train = X_over.Class.values
    resampled_df_train = X_over.drop(['Class'], axis=1)
    print(resampled_df_train.shape)
    
    """separate the test sets by the trait (attribute) we want"""
    y_test_1 = y_test[df_test[attribute]==1]
    y_test_2 = y_test[df_test[attribute]==0]
    
    df_test_1 = df_test[df_test[attribute]==1]
    X_test_1 = df_test_1.drop([attribute], axis=1).values
    df_test_2 = df_test[df_test[attribute]==0]
    X_test_2 = df_test_2.drop([attribute], axis=1).values
    
    """separate the validation sets by the trait (attribute) we want"""
    y_val_1 = y_val[df_val[attribute]==1]
    y_val_2 = y_val[df_val[attribute]==0]
       
    df_val_1 = df_val[df_val[attribute]==1]
    X_val_1 = df_val_1.drop([attribute], axis=1).values
    df_val_2 = df_val[df_val[attribute]==0]
    X_val_2 = df_val_2.drop([attribute], axis=1).values

    """The overall X set should be protected (exclude the attribute)"""
    X_train = resampled_df_train.drop([attribute], axis=1).values
    X_val = df_val.drop([attribute], axis=1).values
    X_test = df_test.drop([attribute], axis=1).values
    
    return X_train, resampled_y_train, X_val, y_val, X_test, y_test, X_val_1, X_val_2, y_val_1, y_val_2, X_test_1, X_test_2, y_test_1, y_test_2


def split_by_trait_balance_size_no_protected_trait(X, y, attribute, random_state):
    # mean race/gender is included in the model
    df_train_val, df_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    """get train and evaluation set"""
    df_train, df_val, y_train, y_val = train_test_split(df_train_val, y_train_val, test_size=0.25, random_state=random_state)
    
    df_train ['Class'] = y_train
    y = df_train[attribute]
    X = df_train.drop([attribute], axis=1)
    X_over, y_over = SMOTE().fit_resample(X,y)
    y_1 = y_over[y_over == 1]
    y_0 = y_over[y_over == 0]
    print (y_1.shape)
    print (y_0.shape)    
    X_over [attribute] = y_over    
    resampled_y_train = X_over.Class
    resampled_df_train = X_over.drop(['Class'], axis=1)
    print(resampled_df_train)
    print(resampled_y_train)
    
    
    """separate the test sets by the trait (attribute) we want"""
    y_test_1 = y_test[df_test[attribute]==1]
    y_test_2 = y_test[df_test[attribute]==0]
    
    df_test_1 = df_test[df_test[attribute]==1]
    X_test_1 = df_test_1.drop([attribute], axis=1).values
    df_test_1 = df_test_1.values
    print("df_test_1", df_test_1.shape)
    print("df_test_1", df_test_1)
    
    print("X_test_1", X_test_1.shape)
    print("X_test_1", X_test_1)
    
    df_test_2 = df_test[df_test[attribute]==0]
    X_test_2 = df_test_2.drop([attribute], axis=1).values
    df_test_2 = df_test_2.values
    print("df_test_2", df_test_2.shape)
    print("df_test_2", df_test_2)
   
    print("X_test_2", X_test_2.shape)
    print("X_test_2", X_test_2)
    
    """separate the validation sets by the trait (attribute) we want"""
    y_val_1 = y_val[df_val[attribute]==1]
    y_val_2 = y_val[df_val[attribute]==0]
       
    df_val_1 = df_val[df_val[attribute]==1].values
#     print("df_val_1", df_val_1.shape)
#     X_val_1 = df_val_1.drop([attribute], axis=1).values
#     print("X_val_1", X_val_1.shape)
    df_val_2 = df_val[df_val[attribute]==0].values
#     print("df_val_2", df_val_2.shape)
#     X_val_2 = df_val_2.drop([attribute], axis=1).values
#     print("X_val_2", X_val_2.shape)

#     print("resampled_df_train", resampled_df_train.shape)
#     print("df_val", df_val.shape)
#     print("df_test", df_test.shape)
    """The overall X set should be protected (exclude the attribute)"""
#     X_train = resampled_df_train.drop([attribute], axis=1).values
#     print("X_train", X_train.shape)
#     print("resampled_y_train", resampled_y_train.shape)
#     X_val = df_val.drop([attribute], axis=1).values
#     print("X_val", X_val.shape)
#     print("y_val", y_val.shape)
#     X_test = df_test.drop([attribute], axis=1).values
#     print("X_test", X_test.shape)
#     print("y_test", y_test.shape)

    resampled_df_train = resampled_df_train.values
    df_val = df_val.values
    df_test = df_test.values
    
    return resampled_df_train, resampled_y_train, df_val, y_val, df_test, y_test, df_val_1, df_val_2, y_val_1, y_val_2, df_test_1, df_test_2, y_test_1, y_test_2


def split_by_trait_balance_proportion(X, y, attribute, random_state):
    """get test set"""
    df_train_val, df_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    """get train and evaluation set"""
    df_train, df_val, y_train, y_val = train_test_split(df_train_val, y_train_val, test_size=0.25, random_state=random_state)
    
    df_train ['Class'] = y_train
    df_train_0 = df_train[df_train[attribute] == 0]
    """0 for male in Gender, or black in Race_W"""
    df_train_1 = df_train[df_train[attribute] == 1]
    """1 for female in Gender, or white in Race_W"""
    print (df_train_0.shape)
    print (df_train_1.shape)

    df_train_0_affect = df_train_0[df_train_0['Class'] == 1]
    df_train_0_unaffect = df_train_0[df_train_0['Class'] == 0]
    df_train_1_affect = df_train_1[df_train_1['Class'] == 1]
    df_train_1_unaffect = df_train_1[df_train_1['Class'] == 0]
    
    class0_affection_ratio = df_train_0_affect.shape[0]/df_train_0_unaffect.shape[0]
    class1_affection_ratio = df_train_1_affect.shape[0]/df_train_1_unaffect.shape[0]
    print(class0_affection_ratio, class1_affection_ratio)
    higher_affection = max(class0_affection_ratio, class1_affection_ratio)
    lower_affection = min(class0_affection_ratio, class1_affection_ratio)

    frames = []
    if (higher_affection == class0_affection_ratio):
        y = df_train_1.Class
        X = df_train_1.drop(['Class'], axis=1)
        X_over, y_over = SMOTE(sampling_strategy = higher_affection).fit_resample(X,y)
        y_affected = y_over[y_over == 1]
        y_unaffected = y_over[y_over == 0]
        print (y_affected.shape[0]/y_unaffected.shape[0])
        X_over ['Class'] = y_over
        frames = [df_train_0, X_over]
    else:
        y = df_train_0.Class
        X = df_train_0.drop(['Class'], axis=1)
        X_over, y_over = SMOTE(sampling_strategy=higher_affection).fit_resample(X, y)
        y_affected = y_over[y_over == 1]
        y_unaffected = y_over[y_over == 0]
        print (y_affected.shape[0] / y_unaffected.shape[0])
        X_over['Class'] = y_over
        frames = [df_train_1, X_over]
        
    result = pd.concat(frames)
    y_train = result.Class.values
    df_train = result.drop(['Class'], axis=1)
    print (df_train.shape)

    """separate the test sets by the trait (attribute) we want"""
    y_test_1 = y_test[df_test[attribute]==1]
    y_test_2 = y_test[df_test[attribute]==0]
    
    df_test_1 = df_test[df_test[attribute]==1]
    X_test_1 = df_test_1.drop([attribute], axis=1).values
    df_test_2 = df_test[df_test[attribute]==0]
    X_test_2 = df_test_2.drop([attribute], axis=1).values
    
    """separate the validation sets by the trait (attribute) we want"""
    y_val_1 = y_val[df_val[attribute]==1]
    y_val_2 = y_val[df_val[attribute]==0]
       
    df_val_1 = df_val[df_val[attribute]==1]
    X_val_1 = df_val_1.drop([attribute], axis=1).values
    df_val_2 = df_val[df_val[attribute]==0]
    X_val_2 = df_val_2.drop([attribute], axis=1).values

    """The overall X set should be protected (exclude the attribute)"""
    X_train = df_train.drop([attribute], axis=1).values
    X_val = df_val.drop([attribute], axis=1).values
    X_test = df_test.drop([attribute], axis=1).values
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_val_1, X_val_2, y_val_1, y_val_2, X_test_1, X_test_2, y_test_1, y_test_2


def split_by_trait_balance_proportion_no_protected_trait(X, y, attribute, random_state):
    """get test set"""
    df_train_val, df_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    """get train and evaluation set"""
    df_train, df_val, y_train, y_val = train_test_split(df_train_val, y_train_val, test_size=0.25, random_state=random_state)
    
    df_train ['Class'] = y_train
    df_train_0 = df_train[df_train[attribute] == 0]
    """0 for male in Gender, or black in Race_W"""
    df_train_1 = df_train[df_train[attribute] == 1]
    """1 for female in Gender, or white in Race_W"""
    print (df_train_0.shape)
    print (df_train_1.shape)

    df_train_0_affect = df_train_0[df_train_0['Class'] == 1]
    df_train_0_unaffect = df_train_0[df_train_0['Class'] == 0]
    df_train_1_affect = df_train_1[df_train_1['Class'] == 1]
    df_train_1_unaffect = df_train_1[df_train_1['Class'] == 0]
    
    class0_affection_ratio = df_train_0_affect.shape[0]/df_train_0_unaffect.shape[0]
    class1_affection_ratio = df_train_1_affect.shape[0]/df_train_1_unaffect.shape[0]
    print(class0_affection_ratio, class1_affection_ratio)
    higher_affection = max(class0_affection_ratio, class1_affection_ratio)
    lower_affection = min(class0_affection_ratio, class1_affection_ratio)

    frames = []
    if (higher_affection == class0_affection_ratio):
        y = df_train_1.Class
        X = df_train_1.drop(['Class'], axis=1)
        X_over, y_over = SMOTE(sampling_strategy = higher_affection).fit_resample(X,y)
        y_affected = y_over[y_over == 1]
        y_unaffected = y_over[y_over == 0]
        print (y_affected.shape[0]/y_unaffected.shape[0])
        X_over ['Class'] = y_over
        frames = [df_train_0, X_over]
    else:
        y = df_train_0.Class
        X = df_train_0.drop(['Class'], axis=1)
        X_over, y_over = SMOTE(sampling_strategy=higher_affection).fit_resample(X, y)
        y_affected = y_over[y_over == 1]
        y_unaffected = y_over[y_over == 0]
        print (y_affected.shape[0] / y_unaffected.shape[0])
        X_over['Class'] = y_over
        frames = [df_train_1, X_over]
        
    result = pd.concat(frames)
    y_train = result.Class.values
    df_train = result.drop(['Class'], axis=1)
    print (df_train.shape)

    """separate the test sets by the trait (attribute) we want"""
    y_test_1 = y_test[df_test[attribute]==1]
    y_test_2 = y_test[df_test[attribute]==0]
    
    df_test_1 = df_test[df_test[attribute]==1]
   
    df_test_2 = df_test[df_test[attribute]==0]
    
    
    """separate the validation sets by the trait (attribute) we want"""
    y_val_1 = y_val[df_val[attribute]==1]
    y_val_2 = y_val[df_val[attribute]==0]
       
    df_val_1 = df_val[df_val[attribute]==1]
    
    df_val_2 = df_val[df_val[attribute]==0]
    
    
    return df_train, y_train, df_val, y_val, df_test, y_test, df_val_1, df_val_2, y_val_1, y_val_2, df_test_1, df_test_2, y_test_1, y_test_2




    
    
    