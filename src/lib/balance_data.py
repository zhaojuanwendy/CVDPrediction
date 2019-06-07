import numpy as np
from sklearn.utils import shuffle

def under_sampling(X,y,sample_size):
    idx = np.random.choice(np.arange(len(y)), sample_size, replace=False)
    X_selected = X[idx]
    y_selected = y[idx]
    return X_selected, y_selected

def balance_data(X, y, ratio=1.0):
    '''

    :param X:
    :param y:
    :param ratio: if positive examples are larger than negative ones, select ratio*neg positive class, if smaller,
    select ratio*pos negative examples  default 1.0 means positive examples equal to negative examples
    :return:
    '''
    X_pos = X[y>0] #positive class
    print("X_pos", X_pos.shape)
    y_pos = y[y>0]
    num_pos = y_pos.shape[0]
    print('num_pos',num_pos)

    X_neg = X[y==0]
    print("X_neg", X_neg.shape)
    y_neg = y[y==0]
    num_neg = y_neg.shape[0]
    print('num_neg',num_neg)

    if num_neg > num_pos:
        # random selected the indexes from negative samples
        if int(ratio * num_pos) > num_neg:
            sample_size = num_neg
        else:
            sample_size = int(ratio * num_pos)
        X_neg, y_neg = under_sampling(X_neg, y_neg, sample_size)
        print("undersample negative class and select", X_neg.shape)
        #combine the  positive  and selected negative class together
    else:
        if round(ratio * num_neg) > num_pos:
            sample_size = num_pos
        else:
            sample_size = int(ratio * num_neg)
        X_pos, y_pos = under_sampling(X_pos, y_pos, sample_size)
        print("undersample positive class and select", X_pos.shape)


    X_new = np.concatenate((X_pos, X_neg), axis=0)
    y_new = np.concatenate((y_pos, y_neg), axis=0)

    #shuffle the data
    X_new, y_new = shuffle(X_new, y_new, random_state=12)
    print("X new shape", X_new.shape)
    print("y new shape", y_new.shape)
    return X_new, y_new