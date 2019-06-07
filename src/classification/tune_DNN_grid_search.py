# Use scikit-learn to grid search the activation function
import pandas as pd
import numpy as np
from os import path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.noise import AlphaDropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint

from prepare_data import prepare_data
from sys import argv
import utility as util
from definitions import LOGS_PATH
from definitions import RESULT_PATH
from definitions import DATA_PATH


import logging
import time
time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

log_files = path.join(LOGS_PATH, 'log_tune_dnn.txt')
logging.basicConfig(filename=log_files+str(time_stamp), level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug('This is a log message.')

def create_network(input_shape, n_dense=3,
                   dense_units=8,
                   activation='selu',
                   dropout=AlphaDropout,
                   dropout_rate=0.1,
                   kernel_initializer='lecun_normal',
                   optimizer='adam'
                   ):
    """Generic function to create a fully-connected neural network.
    # Arguments
        n_dense: int > 0. Number of dense layers.
        dense_units: int > 0. Number of dense units per layer.
        dropout: keras.layers.Layer. A dropout layer to apply.
        dropout_rate: 0 <= float <= 1. The rate of dropout.
        kernel_initializer: str. The initializer for the weights.
        optimizer: str/keras.optimizers.Optimizer. The optimizer to use.
        num_classes: int > 0. The number of classes to predict.
        max_words: int > 0. The maximum number of words per data point.
    # Returns
        A Keras model instance (compiled).
    """
    model = Sequential()
    model.add(Dense(dense_units, input_shape=(input_shape,),
                    kernel_initializer=kernel_initializer))
    model.add(Activation(activation))
    model.add(dropout(dropout_rate))

    for i in range(n_dense - 1):
        model.add(Dense(dense_units, kernel_initializer=kernel_initializer))
        model.add(Activation(activation))
        model.add(dropout(dropout_rate))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
# fix random seed for reproducibility
def plot(history_model):
    plt.plot(history_model.history['val_loss'],
             'g',
             label='Network 1 Val Loss')
    plt.plot(history_model.history['loss'],
             'r',
             label='Network Loss')

    plt.plot(history_model.history['val_acc'],
             'b-',
             label='Network 1 Val acc')
    plt.plot(history_model.history['acc'],
             'y-',
             label='Network acc')

    plt.xlabel('Epochs')
    plt.ylabel('acc/loss')
    plt.legend()

    out_path = path.join(RESULT_PATH, 'figures')
    if not path.exists(out_path):
        util.mkdir_p(out_path)
    plt.savefig(path.join(out_path, 'comparison_of_networks_acc.png'))



seed = 7
np.random.seed(seed)

data_type = 'benchmark'
myargs = util.getopts(argv)
if '-d' in myargs:  #
    print(myargs['-d'])
    data_type = myargs['-d']


# configure the network

network1 = {
    'n_dense': 3,
    'dense_units': 8,
    'activation': 'relu',
    'dropout': Dropout,
    'dropout_rate': 0.1,
    'kernel_initializer': 'glorot_uniform',
    'optimizer': 'adam'
}
network2 = {
    'n_dense': 3,
    'dense_units': 16,
    'activation': 'relu',
    'dropout': Dropout,
    'dropout_rate': 0.2,
    'kernel_initializer': 'glorot_uniform',
    'optimizer': 'adam'
}
network3 = {
    'n_dense': 3,
    'dense_units': 32,
    'activation': 'relu',
    'dropout': Dropout,
    'dropout_rate': 0.4,
    'kernel_initializer': 'glorot_uniform',
    'optimizer': 'adam'
}
callback_early_stopping = [EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')]

records = []
# load the data
X, y = prepare_data(path.join(DATA_PATH, data_type))

print(X.shape)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.5, random_state=0)

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
index = 0
ncvs_aucs=[]
ncvs_ap =[]
ncv_best_params = []
for train_index, test_index in rskf.split(X, y):
    print("iter", index)
    index += 1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(X_train.shape[0])
    print(X_test.shape[0])
    # Define the scaler
    scaler = StandardScaler().fit(X_train)
    # Scale the train set
    X_train = scaler.transform(X_train)
    # Scale the test set
    X_test = scaler.transform(X_test)
    # k-fold cross validation to find the best network parameters
    search_record = []
    for network_params in [network1, network2]:
        cvs_aucs = []
        cvs_ap = []
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
        for cv_tr, cv_val in cv.split(X_train, y_train):
            X_tr, X_val = X[cv_tr], X[cv_val]
            y_tr, y_val = y[cv_tr], y[cv_val]
            print(y_tr.shape[0])
            print(y_val.shape[0])
            batch_size = 128
            model = create_network(X_tr.shape[1], **network_params)
            # reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
            # checkpointer = ModelCheckpoint(filepath="dnn.model.hdf5", verbose=1, save_best_only=True)
            # history_model = model.fit(X_train,
            #                     y_train,
            #                     batch_size=batch_size,
            #                     epochs=50,
            #                     verbose=1,
            #                     validation_split=0.1, callbacks=[reduce_lr, checkpointer])
            history_model = model.fit(X_tr,
                                                          y_tr,
                                                          batch_size=batch_size,
                                                          epochs=30,
                                                          verbose=1)
            y_pred_score = model.predict(X_val)
            auc = roc_auc_score(y_val, y_pred_score)  # main_input
            cvs_aucs.append(auc)
            average_precision = average_precision_score(y_val, y_pred_score)
            cvs_ap.append(average_precision)

        search_record.append({
            'network': network_params,
            'size_of_Batch': batch_size,
            'auc_mean': '%.4f' % np.mean(cvs_aucs),
            'auc_std': '%.4f' % np.std(cvs_aucs),
            'ap_mean': '%.4f' % np.mean(cvs_ap),
            'ap_std': '%.4f' % np.std(cvs_ap)
        })

    final_result = pd.DataFrame(search_record)
    idx = np.argsort(final_result.auc_mean.values)
    best_params = final_result.loc[idx[0],'network']
    ncv_best_params.append(best_params)
    print(best_params)
    final_model = create_network(X_train.shape[1], **best_params)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)

    filepath = path.join(RESULT_PATH, 'models/dnn.model.hdf5')
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
    final_model.fit(X_train,
                                                  y_train,
                                                  batch_size=batch_size,
                                                  epochs=50,
                                                  verbose=1,
                                                  validation_split=0.1, callbacks=[reduce_lr, checkpointer])
    final_model.load_weights(filepath)
    y_score = final_model.predict(X_test)
    ncvs_aucs.append(roc_auc_score(y_test, y_score))
    ncvs_ap.append(average_precision_score(y_test, y_score))

for mean, stdev, param in zip(ncvs_aucs, ncvs_aucs, ncv_best_params):
    logging.debug("%f (%f) best AUC with: %r" % (mean, stdev, param))
    print("%f (%f) with: %r" % (mean, stdev, param))

#end for nested cv
records.append({
    'data_type': data_type,
    'auc': '%.4f (+/- %.4f)' % (np.mean(ncvs_aucs), np.std(ncvs_aucs)),
    'average_precision': '%.4f (+/- %.4f)' % (np.mean(ncvs_ap), np.std(ncvs_ap))
})
result_csv = 'DNN-nested-cv-5-fold-WB-25y-bp.csv'
pd.DataFrame(records).to_csv(path.join(RESULT_PATH, result_csv), index=False)
