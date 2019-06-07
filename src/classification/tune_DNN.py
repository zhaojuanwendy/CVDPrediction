# Use scikit-learn to grid search the activation function
import pandas as pd
import numpy as np
from os import path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.noise import AlphaDropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.optimizers import Adam


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

def create_fix_layers_network(input_shape,units1=128,units2=64,units3=8,
                       dropout_rate1=0.2,dropout_rate2=0.1,dropout_rate3=0.1, lr=0.00004):
    model = Sequential()
    model.add(Dense(units1, activation='relu', input_shape=(input_shape,)))
    # Add one hidden layer
    model.add(Dropout(dropout_rate1))
    model.add(Dense(units2, activation='relu'))
    model.add(Dropout(dropout_rate2))
    model.add(Dense(units3, activation='relu'))
    model.add(Dropout(dropout_rate3))
    model.add(Dense(1, activation='sigmoid'))

    opt = Adam(lr=lr)  # lr=0.00004
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
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

    plt.savefig(path.join(RESULT_PATH, 'comparison_of_dnn_networks_acc.png'))


seed = 7
np.random.seed(seed)
# load dataset

network1 = {
    'units1': 8,#128,
    'units2': 4,#64,
    'units3':4,
    'dropout_rate1': 0.2,
    'dropout_rate2': 0.1,
    'dropout_rate3': 0.1,
    'lr': 0.00004
}
batch_size = 128
callback_early_stopping = [EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')]

records = []

#load the data
myargs = util.getopts(argv)

data_type = 'benchmark'

if '-d' in myargs:  #
    print(myargs['-d'])
    data_type = myargs['-d']

X, y = prepare_data(path.join(DATA_PATH, data_type))

print(X.shape)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.5, random_state=0)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
index = 0
cvs_aucs = []
cvs_ap = []
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
    # create model
    model = create_fix_layers_network(X_train.shape[1], **network1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
    checkpointer = ModelCheckpoint(filepath=path.join(RESULT_PATH, "models/dnn.model.hdf5"), verbose=1, save_best_only=True)

    history_model = model.fit(X_train,
                                y_train,
                                batch_size=batch_size,
                                epochs=100,
                                verbose=1,
                                validation_split=0.11, callbacks=[reduce_lr, checkpointer])
    plot(history_model)
    model.load_weights(path.join(RESULT_PATH, "models/dnn.model.hdf5"))
    y_score = model.predict(X_test)
    auc = roc_auc_score(y_test, y_score)  # main_input
    cvs_aucs.append(auc)
    average_precision = average_precision_score(y_test, y_score)
    cvs_ap.append(average_precision)

#end for cv
records.append({
    'data_type': data_type,
    'network': network1,
    'size_of_Batch': batch_size,
    'auc': '%.4f (+/- %.4f)' % (np.mean(cvs_aucs), np.std(cvs_aucs)),
    'average_precision': '%.4f (+/- %.4f)' % (np.mean(cvs_ap), np.std(cvs_ap))
})
result_csv = 'DNN-nested-cv-10-fold-WB-25y-bp.csv'
pd.DataFrame(records).to_csv(path.join(RESULT_PATH, result_csv), index=False)
