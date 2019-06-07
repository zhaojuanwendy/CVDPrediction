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
from keras.layers import LSTM,Input,concatenate
from keras.models import Model
from keras.optimizers import RMSprop

from prepare_data import prepare_data
from prepare_data import prepare_features
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


def create_network(time_steps, feature_size, aux_feature_size, n_dense=3,
                   lstm_units=41,
                   dense_units=8,
                   activation='selu',
                   dropout=AlphaDropout,
                   dropout_rate=0.1,
                   recurrent_dropout = 0.2,
                   aux_output_weight=0.5,
                   learning_rates = 0.001
                   ):
    """Generic function to create a fully-connected neural network.
    # Arguments
        n_dense: int > 0. Number of dense layers.
        dense_units: int > 0. Number of dense units per layer.
        dropout: keras.layers.Layer. A dropout layer to apply.
        dropout_rate: 0 <= float <= 1. The rate of dropout.
        kernel_initializer: str. The initializer for the weights.
        optimizer: str/keras.optimizers.Optimizer. The optimizer to use.
        max_words: int > 0. The maximum number of words per data point.
    # Returns
        A Keras model instance (compiled).
    """
    main_input = Input(shape=(time_steps, feature_size), name='main_input')
    lstm_out = LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=recurrent_dropout)(main_input)

    auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

    auxiliary_input = Input(shape=(aux_feature_size,), name='aux_input')
    x = concatenate([lstm_out, auxiliary_input])

    for i in range(n_dense - 1):
        x = Dense(dense_units, activation=activation)(x)
        x = dropout(dropout_rate)(x)
    # And finally we add the main logistic regression layer
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)
    # This defines a model with two inputs and two outputs:
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
    optimizer = RMSprop(lr=learning_rates)

    model.compile(optimizer=optimizer,
                  loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
                  loss_weights={'main_output': 1., 'aux_output': aux_output_weight}, metrics=['accuracy'])

    return model


# fix random seed for reproducibility
def plot(history_model):
    plt.figure()
    plt.plot(history_model.history['val_main_output_loss'],
             'g--',
             label='Val Main output Loss')
    plt.plot(history_model.history['main_output_loss'],
             'r',
             label='Main output Loss')
    plt.plot(history_model.history['val_aux_output_loss'],
             'b--',
             label='Val AUX output Loss')
    plt.plot(history_model.history['aux_output_loss'],
             'y',
             label='Aux Loss')

    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    out_path = path.join(RESULT_PATH, 'figures')
    if not path.exists(out_path):
        util.mkdir_p(out_path)

    plt.savefig(path.join(out_path,'lstm_comparison_of_networks_loss.png'))

    plt.figure()
    plt.plot(history_model.history['val_main_output_acc'],
             'g--',
             label='Network 1 Val Main output acc')
    plt.plot(history_model.history['main_output_acc'],
             'r',
             label='Main output acc')

    plt.plot(history_model.history['val_aux_output_acc'],
             'b--',
             label='Val Aux acc')
    plt.plot(history_model.history['aux_output_acc'],
             'y',
             label='Aux acc')

    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig(path.join(out_path, 'lstm_comparison_of_networks_acc.png'))


seed = 7
np.random.seed(seed)

data_type = 'temporal'

# configure the network
network1 = {
    'n_dense': 2,
    'lstm_units': 41,
    'dense_units': 128,
    'activation': 'relu',
    'dropout': Dropout,
    'dropout_rate': 0.3,
    'recurrent_dropout': 0.2,
    'aux_output_weight': 0.5,
    'learning_rates': 0.003
}
# callback_early_stopping = [EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')]

records =[]
time_features_start = 13 #'Gender','RACE_B','RACE_W','AGE','DURATIOM_BF2007,'Smoking', 'MAX_BMI_missing', ...
time_steps = 7 # number of time intervals
feature_size = 74 #number of distinct temporal features(e.g blood pressure,BMI)

n_fold = 10

data_path = path.join(DATA_PATH,data_type)
X, y = prepare_data(data_path)
print('X shape:', X.shape)
rskf = RepeatedStratifiedKFold(n_splits=n_fold, n_repeats=1, random_state=42)
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

    X_train_time, X_train_aux = \
        prepare_features(X_train, time_features_start, time_steps, feature_size, use_snps_features,snps_size)

    X_test_time, X_test_aux = \
        prepare_features(X_test, time_features_start, time_steps,feature_size, use_snps_features,snps_size)

    aux_feature_size = time_features_start

    # create model
    model1 = create_network(time_steps,feature_size,aux_feature_size=aux_feature_size, **network1)
    batch_size = 128
    filepath = path.join(RESULT_PATH, 'models/lstm.model.hdf5')
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

    history_model1 = model1.fit({'main_input': X_train_time, 'aux_input': X_train_aux},
                                {'main_output': y_train, 'aux_output': y_train},
                                callbacks=[checkpointer, earlyStopping], verbose=1,
                                epochs=50, validation_split=0.11, batch_size=batch_size)


    model1.load_weights(filepath)
    y_score = model1.predict({'main_input': X_test_time, 'aux_input': X_test_aux})
    print('plotting')
    plot(history_model1)
    auc = roc_auc_score(y_test, y_score[0])  # main_input
    cvs_aucs.append(auc)
    average_precision = average_precision_score(y_test, y_score[0])
    cvs_ap.append(average_precision)

df_results_cv = pd.DataFrame({
    'a_iter': range(10),
    'auc': cvs_aucs,
    'average_prec': cvs_ap
})
detailed_result_csv = 'LSTM-nested-cv-detailed-{}-fold.csv'.format(n_fold)
df_results_cv.to_csv(path.join(RESULT_PATH, detailed_result_csv), index=False)

#end for cv
records.append({
    'a_data_type': data_type,
    'network': network1,
    'size_of_Batch': batch_size,
    'auc': '%.4f (+/- %.4f)' % (np.mean(cvs_aucs), np.std(cvs_aucs)),
    'average_precision': '%.4f (+/- %.4f)' % (np.mean(cvs_ap), np.std(cvs_ap))
})
result_csv = 'LSTM-nested-cv-average-{}-fold.csv'.format(n_fold)
pd.DataFrame(records).to_csv(path.join(RESULT_PATH, result_csv), index=False)
