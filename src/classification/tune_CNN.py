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

from keras.layers import Convolution1D,MaxPooling1D
from keras.optimizers import Adam,Nadam
from keras.layers import Dense, Flatten, Dropout,LeakyReLU
from keras.layers.noise import AlphaDropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.layers import Input,concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization

from prepare_data import prepare_data
from prepare_data import prepare_features
#
from sys import argv
import utility as util
#
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


def create_network(time_steps, feature_size, aux_feature_size, filters=256, kernel_size=64,
                   learning_rate=0.0001,
                   aux_output_weight=0.5
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
    x = Convolution1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(main_input)
    x = MaxPooling1D(pool_size=3)(x)
    # x = Flatten()(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU()(x)
    # x = Convolution1D(filters=8, kernel_size=4, padding='same')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU()(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    # x = LeakyReLU()(x)
    auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(x)
    auxiliary_input = Input(shape=(aux_feature_size,), name='aux_input')
    x = concatenate([x, auxiliary_input])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(8, activation='relu')(x)
    # And finally we add the main logistic regression layer
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)
    # This defines a model with two inputs and two outputs:
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

    model.compile(optimizer=Nadam(learning_rate),
                  loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
                   metrics=['accuracy'])

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

    plt.savefig(path.join(out_path, 'cnn_comparison_of_networks_loss.png'))

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

    plt.savefig(path.join(out_path, 'cnn_comparison_of_networks_acc.png'))

root_data_path = path.join(DATA_PATH,'processed')
results_path = RESULT_PATH

seed = 7
np.random.seed(seed)
# load dataset
data_type = 'temporal'

# split into input (X) and output (Y) variables
network1 = {
    'filters': 48, #64
    'kernel_size': 7,
    # 'n_dense': 2,
    # 'dense_units1': 128,
    # 'dense_units2': 8,
    # 'activation': 'relu',
    # 'dropout': Dropout,
    # 'dropout_rate': 0.2,
    'learning_rate': 0.0005,
    'aux_output_weight': 0.5
}
callback_early_stopping = [EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')]

records =[]
time_features_start = 13 #'Gender','RACE_B','RACE_W','AGE','DURATIOM_BF2007,'Smoking', 'MAX_BMI_missing', ...
time_steps = 7 # number of time intervals
feature_size = 74 #number of distinct temporal features(e.g blood pressure,BMI)

data_path = path.join(DATA_PATH, data_type)
X, y = prepare_data(data_path)
print('loading data shape:', X.shape)
print('X shape:', X.shape)

rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=42)
index = 0
cvs_aucs=[]
cvs_ap =[]

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

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.000001, verbose=1)

    filepath = path.join(RESULT_PATH, 'models/cnn.model.hdf5')
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)

    history_model1 = model1.fit({'main_input': X_train_time, 'aux_input': X_train_aux},
              {'main_output': y_train, 'aux_output': y_train}, callbacks=[reduce_lr, checkpointer], verbose=1,epochs=80, validation_split=0.11,
              batch_size=batch_size)

    model1.load_weights(filepath)
    y_score = model1.predict({'main_input': X_test_time, 'aux_input': X_test_aux})
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
detailed_result_csv = 'CNN-nested-cv-detailed-10-fold.csv'
df_results_cv.to_csv(path.join(results_path, detailed_result_csv), index=False)

#end for cv
records.append({
    'a_data_type': data_type,
    'network': network1,
    'size_of_Batch': batch_size,
    'auc': '%.4f (+/- %.4f)' % (np.mean(cvs_aucs), np.std(cvs_aucs)),
    'average_precision': '%.4f (+/- %.4f)' % (np.mean(cvs_ap), np.std(cvs_ap))
})
result_csv = 'CNN-nested-cv-average-10-fold.csv'
pd.DataFrame(records).to_csv(path.join(results_path, result_csv), index=False)
