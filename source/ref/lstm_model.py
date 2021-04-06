import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from pprint import pprint
from keras import optimizers
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils.io_utils import HDF5Matrix
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, LSTM, Bidirectional, Activation
from sklearn.metrics import classification_report # accuracy_score, precision_score, recall_score, f1_score,


############ Gobal Variables ############
EPOCH = 200
BATCH = 100
OUTPUT_DIM = 5
TIMESTEP = 60
FEATURES = 1092
#########################################


def _load_from_csv(tr_file, te_file):
    def _read_from_csv(target):
        print("[*] Data Loading from csv: ", target)
        # return pd.read_csv(target).rename(columns=lambda x: x.strip())
        df_list = []; lines_in_chunk = 500
        lines_number = sum(1 for line in open(target))
        reader = pd.read_csv(target, chunksize=lines_in_chunk)
        for chunk in tqdm(reader, total=lines_number/lines_in_chunk):
            df_list.append(chunk)
        return pd.concat(df_list, ignore_index=True)

    tr_df = _read_from_csv(tr_file)
    te_df = _read_from_csv(te_file)
    return tr_df, te_df


def _split_data(tr_df, te_df):
    tr_df.loc[(tr_df.Driver == 'A'), 'Driver'] = 0
    tr_df.loc[(tr_df.Driver == 'B'), 'Driver'] = 1
    tr_df.loc[(tr_df.Driver == 'F'), 'Driver'] = 2
    tr_df.loc[(tr_df.Driver == 'G'), 'Driver'] = 3
    tr_df.loc[(tr_df.Driver == 'I'), 'Driver'] = 4

    te_df.loc[(te_df.Driver == 'A'), 'Driver'] = 0
    te_df.loc[(te_df.Driver == 'B'), 'Driver'] = 1
    te_df.loc[(te_df.Driver == 'F'), 'Driver'] = 2
    te_df.loc[(te_df.Driver == 'G'), 'Driver'] = 3
    te_df.loc[(te_df.Driver == 'I'), 'Driver'] = 4

    #TODO: for debugging
    print('[-] train data label:')
    print(str(sum(tr_df['Driver'] == 0)) + ', ' + str(sum(tr_df['Driver'] == 1)) + ', ' +
          str(sum(tr_df['Driver'] == 2)) + ', ' + str(sum(tr_df['Driver'] == 3)) + ', ' +
          str(sum(tr_df['Driver'] == 4)))

    # TODO: for debugging
    print('[-] test data label:')
    print(str(sum(te_df['Driver'] == 0)) + ', ' + str(sum(te_df['Driver'] == 1)) + ', ' +
          str(sum(te_df['Driver'] == 2)) + ', ' + str(sum(te_df['Driver'] == 3)) + ', ' +
          str(sum(te_df['Driver'] == 4)))

    tr_X = tr_df.iloc[:, 0:1092]; tr_y = to_categorical(tr_df['Driver'], 5)
    te_X = te_df.iloc[:, 0:1092]; te_y = to_categorical(te_df['Driver'], 5)

    print('[-] tr_X.shape, te_X.shape, tr_y.shape, te_y.shape:')
    print(tr_X.shape, te_X.shape, tr_y.shape, te_y.shape)
    return tr_X, te_X, tr_y, te_y


def _make_data_cols_name():
    cols = []
    for i in range(1092):
        cols.append('F-' + str(i+1))
    return cols


def _load_from_hdf5(path, trip_cnt=None):
    print("Data Loading from hdf5: ", path)
    def _read_from_hdf5(target):
        with h5py.File(target, "r") as f:
            a_group_key = list(f.keys())[0]
            return list(f[a_group_key])
    targets = sorted([x for x in Path(path).glob("*.h5")])
    lists = list()
    trip_cnt = 9 if trip_cnt is None else 9 if trip_cnt > 9 else 1 if trip_cnt < 1 else trip_cnt
    for i, target in enumerate(tqdm(targets, total=len(targets)-1)):
        if trip_cnt == i: break
        lists.extend(_read_from_hdf5(target))
    tr_df = pd.DataFrame(lists, columns=_make_data_cols_name())
    te_df = pd.DataFrame(_read_from_hdf5(targets[-1]), columns=_make_data_cols_name())

    if path.find('driver_A') >= 0:
        tr_df['Driver'] = 'A'
        te_df['Driver'] = 'A'
    elif path.find('driver_B') >= 0:
        tr_df['Driver'] = 'B'
        te_df['Driver'] = 'B'
    elif path.find('driver_F') >= 0:
        tr_df['Driver'] = 'F'
        te_df['Driver'] = 'F'
    elif path.find('driver_G') >= 0:
        tr_df['Driver'] = 'G'
        te_df['Driver'] = 'G'
    elif path.find('driver_I') >= 0:
        tr_df['Driver'] = 'I'
        te_df['Driver'] = 'I'

    return tr_df.reset_index(drop=True), te_df.reset_index(drop=True)


def load_and_split_from_csv(tr_file, te_file):
    tr_df, te_df = _load_from_csv(tr_file, te_file)
    return _split_data(tr_df, te_df)


def load_and_split_from_hdf5(trip_cnt=None):
    drivers = ['driver/driver_A', 'driver/driver_B', 'driver/driver_F',
               'driver/driver_G', 'driver/driver_I']

    tr_list = list(); te_list = list()
    for driver in drivers:
        tr_tmp, te_tmp = _load_from_hdf5(driver, trip_cnt=trip_cnt)
        tr_list.append(tr_tmp); te_list.append(te_tmp)

    tr_df = pd.concat(tr_list, axis=0)
    te_df = pd.concat(te_list, axis=0)
    tr_df = tr_df.reset_index(drop=True)
    te_df = te_df.reset_index(drop=True)

    return _split_data(tr_df, te_df)


def reshape_for_timestep(df):
    print('[*] Transform shape of dataset for Timestep')
    out = []
    stride = 1
    df = df.values
    for i in tqdm(range(0, df.shape[0] - TIMESTEP + 1, stride), mininterval=1):
        out.append(df[i:i + TIMESTEP, :])
    print('[-] reshape for timestep: ' + str(np.shape(out)))
    return np.array(out)


def multi_blstm():
    print('[*] Model Training ...')
    model = Sequential()

    model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=(TIMESTEP, FEATURES)))
    model.add(Activation('relu'))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Activation('relu'))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Activation('relu'))
    model.add(Dense(OUTPUT_DIM))
    model.add(Activation('softmax'))

    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def train_and_predict(func, tr_X, tr_y, te_X, te_y):
    print('[*] Train the model ...')
    early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=5)
    model = KerasClassifier(build_fn=func, epochs=EPOCH, batch_size=BATCH,
                            validation_split=0.2, verbose=1)
    model.fit(tr_X, tr_y, callbacks=[early_stop])

    print('[*] Predict result')
    # pred_y = np.argmax(model.predict(te_X), axis=1)
    pred_y = model.predict(te_X)
    te_y = np.argmax(te_y, axis=1)
    result = classification_report(te_y, pred_y, output_dict=True)

    #TODO: for debugging
    print(sum(pred_y == 0))
    print(sum(pred_y == 1))
    print(sum(pred_y == 2))
    print(sum(pred_y == 3))
    print(sum(pred_y == 4))

    return result['accuracy']


if __name__=='__main__':
    # yourArray = np.random.randn(64, 64)  # just an example
    # print(get_strides(yourArray, 5, 1).shape)

    print('[#] Start Active Learning Framework!')
    tr_X, te_X, tr_y, te_y = load_and_split_from_csv('driver/all/ABFGI_train.csv',
                                                     'driver/all/ABFGI_test.arff.csv')

    print(tr_X.shape, te_X.shape, tr_y.shape, te_y.shape) #TODO: for debugging
    print(tr_X.head()) #TODO: for debugging

    # tr_X, te_X, tr_y, te_y = load_and_split_from_hdf5()
    # print(tr_X.shape, te_X.shape, tr_y.shape, te_y.shape) #TODO: for debugging
    # print(tr_X.head()) #TODO: for debugging

    report = list()
    tr_X = reshape_for_timestep(tr_X); te_X = reshape_for_timestep(te_X)
    result = train_and_predict(multi_blstm, tr_X, tr_y, te_X, te_y)
    report.append(result)

    pprint(report)
