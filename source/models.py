import numpy as np

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Dense

from helpers import driver_dict


CLASS_NUM = len(driver_dict)
BATCH_SIZE = 30
EPOCHS = 100


def mlp(tr_X, tr_y, te_X, te_y):
    print('[#] Start MLP Classifier')

    tr_y = to_categorical(tr_y, CLASS_NUM)
    te_y = to_categorical(te_y, CLASS_NUM)

    model = Sequential()
    model.add(Dense(512, input_dim=tr_X.shape[1]))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(CLASS_NUM, activation='softmax'))

    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    # model.summary()

    early_stopping = EarlyStopping(monitor='loss', patience=10)
    model.fit(tr_X, tr_y, validation_split=0.2, batch_size=BATCH_SIZE, epochs=EPOCHS,
              verbose=0, shuffle=False, callbacks=[early_stopping])

    pred_y = np.argmax(model.predict(te_X, batch_size=BATCH_SIZE), axis=1)
    te_y = np.argmax(te_y, axis=1)
    report = classification_report(te_y, pred_y)

    print('pred_y distribution')
    print(list(pred_y).count(0), list(pred_y).count(1),
          list(pred_y).count(2), list(pred_y).count(3))

    print('te_y distribution')
    print(list(te_y).count(0), list(te_y).count(1),
          list(te_y).count(2), list(te_y).count(3))

    print(report)

    prec, rec, f1, _support = precision_recall_fscore_support(te_y, pred_y)

    acc = metrics.accuracy_score(te_y, pred_y)
    print('[*] Accuracy (SVM): ', acc)
    print('[*] Precision: ', prec)
    print('[*] Recall: ', rec)
    print('[*] F1 Score: ', f1)

    return model, acc, prec, rec, f1


def calculate_entropy(model, data):
    def _entropy(x):
        x = np.array(x)
        return -np.sum(x * np.log2(x))
    y = model.predict(data, batch_size=BATCH_SIZE)
    entropys = np.array([_entropy(item) for item in y])
    return entropys.mean()


def svm(tr_X, tr_y, te_X, te_y):
    print('[#] Start SVM Classifier')
    svc = SVC(kernel='linear', C=1.0, verbose=False)
    svc.fit(tr_X, tr_y)
    pred_y = svc.predict(te_X)
    
    report = classification_report(te_y, pred_y)
    print(report)
    
    prec, rec, f1, _support = precision_recall_fscore_support(te_y, pred_y)
    
    acc = metrics.accuracy_score(te_y, pred_y)
    print('[*] Accuracy (SVM): ', acc)
    print('[*] Precision: ', prec)
    print('[*] Recall: ', rec)
    print('[*] F1 Score: ', f1)
    
    return svc, acc, prec, rec, f1


def calculate_distance(trained_svm, pool_data):
    # decision boundary로부터의 거리
    y = trained_svm.decision_function(pool_data) # (datapoint 개수, 각 decision boundary로부터의 거리)
    w_norm = np.linalg.norm(trained_svm.coef_) # scalar value -> coefficient
    dist = y / w_norm # data 개수 x 5

    # 일단, datapoint가 decision boundary와의 거리를 abs() 먹여서 양수로 만든다
    abs_dist = abs(dist) # 양수로 바꿔주고
    
    min_dist = abs_dist.min(axis=1) # decision boundary와의 거리들 중 최소 거리를 parsing
    # mean_dist = abs_dist.mean(axis=1)
        
    target_dist = min_dist.mean()
        
    return target_dist