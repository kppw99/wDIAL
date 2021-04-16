import numpy as np

from scipy.stats import entropy

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score

from keras.optimizers import Adamax
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Dense, BatchNormalization

from helpers import driver_dict


CLASS_NUM = len(driver_dict)
BATCH_SIZE = 30
EPOCHS = 100


def mlp(tr_X, tr_y, te_X, te_y):
    print('[#] Start MLP Classifier')
    def _base_model():
        model = Sequential()
        model.add(Dense(512, input_dim=tr_X.shape[1], kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(CLASS_NUM, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=Adamax(lr=0.001), metrics=['accuracy'])
        model.summary()
        return model

    model = _base_model()
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.01, patience=10)
    model.fit(tr_X, tr_y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
              shuffle=True, callbacks=[early_stopping])

    pred_y = np.argmax(model.predict(te_X), axis=1)
    report = classification_report(te_y, pred_y)
    prec, rec, f1, _support = precision_recall_fscore_support(te_y, pred_y)
    acc = metrics.accuracy_score(te_y, pred_y)

    print(report)
    print('[-] te_y distribution:', list(te_y).count(0), list(te_y).count(1),
          list(te_y).count(2), list(te_y).count(3))
    print('[-] pred_y distribution:', list(pred_y).count(0), list(pred_y).count(1),
          list(pred_y).count(2), list(pred_y).count(3))
    print('[*] Accuracy (MLP): ', acc)
    print('[*] Precision: ', prec)
    print('[*] Recall: ', rec)
    print('[*] F1 Score: ', f1)

    return model, acc, prec, rec, f1


def calculate_entropy(model, data):
    y = model.predict(data, batch_size=BATCH_SIZE)
    entropys = np.array([entropy(item, base=2) for item in y])
    return entropys.mean()


def svm(tr_X, tr_y, te_X, te_y):
    print('[#] Start SVM Classifier')
    svc = SVC(kernel='linear', C=1.0, verbose=False)
    svc.fit(tr_X, tr_y)
    pred_y = svc.predict(te_X)

    report = classification_report(te_y, pred_y)
    prec, rec, f1, _support = precision_recall_fscore_support(te_y, pred_y)
    acc = metrics.accuracy_score(te_y, pred_y)

    print(report)
    print('[*] Accuracy (SVM): ', acc)
    print('[*] Precision: ', prec)
    print('[*] Recall: ', rec)
    print('[*] F1 Score: ', f1)

    return svc, acc, prec, rec, f1


def bagging_svm(tr_X, tr_y, te_X, te_y, n_estimators):
    print('[#] Start SVM Bagging Classifier')
    svc = SVC(kernel='linear')
    n_estimators = n_estimators
    n_jobs = -1
    verbose = 0
    model = BaggingClassifier(base_estimator=svc, n_estimators=n_estimators,
                              max_samples=1. / n_estimators, n_jobs=n_jobs, verbose=verbose)
    model.fit(tr_X, tr_y)
    pred_y = model.predict(te_X)

    report = classification_report(te_y, pred_y)
    prec, rec, f1, _support = precision_recall_fscore_support(te_y, pred_y)
    acc = metrics.accuracy_score(te_y, pred_y)

    print(report)
    print('[*] Accuracy (SVM): ', acc)
    print('[*] Precision: ', prec)
    print('[*] Recall: ', rec)
    print('[*] F1 Score: ', f1)

    return model, acc, prec, rec, f1


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


def calculate_bagging_distance(trained_svm, pool_data):
    y = trained_svm.predict_proba(pool_data)
    entropys = np.array([entropy(item, base=2) for item in y])
    return entropys.mean()


def random_forest(tr_X, tr_y, te_X, te_y):
    print('[#] Start RandomForest Classifier')
    model = RandomForestClassifier(n_jobs=-1, verbose=0)
    model.fit(tr_X, tr_y)
    pred_y = model.predict(te_X)

    report = classification_report(te_y, pred_y)
    prec, rec, f1, _support = precision_recall_fscore_support(te_y, pred_y)
    acc = metrics.accuracy_score(te_y, pred_y)
    ras = roc_auc_score(te_y, model.predict_proba(te_X), multi_class="ovr", average='macro')

    print(report)
    print('[*] Accuracy (Random Forest): ', acc)
    print('[*] ROC_AUC_SCORE: ', ras)
    print('[*] Precision: ', prec)
    print('[*] Recall: ', rec)
    print('[*] F1 Score: ', f1)

    return model, acc, prec, rec, f1


def calculate_proba(model, data):
    y = model.predict_proba(data)
    entropys = np.array([entropy(item, base=2) for item in y])
    return entropys.mean()