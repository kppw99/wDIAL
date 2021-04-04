import pandas as pd
from xgboost import XGBClassifier

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def load_data(tr_file, te_file):
    def _read_from_csv(target):
        return pd.read_csv(target).rename(columns=lambda x: x.strip())
    tr_df = _read_from_csv(tr_file)
    te_df = _read_from_csv(te_file)

    tr_X = tr_df.iloc[:, 0:1092]; tr_y = tr_df['Driver']
    te_X = te_df.iloc[:, 0:1092]; te_y = te_df['Driver']

    return tr_X, te_X, tr_y, te_y


def load_toy_data():
    iris_dataset = load_iris()
    tr_X, te_X, tr_y, te_y = train_test_split(iris_dataset['data'],
                                          iris_dataset['target'],
                                          test_size=0.2)

    sc = StandardScaler()
    sc.fit(tr_X)
    tr_X_std = sc.transform(tr_X)
    te_X_std = sc.transform(te_X)

    return tr_X_std, te_X_std, tr_y, te_y


if __name__=='__main__':
    dataset = 'toy'

    if dataset == 'toy':
        tr_X, te_X, tr_y, te_y = load_toy_data()
    else:
        tr_X, te_X, tr_y, te_y = \
            load_data('driver/all/ABFGI_train.csv', 'driver/all/ABFGI_test.arff.csv')

    print('[#] Start SVM Classifier')
    svc = SVC(kernel='poly', C=1.0)
    svc.fit(tr_X, tr_y)
    pred_y = svc.predict(te_X)
    print('[*] 정확도 (SVM): ', metrics.accuracy_score(te_y, pred_y))

    print('[#] Start RandomForest Classifier')
    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(tr_X, tr_y)
    pred_y = forest.predict(te_X)
    print('[*] 정확도 (RandomForest): ', metrics.accuracy_score(te_y, pred_y))

    print('[#] Start XGBoost Classifier')
    xgb = XGBClassifier(silent=False, booster='gbtree',
                        scale_pos_weight=1, learning_rate=0.01,
                        colsample_bytree=0.4, subsample=0.8,
                        objective='binary:logistic', n_estimators=100,
                        max_depth=4, gamma=10, seed=777)

    pred_y = xgb.fit(tr_X, tr_y).predict(te_X)
    print(classification_report(te_y, pred_y))
    print('[*] 정확도 (XGBoost): ', metrics.accuracy_score(te_y, pred_y))
