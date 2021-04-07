from preprocessing import *

import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import classification_report


def _xgboost(tr_X, tr_y, te_X, te_y):
    print('[#] Start XGBoost Classifier')
    xg_model = xgb.XGBClassifier(silent=False, booster='gbtree',
                        scale_pos_weight=1, learning_rate=0.01,
                        colsample_bytree=0.4, subsample=0.8, verbose=1,
                        objective='binary:logistic', n_estimators=100,
                        max_depth=4, gamma=10, seed=777)

    pred_y = xg_model.fit(tr_X, tr_y).predict(te_X)
    print(classification_report(te_y, pred_y))
    print('[*] Accuracy (XGBoost): ', metrics.accuracy_score(te_y, pred_y))


def do_XGBoost(tr_file, te_file) -> None:
    tr_X, te_X, tr_y, te_y = load_data(tr_file, te_file)
    _xgboost(tr_X, tr_y, te_X, te_y)


if __name__=='__main__':
    do_XGBoost('../data/train.csv', '../data/test.csv')

