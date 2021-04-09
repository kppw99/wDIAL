import lightgbm as lgb
from preprocessing import *

from sklearn import metrics
from sklearn.metrics import classification_report


def lightgbm(tr_X, tr_y, te_X, te_y):
    print('[#] Start LightGBM ...')
    params = {
        'learning_rate': 0.01,
        'max_depth': -1,
        'min_data_in_leaf': 20,
        'boosting': 'rf',
        'num_class': len(tr_y.unique()),
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'is_training_metric': True,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'num_leaves': 64,
        'bagging_freq': 5,
        'seed': 2021
    }

    train_data = lgb.Dataset(tr_X, label=tr_y)
    test_data = lgb.Dataset(te_X, label=te_y)

    model = lgb.train(params, train_data, 1300, test_data, verbose_eval=100, early_stopping_rounds=100)
    pred_y = model.predict(te_X).argmax(axis=1)

    print(classification_report(te_y.values, pred_y))
    print('[*] Accuracy (XGBoost): ', metrics.accuracy_score(te_y.values, pred_y))


def do_LightGBM(tr_file, te_file) -> None:
    tr_X, te_X, tr_y, te_y = load_data(tr_file, te_file, label_encoder=True, drop=['A'])
    lightgbm(tr_X, tr_y, te_X, te_y)


if __name__=='__main__':
    do_LightGBM('../data/all/ABFGI_train.csv', '../data/all/ABFGI_test.arff.csv')