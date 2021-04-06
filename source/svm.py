from preprocessing import *

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def load_data(tr_file, te_file):
    """각 Driver의 trip data까지의 data를 모아서 Train dataset을 구성한다.

        Args:
            trip: 0 ~ 8
            save: None (default) or filename -> dataset을 csv 파일로 저장한다.

        Returns:
            train_data, train_label, test_data, test_label
        """
    print('[#] Loading data:', tr_file, te_file)
    tr_df = get_dataset(tr_file)
    te_df = get_dataset(te_file)

    tr_X = tr_df.iloc[:, 0:1092];
    tr_y = tr_df['Driver']
    te_X = te_df.iloc[:, 0:1092];
    te_y = te_df['Driver']

    del tr_df; del te_df
    return tr_X, te_X, tr_y, te_y


def svm(tr_X, tr_y, te_X, te_y):
    print('[#] Start SVM Classifier')
    svc = SVC(kernel='poly', C=1.0, verbose=1)
    svc.fit(tr_X, tr_y)
    pred_y = svc.predict(te_X)
    print(classification_report(te_y, pred_y))
    print('[*] 정확도 (SVM): ', metrics.accuracy_score(te_y, pred_y))


def do_svm(tr_file, te_file):
    tr_X, te_X, tr_y, te_y = load_data(tr_file, te_file)
    svm(tr_X, tr_y, te_X, te_y)


if __name__=='__main__':
    # tr_file = '../data/train.csv'
    # te_file = '../data/test.csv'
    # tr_X, te_X, tr_y, te_y = load_data(tr_file, te_file)
    # svm(tr_X, tr_y, te_X, te_y)
    None