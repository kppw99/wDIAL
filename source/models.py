import numpy as np

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support


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