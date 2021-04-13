from preprocessing import *
from svm import *
from random_forest import *
from xgb import *

# from sklearn import datasets
# from sklearn.model_selection import train_test_split
#
#
# def load_toy_dataset():
#     iris = datasets.load_iris()
#     return train_test_split(iris.data, iris.target, stratify=iris.target)
#
#
# train_set = A
# unlabeled_set = [B, C, D, E, F, G, H]  # trip
#
# for iteration_num in range(len(unlabeled_set)):
#
#     if iteration_num == 1:
#         train
#         classifer
#         with A
#
#     else:
#         train
#         classifier
#         with A + selected_set
#
#     let
#     trained_classifier -> rank
#     of
#     uncertainity
#     at
#     unlabeled
#     set