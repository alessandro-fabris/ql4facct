import quapy as qp
import numpy as np
from data import adultcsv_loader
from common import *
from sklearn.linear_model import LogisticRegression

from quapy.method.aggregative import CC, ACC, PACC, EMQ, HDy

qp.environ['SAMPLE_SIZE'] = 100

def new_cls():
    return LogisticRegression()

# def new_quantifier():
    # return qp.method.aggregative.CC(new_classifier())
    # return qp.method.aggregative.ACC(new_classifier())

X, y, A = adultcsv_loader("./adult.csv", protected_attr='gender')

D1, D2, D3 = split_data(X, y, A, seed=0)

f = new_cls()

for Q in [CC(new_cls()), ACC(new_cls()), PACC(new_cls()), EMQ(new_cls()), HDy(new_cls())]:

    # bias_ave, bias_std, error_ave, error_std = eval_prevalence_variations_D1(D1, D2, D3, f, Q, nprevs=11)
    # bias_ave, bias_std, error_ave, error_std = eval_prevalence_variations_D2(D1, D2, D3, f, Q, nprevs=11)
    bias_ave, bias_std, error_ave, error_std = eval_prevalence_variations_D3(D1, D2, D3, f, Q, nprevs=11)
    print('Classifier: ', f.__class__.__name__)
    print('Quantifier: ', Q.__class__.__name__)
    print(f'bias = {bias_ave:.5f}+-{bias_std:.5f}')
    print(f'error = {error_ave:.5f}+-{error_std:.5f}')







