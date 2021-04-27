from copy import deepcopy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVR
import numpy as np
import quapy as qp
from common import gen_split_data, classify, independence_gap
from data import adultcsv_loader
from quapy.data import LabelledCollection
from quapy.method.aggregative import PACC
from quapy.method.base import BaseQuantifier



class IndependenceEstimator:

    def __init__(self, quantifier:BaseQuantifier, sample_size=500, n_prevpoints=21, n_repetitions=1000):
        self.q0 = deepcopy(quantifier)
        self.q1 = deepcopy(quantifier)
        self.sample_size = sample_size
        self.n_prevpoints = n_prevpoints
        self.n_repetitions = n_repetitions

    def fit(self, s0:LabelledCollection, s1:LabelledCollection):
        s0tr, s0te = s0.split_stratified()
        s1tr, s1te = s1.split_stratified()

        self.q0.fit(s0tr)
        self.q1.fit(s1tr)

        true_s0, estim_s0 = qp.evaluation.artificial_sampling_prediction(self.q0, s0te, sample_size=self.sample_size,
                                                                         n_prevpoints=self.n_prevpoints,
                                                                         n_repetitions=self.n_repetitions, n_jobs=-1)
        true_s1, estim_s1 = qp.evaluation.artificial_sampling_prediction(self.q1, s1te, sample_size=self.sample_size,
                                                                         n_prevpoints=self.n_prevpoints,
                                                                         n_repetitions=self.n_repetitions, n_jobs=-1)
        idx = np.random.permutation(len(true_s0))
        true_s0 = true_s0[idx]
        estim_s0 = estim_s0[idx]

        idx = np.random.permutation(len(true_s1))
        true_s1 = true_s1[idx]
        estim_s1 = estim_s1[idx]

        true_gap = independence_gap(true_s0[:,1], true_s1[:,1])
        estim_gap = independence_gap(estim_s0[:,1], estim_s1[:,1])

        self.reg = LinearSVR().fit(X=estim_gap.reshape(-1,1), y=true_gap.reshape(-1,1))

    def predict(self, s0, s1):
        estim_s0 = self.q0.quantify(s0)
        estim_s1 = self.q1.quantify(s1)
        estim_gap = independence_gap(estim_s0[1], estim_s1[1])
        return self.reg.predict(estim_gap.reshape(-1,1))[0]


print('__main__')
X, y, A = adultcsv_loader("datasets/adult.csv", protected_attr="gender")
scaler = sklearn.preprocessing.StandardScaler()
X = scaler.fit_transform(X)
classifier = LogisticRegression(class_weight='balanced')

quantifier = PACC(LogisticRegression())

results_q = []
results_m = []
for run, (D1, D2, D3, AD1) in enumerate(gen_split_data(X, y, A, repetitions=5)):
    f = classifier.fit(*D1.Xy)
    D2_y1, D2_y0 = classify(f, D2)
    D3_y1, D3_y0 = classify(f, D3)

    quantifier.fit(D2_y1)
    estim_s1A1 = quantifier.quantify(D3_y1.instances)[1]
    true_s1A1 = D3_y1.prevalence()[1]

    quantifier.fit(D2_y0)
    estim_s0A1 = quantifier.quantify(D3_y0.instances)[1]
    true_s0A1 = D3_y0.prevalence()[1]

    true_gap = independence_gap(true_s0A1, true_s1A1)
    estim_gap = independence_gap(estim_s0A1, estim_s1A1)
    error = estim_gap - true_gap
    results_q.append(error)

    ie = IndependenceEstimator(quantifier)
    ie.fit(D2_y0, D2_y1)
    estim_gap2 = ie.predict(D3_y0.instances, D3_y1.instances)
    error2 = estim_gap2 - true_gap
    results_m.append(error2)

    print(f'Q error {error:.4f}')
    print(f'Q error2 {error2:.4f}')

results_q = np.asarray(results_q)
results_m = np.asarray(results_m)

print(f'MAE(Q)={np.abs(results_q).mean():.5f}')
print(f'MAE(M)={np.abs(results_m).mean():.5f}')





