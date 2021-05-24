from copy import deepcopy
from sklearn.svm import LinearSVR
import numpy as np
import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.base import BaseQuantifier
from abc import ABC, abstractmethod
#from common import independence_gap



def independence_gap(s0_A1, s1_A1):
    return s0_A1 - s1_A1


class IndependenceGapEstimator(ABC):
    @abstractmethod
    def fit(self, s0: LabelledCollection, s1: LabelledCollection): ...

    @abstractmethod
    def predict(self, s0, s1): ...


class DummyIGE(IndependenceGapEstimator):

    def __init__(self):
        pass

    def fit(self, s0: LabelledCollection, s1: LabelledCollection):
        pass

    def predict(self, s0, s1):
        return 0



def _gen_artificial_samples(q0, q1, s0te, s1te, sample_size, n_prevs, n_repetitions):
    true_s0, estim_s0 = qp.evaluation.artificial_sampling_prediction(
        q0, s0te, sample_size=sample_size, n_prevpoints=n_prevs, n_repetitions=n_repetitions, n_jobs=-1)
    true_s1, estim_s1 = qp.evaluation.artificial_sampling_prediction(
        q1, s1te, sample_size=sample_size, n_prevpoints=n_prevs, n_repetitions=n_repetitions, n_jobs=-1)

    idx = np.random.permutation(len(true_s0))
    true_s0 = true_s0[idx]
    estim_s0 = estim_s0[idx]

    idx = np.random.permutation(len(true_s1))
    true_s1 = true_s1[idx]
    estim_s1 = estim_s1[idx]

    return true_s0[:,1], true_s1[:,1], estim_s0[:,1], estim_s1[:,1]


def _gen_natural_samples(q0, q1, s0te, s1te, sample_size, n_repetitions):
    true_s0, estim_s0 = qp.evaluation.natural_sampling_prediction(
        q0, s0te, sample_size=sample_size, n_repetitions=n_repetitions, n_jobs=-1)
    true_s1, estim_s1 = qp.evaluation.natural_sampling_prediction(
        q1, s1te, sample_size=sample_size, n_repetitions=n_repetitions, n_jobs=-1)
    return true_s0[:, 1], true_s1[:, 1], estim_s0[:, 1], estim_s1[:, 1]


class NaturalSamplingAdjustment(IndependenceGapEstimator):

    def __init__(self, quantifier:BaseQuantifier, sample_size=500, n_repetitions=1000):
        self.q0 = deepcopy(quantifier)
        self.q1 = deepcopy(quantifier)
        self.sample_size = sample_size
        self.n_repetitions = n_repetitions

    def fit(self, s0:LabelledCollection, s1:LabelledCollection):
        s0tr, s0te = s0.split_stratified()
        s1tr, s1te = s1.split_stratified()

        self.q0.fit(s0tr)
        self.q1.fit(s1tr)

        true_s0, true_s1, estim_s0, estim_s1 = _gen_natural_samples(
            self.q0, self.q1, s0te, s1te, self.sample_size, self.n_repetitions)

        true_gap = independence_gap(true_s0, true_s1)
        estim_gap = independence_gap(estim_s0, estim_s1)

        X = np.vstack([estim_gap, estim_s0, estim_s1]).T
        y = true_gap.reshape(-1,1)

        self.reg = LinearSVR().fit(X, y)

    def predict(self, s0, s1):
        estim_s0 = self.q0.quantify(s0)
        estim_s1 = self.q1.quantify(s1)
        estim_gap = independence_gap(estim_s0[1], estim_s1[1])
        X = np.asarray([estim_gap, estim_s0[1], estim_s1[1]]).reshape(1,-1)
        return self.reg.predict(X)[0]


class ArtificialSamplingAdjustment(IndependenceGapEstimator):

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

        true_s0, true_s1, estim_s0, estim_s1 = _gen_artificial_samples(
            self.q0, self.q1, s0te, s1te, self.sample_size, self.n_prevpoints, self.n_repetitions // self.n_prevpoints)

        true_gap = independence_gap(true_s0, true_s1)
        estim_gap = independence_gap(estim_s0, estim_s1)

        X = np.vstack([estim_gap, estim_s0, estim_s1]).T
        y = true_gap.reshape(-1,1)

        self.reg = LinearSVR().fit(X, y)

    def predict(self, s0, s1):
        estim_s0 = self.q0.quantify(s0)
        estim_s1 = self.q1.quantify(s1)
        estim_gap = independence_gap(estim_s0[1], estim_s1[1])
        X = np.asarray([estim_gap, estim_s0[1], estim_s1[1]]).reshape(1,-1)
        return self.reg.predict(X)[0]


class MixSamplingAdjustment(IndependenceGapEstimator):

    def __init__(self, quantifier:BaseQuantifier, sample_size=500, n_prevpoints=21, n_repetitions=500, refit=True, regressor=LinearSVR()):
        self.q0 = deepcopy(quantifier)
        self.q1 = deepcopy(quantifier)
        self.sample_size = sample_size
        self.n_prevpoints = n_prevpoints
        self.n_repetitions = n_repetitions
        self.refit = refit
        self.r = regressor

    def fit(self, s0:LabelledCollection, s1:LabelledCollection):
        s0tr, s0te = s0.split_stratified()
        s1tr, s1te = s1.split_stratified()

        self.q0.fit(s0tr)
        self.q1.fit(s1tr)

        true_s0_art, true_s1_art, estim_s0_art, estim_s1_art = _gen_artificial_samples(
            self.q0, self.q1, s0te, s1te, self.sample_size, self.n_prevpoints, self.n_repetitions // self.n_prevpoints)
        true_s0_nat, true_s1_nat, estim_s0_nat, estim_s1_nat = _gen_natural_samples(
            self.q0, self.q1, s0te, s1te, self.sample_size, self.n_repetitions)

        if self.refit:
            self.q0.fit(s0)
            self.q1.fit(s1)

        true_s0 = np.concatenate([true_s0_art, true_s0_nat])
        true_s1 = np.concatenate([true_s1_art, true_s1_nat])
        estim_s0 = np.concatenate([estim_s0_art, estim_s0_nat])
        estim_s1 = np.concatenate([estim_s1_art, estim_s1_nat])

        true_gap = independence_gap(true_s0, true_s1)
        estim_gap = independence_gap(estim_s0, estim_s1)

        X = np.vstack([estim_gap, estim_s0, estim_s1]).T
        y = true_gap.reshape(-1,1)

        self.r.fit(X, y)

    def predict(self, s0, s1):
        estim_s0 = self.q0.quantify(s0)
        estim_s1 = self.q1.quantify(s1)
        estim_gap = independence_gap(estim_s0[1], estim_s1[1])
        X = np.asarray([estim_gap, estim_s0[1], estim_s1[1]]).reshape(1,-1)
        return np.clip(self.r.predict(X)[0],-1,1)




