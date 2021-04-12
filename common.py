import itertools
from typing import List

import quapy as qp
import numpy as np
from quapy.data import LabelledCollection
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
from tqdm import tqdm
from enum import Enum
from dataclasses import dataclass

from quapy.method.base import BaseQuantifier


class Protocol(Enum):
    VAR_D1_PREV = 0  # currently unused
    VAR_D1_PREVFLIP = 1  # artificial variations of P(A=y) in D1
    VAR_D2_PREV = 2
    VAR_D2_SIZE = 3
    VAR_D3_PREV = 4
    """
    VAR_D3_PREV applies the artificial sampling protocol to D3_y0 (leaving D3_y1 fixed) and then to D3_y1 (leaving
    D3_y0 fixed).
    """

"""

class Result_(dataclass):
    protocol: Protocols
    dataset: str
    D1_pAeq: np.ndarray
    D1_size: np.ndarray
    D2_s0_size: np.ndarray
    D2_s1_size: np.ndarray
    D3_s0_size: np.ndarray
    D3_s1_size: np.ndarray
    true_D1_A1: np.ndarray
    true_D2_s0_A1: np.ndarray
    true_D2_s1_A1: np.ndarray
    true_D3_s0_A1: np.ndarray
    true_D3_s1_A1: np.ndarray
    estim_D3_s0_A1: np.ndarray
    estim_D3_s1_A1: np.ndarray
"""

class Result:
    def __init__(self,
                 protocol,
                 true_D3_s0_A1,
                 true_D3_s1_A1,
                 estim_D3_s0_A1,
                 estim_D3_s1_A1,
                 D1_pAeq=None,
                 D1_size=None,
                 D2_s0_size=None,
                 D2_s1_size=None,
                 D3_s0_size=None,
                 D3_s1_size=None,
                 true_D1_A1=None,
                 true_D2_s0_A1=None,
                 true_D2_s1_A1=None
    ):
        self.protocol = protocol
        self.true_D3_s0_A1 = true_D3_s0_A1
        self.true_D3_s1_A1 = true_D3_s1_A1
        self.estim_D3_s0_A1 = estim_D3_s0_A1
        self.estim_D3_s1_A1 = estim_D3_s1_A1

    def independence_gap(self):
        return independence_gap(self.true_D3_s0_A1, self.true_D3_s1_A1, self.estim_D3_s0_A1, self.estim_D3_s1_A1)

    def independence_abs_error(self):
        return independence_abs_error(self.true_D3_s0_A1, self.true_D3_s1_A1, self.estim_D3_s0_A1, self.estim_D3_s1_A1)

    def independence_sqr_error(self):
        return independence_sqr_error(self.true_D3_s0_A1, self.true_D3_s1_A1, self.estim_D3_s0_A1, self.estim_D3_s1_A1)

    def from_slice(self, s:slice):
        return Result(self.protocol,
                      self.true_D3_s0_A1[s],
                      self.true_D3_s1_A1[s],
                      self.estim_D3_s0_A1[s],
                      self.estim_D3_s1_A1[s])
    @classmethod
    def concatenate(cls, results:List["Result"]):
        assert len(set(r.protocol for r in results)) == 1, 'merging more than one protocol'
        return Result(
            protocol=results[0].protocol,
            true_D3_s0_A1=np.concatenate([r.true_D3_s0_A1 for r in results]),
            true_D3_s1_A1=np.concatenate([r.true_D3_s1_A1 for r in results]),
            estim_D3_s0_A1=np.concatenate([r.estim_D3_s0_A1 for r in results]),
            estim_D3_s1_A1=np.concatenate([r.estim_D3_s1_A1 for r in results]),
        )

    def __len__(self):
        return len(self.true_D3_s0_A1)


def uniform_sampling_with_indices(D: LabelledCollection, size):
    unif_index = D.uniform_sampling_index(size)
    return D.sampling_from_index(unif_index), unif_index


def natural_sampling_generator_varsize(D: LabelledCollection, sample_sizes, repeats):
    for _ in range(repeats):
        for sample_size in sample_sizes:
            yield uniform_sampling_with_indices(D, sample_size)

"""
def get_prevalences(D2_y0, D2_y1, D3_y0, D3_y1):
    return {
         "prev_A1_D2_y1": D2_y1.labels.mean(),
         "prev_A1_D2_y0": D2_y0.labels.mean(),
         "prev_A1_D3_y0": D3_y0.labels.mean(),
         "perc_pred_y1_D2": len(D2_y1) / len(D2_y0 + D2_y1),
         "perc_pred_y1_D3": len(D3_y1) / len(D3_y0 + D3_y1)
    }
"""


def as_labelled_collections(D1, D2, D3):
    def gety(yAlabels):
        return np.logical_or(yAlabels == 1, yAlabels == 3).astype(int)

    def getA(yAlabels):
        return np.logical_or(yAlabels == 2, yAlabels == 3).astype(int)

    # recodify targets in D1 for y, and D2,D3 for A
    AD1 = getA(D1.labels)
    D1 = LabelledCollection(D1.instances, gety(D1.labels), n_classes=2)
    D2 = LabelledCollection(D2.instances, getA(D2.labels), n_classes=2)
    D3 = LabelledCollection(D3.instances, getA(D3.labels), n_classes=2)

    return D1, D2, D3, AD1


def gen_split_data(X, y, A, repetitions=2):
    def joint_code_yA(y, A):
        return y + 2 * A

    # jointly codify (y,A) as 0,0->0; 1,0->1; 0,1->2; 1,1->3 to stratify
    yA = joint_code_yA(y,A)

    data = LabelledCollection(X, yA)
    seeds = np.arange(repetitions)
    for seed in seeds:
        P1, P2_3 = data.split_stratified(train_prop=1/3, random_state=seed)
        P2, P3 = P2_3.split_stratified(train_prop=1/2, random_state=seed)
        for D1, D2, D3 in itertools.permutations([P1, P2, P3]):
            yield as_labelled_collections(D1, D2, D3)


def classify(classifier:BaseEstimator, data:LabelledCollection):
    y_hat = classifier.predict(data.instances)
    pred_positives = LabelledCollection(data.instances[y_hat == 1], data.labels[y_hat == 1], n_classes=data.n_classes)
    pred_negatives = LabelledCollection(data.instances[y_hat == 0], data.labels[y_hat == 0], n_classes=data.n_classes)
    return pred_positives, pred_negatives


"""
def compute_bias_error(true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1):

    def cross_diff(M0pos, M1pos):
        M0pos = np.asarray(M0pos)
        M1pos = np.asarray(M1pos)
        return M0pos.reshape(-1, 1) - M1pos

    true_diff = cross_diff(true_M0_A1, true_M1_A1)
    estim_diff = cross_diff(estim_M0_A1, estim_M1_A1)
    bias = estim_diff - true_diff
    error = np.abs(bias)
    return bias.mean(), bias.std(), error.mean(), error.std()

"""

"""
def eval_prevalence_variations_D1(D1, D2, D3, AD1, classifier, quantifier, sample_size=500, nprevs=101, nreps=2):
    idcs_00_11 = [i for i, (y,a) in enumerate(zip(D1.labels, AD1)) if (a == 0 and y == 0) or (a == 1 and y == 1)]
    idcs_01_10 = [i for i, (y,a) in enumerate(zip(D1.labels, AD1)) if (a == 0 and y == 1) or (a == 1 and y == 0)]
    assert len(idcs_00_11) + len(idcs_01_10) == len(D1.labels)
    true_M0_A1, true_M1_A1 = [], []
    estim_M0_A1, estim_M1_A1 = [], []
    p_Aeqy = []

    prevs = np.linspace(0., 1., nprevs, endpoint=True)
    for prev in prevs:
        for _ in range(nreps):
            sample_size_00_11 = int(round(prev*sample_size))
            sample_size_01_10 = sample_size - sample_size_00_11
            idcs_sample = list(np.random.permutation(idcs_00_11)[:sample_size_00_11]) + \
                          list(np.random.permutation(idcs_01_10)[:sample_size_01_10])
            sample_D1 = LabelledCollection([D1.instances[i] for i in idcs_sample], [D1.labels[i] for i in idcs_sample])

            assert sample_D1.prevalence().prod() != 0
            f = classifier.fit(*sample_D1.Xy)

            D2_y1, D2_y0 = classify(f, D2)
            D3_y1, D3_y0 = classify(f, D3)
            dict_prev = get_prevalences(D2_y0, D2_y1, D3_y0, D3_y1)

            M1 = deepcopy(quantifier).fit(D2_y1)
            M0 = deepcopy(quantifier).fit(D2_y0)

            estim_M1_A1.append(M1.quantify(D3_y1.instances)[1])
            estim_M0_A1.append(M0.quantify(D3_y0.instances)[1])

            true_M1_A1.append(D3_y1.prevalence()[1])
            true_M0_A1.append(D3_y0.prevalence()[1])
            p_Aeqy.append(prev)

    return true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, p_Aeqy
"""

def eval_prevalence_variations_D1_flip(D1, D2, D3, AD1, classifier, quantifier, sample_size=500, nprevs=101, nreps=2):
    true_M0_A1, true_M1_A1 = [], []
    estim_M0_A1, estim_M1_A1 = [], []
    p_Aeqy = []

    prevs = np.linspace(0., 1., nprevs, endpoint=True)

    def flip_labels(Y, A, p_Aeqy):
        idcs_00 = [i for i, (y, a) in enumerate(zip(Y, A)) if (a == 0 and y == 0)]
        idcs_01 = [i for i, (y, a) in enumerate(zip(Y, A)) if (a == 0 and y == 1)]
        idcs_10 = [i for i, (y, a) in enumerate(zip(Y, A)) if (a == 1 and y == 0)]
        idcs_11 = [i for i, (y, a) in enumerate(zip(Y, A)) if (a == 1 and y == 1)]
        assert len(idcs_00) + len(idcs_01) + len(idcs_10) + len(idcs_11) == len(Y) == len(A)
        p_Aeqy_given_A0 = len(idcs_00) / (len(idcs_00) + len(idcs_01))
        p_Aeqy_given_A1 = len(idcs_11) / (len(idcs_10) + len(idcs_11))
        if p_Aeqy_given_A0 > p_Aeqy:
            num_flip = int(round((1-p_Aeqy)*(len(idcs_00) + len(idcs_01)))) - len(idcs_01)
            Y = [1 if i in idcs_00[:num_flip] else y for i, y in enumerate(Y)]
        else:
            num_flip = int(round(p_Aeqy*(len(idcs_00) + len(idcs_01)))) - len(idcs_00)
            Y = [0 if i in idcs_01[:num_flip] else y for i, y in enumerate(Y)]
        assert num_flip >= 0
        if p_Aeqy_given_A1 > p_Aeqy:
            num_flip = int(round((1-p_Aeqy)*(len(idcs_10) + len(idcs_11)))) - len(idcs_10)
            Y = [0 if i in idcs_11[:num_flip] else y for i, y in enumerate(Y)]
        else:
            num_flip = int(round(p_Aeqy*(len(idcs_10) + len(idcs_11)))) - len(idcs_11)
            Y = [1 if i in idcs_10[:num_flip] else y for i, y in enumerate(Y)]
        assert num_flip >= 0
        return Y

    for prev in prevs:
        for _ in range(nreps):
            idcs_sample = np.random.permutation(range(len(D1)))[:sample_size]
            X_sample = D1.instances[idcs_sample]
            y_sample = D1.labels[idcs_sample]
            A_sample = AD1[idcs_sample]
            y_sample = flip_labels(y_sample, A_sample, prev)
            sample_D1 = LabelledCollection(X_sample, y_sample)

            assert sample_D1.prevalence().prod() != 0
            f = classifier.fit(*sample_D1.Xy)

            D2_y1, D2_y0 = classify(f, D2)
            D3_y1, D3_y0 = classify(f, D3)
            dict_prev = get_prevalences(D2_y0, D2_y1, D3_y0, D3_y1)

            M1 = deepcopy(quantifier).fit(D2_y1)
            M0 = deepcopy(quantifier).fit(D2_y0)

            estim_M1_A1.append(M1.quantify(D3_y1.instances)[1])
            estim_M0_A1.append(M0.quantify(D3_y0.instances)[1])

            true_M1_A1.append(D3_y1.prevalence()[1])
            true_M0_A1.append(D3_y0.prevalence()[1])
            p_Aeqy.append(prev)

    return true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, p_Aeqy


def eval_prevalence_variations_D2(D1, D2, D3, classifier, quantifier, sample_size=1000, nprevs=101, nreps=2):
    f = classifier.fit(*D1.Xy)
    D2_y1, D2_y0 = classify(f, D2)
    D3_y1, D3_y0 = classify(f, D3)
    dict_prev = get_prevalences(D2_y0, D2_y1, D3_y0, D3_y1)

    def vary_and_test(quantifier, D2split, D3split, nreps):
        M = deepcopy(quantifier)
        estim_M_A1 = []
        prev_D2_sample = []
        for sample_D2 in tqdm(D2split.artificial_sampling_generator(sample_size=sample_size, n_prevalences=nprevs, repeats=nreps),
                              total=nprevs):
            if sample_D2.prevalence().prod() == 0: continue
            M.fit(sample_D2)
            estim_M_A1.append(M.quantify(D3split.instances)[1])
            prev_D2_sample.append(sample_D2.prevalence()[1])
        true_M_A1 = [D3split.prevalence()[1]] * len(estim_M_A1)
        return true_M_A1, estim_M_A1, prev_D2_sample
    true_M1_A1, estim_M1_A1, prev_D2_y1 = vary_and_test(quantifier, D2_y1, D3_y1, nreps)
    true_M0_A1, estim_M0_A1, prev_D2_y0 = vary_and_test(quantifier, D2_y0, D3_y0, nreps)

    return true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, prev_D2_y0, prev_D2_y1,  dict_prev

def eval_size_variations_D2(D1, D2, D3, classifier, quantifier, nreps=10, sample_sizes=None):
    if sample_sizes is None:
        min_size = 1000
        sample_sizes = [int(round(val)) for val in np.geomspace(min_size, len(D2.labels), num=5)]
    f = classifier.fit(*D1.Xy)
    D2_y_hat = classifier.predict(D2.instances)
    D3_y1, D3_y0 = classify(f, D3)

    def vary_and_test(quantifier, D2, D2_y_hat, D3_y0, D3_y1, sample_sizes, nreps):
        M0 = deepcopy(quantifier)
        M1 = deepcopy(quantifier)
        estim_M0_A1 = []
        estim_M1_A1 = []
        size_D2 = []
        for sample_D2, sample_idcs in tqdm(natural_sampling_generator_varsize(D2, sample_sizes, nreps),
                              desc='training quantifiers at size variations in D2'):
            assert sample_D2.prevalence().prod() != 0
            sample_D2_y_hat = D2_y_hat[sample_idcs]
            sample_D2_y0 = LabelledCollection(sample_D2.instances[sample_D2_y_hat == 0], sample_D2.labels[sample_D2_y_hat == 0], n_classes=sample_D2.n_classes)
            sample_D2_y1 = LabelledCollection(sample_D2.instances[sample_D2_y_hat == 1], sample_D2.labels[sample_D2_y_hat == 1], n_classes=sample_D2.n_classes)
            M0.fit(sample_D2_y0)
            estim_M0_A1.append(M0.quantify(D3_y0.instances)[1])
            M1.fit(sample_D2_y1)
            estim_M1_A1.append(M1.quantify(D3_y1.instances)[1])
            size_D2.append(len(sample_D2))
        true_M0_A1 = [D3_y0.prevalence()[1]] * len(estim_M0_A1)
        true_M1_A1 = [D3_y1.prevalence()[1]] * len(estim_M1_A1)
        return true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, size_D2

    return vary_and_test(quantifier, D2, D2_y_hat, D3_y0, D3_y1, sample_sizes, nreps)


def eval_prevalence_variations_D3(D1, D2, D3, classifier, quantifier, sample_size=500, nprevs=101, nreps=2):
    f = classifier.fit(*D1.Xy)

    D2_y1, D2_y0 = classify(f, D2)
    D3_y1, D3_y0 = classify(f, D3)

    M0 = deepcopy(quantifier).fit(D2_y0)
    M1 = deepcopy(quantifier).fit(D2_y1)

    true_D3_y0_val = D3_y0.prevalence()[1]
    true_D3_y1_val = D3_y1.prevalence()[1]

    estim_D3_y0_val = M0.quantify(D3_y0.instances)[1]
    estim_D3_y1_val = M1.quantify(D3_y1.instances)[1]

    def estimate_prevalences(quantSa:BaseQuantifier, Sa: LabelledCollection, truePrevSb:float, estimPrevSb:float):
        truePrevsSa, estimPrevsSa = qp.evaluation.artificial_sampling_prediction(
            quantSa,  # the quantification method
            Sa,  # the test set on which the method will be evaluated
            sample_size=sample_size,  # indicates the size of samples to be drawn
            n_prevpoints=nprevs,  # how many prevalence points will be extracted from the interval [0, 1] for each category
            n_repetitions=nreps,  # number of times each prevalence will be used to generate a test sample
            n_jobs=-1,  # indicates the number of parallel workers (-1 indicates, as in sklearn, all CPUs)
            random_seed=42,  # setting a random seed allows to replicate the test samples across runs
            verbose=True  # set to True to show some standard-line outputs
        )
        truePrevsSa = truePrevsSa[:,1]
        estimPrevsSa = estimPrevsSa[:,1]
        truePrevSb = np.full_like(truePrevsSa, fill_value=truePrevSb)
        estimPrevSb = np.full_like(estimPrevsSa, fill_value=estimPrevSb)
        return truePrevsSa, estimPrevsSa, truePrevSb, estimPrevSb

    true_s0var, estim_s0var, true_s1fix, estim_s1fix = estimate_prevalences(M0, D3_y0, true_D3_y1_val, estim_D3_y1_val)
    true_s1var, estim_s1var, true_s0fix, estim_s0fix = estimate_prevalences(M1, D3_y1, true_D3_y0_val, estim_D3_y0_val)

    # stack the blocks
    true_D3_s0_A1 = np.concatenate([true_s0var, true_s0fix])
    true_D3_s1_A1 = np.concatenate([true_s1fix, true_s1var])
    estim_D3_s0_A1 = np.concatenate([estim_s0var, estim_s0fix])
    estim_D3_s1_A1 = np.concatenate([estim_s1fix, estim_s1var])

    result = Result(Protocol.VAR_D3_PREV, true_D3_s0_A1, true_D3_s1_A1, estim_D3_s0_A1, estim_D3_s1_A1)

    # info for plots
    result.var_s0 = slice(0, len(true_s0var))  # slice containing results with variations in s0 (s1 fix)
    result.var_s1 = slice(len(true_s1fix), len(true_s1fix) + len(true_s1var))  # slice for variations in s1 (s0 fix)
    result.D2_s0_prev = D2_y0.prevalence()[1]
    result.D2_s1_prev = D2_y1.prevalence()[1]

    return result


def independence(s0_A1, s1_A1):
    return s0_A1 - s1_A1


def independence_gap(true_s0_A1, true_s1_A1, estim_s0_A1, estim_s1_A1):
    """
    Computes the gap between the estimated independence and the true independence.
    Positive (negative) values thus represent a tendency to overestimate (underestimate) the true value.
    """
    true_inds = independence(true_s0_A1, true_s1_A1)
    estim_inds = independence(estim_s0_A1, estim_s1_A1)
    gaps = estim_inds - true_inds
    return gaps


def independence_abs_error(true_s0_A1, true_s1_A1, estim_s0_A1, estim_s1_A1):
    """
    Computes the error (absolute value of the gaps) between the estimated independence and the true independence.
    Errors are always non-zero.
    """
    errors = np.abs(independence_gap(true_s0_A1, true_s1_A1, estim_s0_A1, estim_s1_A1))
    return errors


def independence_sqr_error(true_s0_A1, true_s1_A1, estim_s0_A1, estim_s1_A1):
    """
    Computes the error (absolute value of the gaps) between the estimated independence and the true independence.
    Errors are always non-zero.
    """
    errors = independence_gap(true_s0_A1, true_s1_A1, estim_s0_A1, estim_s1_A1) ** 2
    return errors
