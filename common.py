import itertools
import pandas as pd
import quapy as qp
import numpy as np

from method import IndependenceGapEstimator
from quapy.data import LabelledCollection
from sklearn.base import BaseEstimator
from copy import deepcopy
from tqdm import tqdm
from enum import Enum
from quapy.method.base import BaseQuantifier


def independence_gap(s0_A1, s1_A1):
    return s0_A1 - s1_A1


class Protocols(Enum):
    VAR_D1_PREV = 0  # currently unused
    VAR_D1_PREVFLIP = 1  # artificial variations of P(A=y) in D1
    VAR_D2_PREV = 2
    VAR_D2_SIZE = 3
    VAR_D3_PREV = 4
    """
    VAR_D3_PREV applies the artificial sampling protocol to D3_y0 (leaving D3_y1 fixed) and then to D3_y1 (leaving
    D3_y0 fixed).
    """


class Result:
    def __init__(self, df: pd.DataFrame):
        self.data = df

    @classmethod
    def with_columns(cls, protocol, true_D3_s0_A1, true_D3_s1_A1, estim_D3_s0_A1, estim_D3_s1_A1, **kwargs):
        columns = {
            'protocol': protocol,
            'trueD3s0A1': true_D3_s0_A1,
            'trueD3s1A1': true_D3_s1_A1,
            'estimD3s0A1': estim_D3_s0_A1,
            'estimD3s1A1': estim_D3_s1_A1
        }
        columns.update(kwargs)
        return Result(pd.DataFrame(columns))

    def independence_gap(self):
        d = self.data
        return independence_gap(d['trueD3s0A1'], d['trueD3s1A1'])

    def estim_independence_gap(self):
        d = self.data
        if 'direct' in d and not d['direct'].isnull().values.any():
            return d['direct']
            #print('DIRECT IS PRESENT', d['Q_name'].unique())
        else:
            #print('--------', d['Q_name'].unique())
            return independence_gap(d['estimD3s0A1'], d['estimD3s1A1'])

    def independence_signed_error(self):
        true_inds = self.independence_gap()
        estim_inds = self.estim_independence_gap()
        errors = estim_inds - true_inds
        return errors
        #return estim_signed_error(d['trueD3s0A1'], d['trueD3s1A1'], d['estimD3s0A1'], d['estimD3s1A1'])

    def independence_abs_error(self):
        """
        Computes the error (absolute value of the gaps) between the estimated independence and the true independence.
        Errors are always non-zero.
        """
        return np.abs(self.independence_signed_error())
        #return independence_abs_error(d['trueD3s0A1'], d['trueD3s1A1'], d['estimD3s0A1'], d['estimD3s1A1'])

    def D3s0_abs_error(self):
        d = self.data
        return qp.error.ae(d['trueD3s0A1'].values.reshape(-1,1), d['estimD3s0A1'].values.reshape(-1,1))

    def D3s1_abs_error(self):
        d = self.data
        return qp.error.ae(d['trueD3s1A1'].values.reshape(-1,1), d['estimD3s1A1'].values.reshape(-1,1))

    def independence_sqr_error(self):
        #d = self.data
        #return independence_sqr_error(d['trueD3s0A1'], d['trueD3s1A1'], d['estimD3s0A1'], d['estimD3s1A1'])

        """
        Computes the error (absolute value of the gaps) between the estimated independence and the true independence.
        Errors are always non-zero.
        """
        return self.independence_signed_error() ** 2

    def set(self, colname, value):
        self.data[colname]=value

    def get(self, colname):
        return self.data[colname]

    @classmethod
    def concat(cls, results):
        return Result(pd.concat([r.data for r in results]))

    def __len__(self):
        return len(self.data)

    def save(self, path):
        self.data.to_pickle(path)

    @classmethod
    def load(cls, path):
        return Result(df=pd.read_pickle(path))

    def select_protocol(self, protocol:Protocols):
        return self.filter('protocol', protocol)

    def filter(self, attr, val):
        df = self.data
        return Result(df.loc[df[attr] == val])


def uniform_sampling_with_indices(D: LabelledCollection, size):
    unif_index = D.uniform_sampling_index(size)
    return D.sampling_from_index(unif_index), unif_index


def natural_sampling_generator_varsize(D: LabelledCollection, sample_sizes, repeats):
    for _ in range(repeats):
        for sample_size in sample_sizes:
            yield uniform_sampling_with_indices(D, sample_size)


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


def at_least_npos(data:LabelledCollection, npos:int):
    return data.labels.sum()>=npos


def at_least_nneg(data:LabelledCollection, nneg:int):
    return len(data.labels) - data.labels.sum() >= nneg


def eval_prevalence_variations_D1(D1, D2, D3, AD1, classifier, model, sample_size=500, nprevs=101, nreps=2):
    if isinstance(model, BaseQuantifier):
        func = eval_prevalence_variations_D1_quant
    elif isinstance(model, IndependenceGapEstimator):
        func = eval_prevalence_variations_D1_estim
    return func(D1, D2, D3, AD1, classifier, model, sample_size, nprevs, nreps)


def __var_D1_generator(D1, D2, D3, AD1, classifier, sample_size, nprevs, nreps):
    D1indexesAeqY = LabelledCollection(instances=np.arange(len(D1)), labels=D1.labels == AD1)
    steps = qp.functional.num_prevalence_combinations(nprevs, n_classes=2, n_repeats=nreps)
    for idcs in tqdm(D1indexesAeqY.artificial_sampling_index_generator(sample_size, nprevs, nreps), total=steps):
        sample_D1 = D1.sampling_from_index(idcs)
        if sample_D1.prevalence().prod() == 0:
            continue
        f = classifier.fit(*sample_D1.Xy)

        D2_y1, D2_y0 = classify(f, D2)
        D3_y1, D3_y0 = classify(f, D3)

        if not at_least_npos(D2_y1, npos=5): continue
        if not at_least_npos(D2_y0, npos=5): continue
        if not at_least_nneg(D2_y1, nneg=5): continue
        if not at_least_nneg(D2_y0, nneg=5): continue

        prevAeqY = D1indexesAeqY.labels[idcs].mean()

        yield D2_y0, D2_y1, D3_y0, D3_y1, prevAeqY


def eval_prevalence_variations_D1_quant(D1, D2, D3, AD1, classifier, quantifier, sample_size=500, nprevs=101, nreps=2):
    true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, p_Aeqy = [], [], [], [], []

    for D2_y0, D2_y1, D3_y0, D3_y1, prevAeqY in __var_D1_generator(D1, D2, D3, AD1, classifier, sample_size, nprevs, nreps):
        M1 = deepcopy(quantifier).fit(D2_y1)
        M0 = deepcopy(quantifier).fit(D2_y0)
        estim_M1_A1.append(M1.quantify(D3_y1.instances)[1])
        estim_M0_A1.append(M0.quantify(D3_y0.instances)[1])
        true_M1_A1.append(D3_y1.prevalence()[1])
        true_M0_A1.append(D3_y0.prevalence()[1])
        p_Aeqy.append(prevAeqY)

    return Result.with_columns(Protocols.VAR_D1_PREV, true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, p_Aeqy=p_Aeqy)


def eval_prevalence_variations_D1_estim(D1, D2, D3, AD1, classifier, estimator:IndependenceGapEstimator, sample_size=500, nprevs=101, nreps=2):
    true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, p_Aeqy = [], [], [], [], []
    direct_estimation = []

    for D2_y0, D2_y1, D3_y0, D3_y1, prevAeqY in __var_D1_generator(D1, D2, D3, AD1, classifier, sample_size, nprevs, nreps):
        estimator.fit(D2_y0, D2_y1)
        ig_hat = estimator.predict(D3_y0.instances, D3_y1.instances)
        direct_estimation.append(ig_hat)
        estim_M1_A1.append(0)
        estim_M0_A1.append(0)
        true_M1_A1.append(D3_y1.prevalence()[1])
        true_M0_A1.append(D3_y0.prevalence()[1])
        p_Aeqy.append(prevAeqY)

    return Result.with_columns(Protocols.VAR_D1_PREV, true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1,
                               p_Aeqy=p_Aeqy, direct=direct_estimation)


def __var_D1_flip_generator(D1, D2, D3, AD1, classifier, sample_size, nprevs, nreps):
    prevs = np.linspace(0., 1., nprevs, endpoint=True)

    def flip_labels(Y, A, p_Aeqy):
        Y = np.copy(Y)
        idcs_00 = np.asarray([i for i, (y, a) in enumerate(zip(Y, A)) if (a == 0 and y == 0)])
        idcs_01 = np.asarray([i for i, (y, a) in enumerate(zip(Y, A)) if (a == 0 and y == 1)])
        idcs_10 = np.asarray([i for i, (y, a) in enumerate(zip(Y, A)) if (a == 1 and y == 0)])
        idcs_11 = np.asarray([i for i, (y, a) in enumerate(zip(Y, A)) if (a == 1 and y == 1)])
        assert len(idcs_00) + len(idcs_01) + len(idcs_10) + len(idcs_11) == len(Y) == len(A)

        p_Aeqy_given_A0 = len(idcs_00) / (len(idcs_00) + len(idcs_01))
        p_Aeqy_given_A1 = len(idcs_11) / (len(idcs_10) + len(idcs_11))
        if p_Aeqy_given_A0 > p_Aeqy:
            num_flip = int(round((1-p_Aeqy)*(len(idcs_00) + len(idcs_01)))) - len(idcs_01)
            if num_flip > 0:
                Y[idcs_00[:num_flip]] = 1
        else:
            num_flip = int(round(p_Aeqy*(len(idcs_00) + len(idcs_01)))) - len(idcs_00)
            if num_flip>0:
                Y[idcs_01[:num_flip]] = 0
        assert num_flip >= 0
        if p_Aeqy_given_A1 > p_Aeqy:
            num_flip = int(round((1-p_Aeqy)*(len(idcs_10) + len(idcs_11)))) - len(idcs_10)
            if num_flip > 0:
                Y[idcs_11[:num_flip]] = 0
        else:
            num_flip = int(round(p_Aeqy*(len(idcs_10) + len(idcs_11)))) - len(idcs_11)
            if num_flip > 0:
                Y[idcs_10[:num_flip]] = 1
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

            if any(D2_y1.prevalence()*len(D2_y1) <=20) or any(D2_y0.prevalence()*len(D2_y0) <=20):
                with open("flag_prev", 'at') as fo:
                    fo.write(f"D2_y1: {D2_y1.prevalence()*len(D2_y1)}, D2_y0: {D2_y0.prevalence()*len(D2_y0)}")

            if not at_least_npos(D2_y1, npos=5): continue
            if not at_least_npos(D2_y0, npos=5): continue
            if not at_least_nneg(D2_y1, nneg=5): continue
            if not at_least_nneg(D2_y0, nneg=5): continue

            yield D2_y0, D2_y1, D3_y0, D3_y1, prev


def eval_prevalence_variations_D1_flip(D1, D2, D3, AD1, classifier, model, sample_size=500, nprevs=101, nreps=2):
    if isinstance(model, BaseQuantifier):
        func = eval_prevalence_variations_D1_flip_quant
    elif isinstance(model, IndependenceGapEstimator):
        func = eval_prevalence_variations_D1_flip_estim
    return func(D1, D2, D3, AD1, classifier, model, sample_size, nprevs, nreps)


def eval_prevalence_variations_D1_flip_quant(D1, D2, D3, AD1, classifier, quantifier, sample_size=500, nprevs=101, nreps=2):
    true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, p_Aeqy = [], [], [], [], []

    for D2_y0, D2_y1, D3_y0, D3_y1, prev in __var_D1_flip_generator(D1, D2, D3, AD1, classifier, sample_size, nprevs, nreps):
        M1 = deepcopy(quantifier).fit(D2_y1)
        M0 = deepcopy(quantifier).fit(D2_y0)
        estim_M1_A1.append(M1.quantify(D3_y1.instances)[1])
        estim_M0_A1.append(M0.quantify(D3_y0.instances)[1])
        true_M1_A1.append(D3_y1.prevalence()[1])
        true_M0_A1.append(D3_y0.prevalence()[1])
        p_Aeqy.append(prev)

    return Result.with_columns(Protocols.VAR_D1_PREVFLIP, true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1,
                               p_Aeqy=p_Aeqy)


def eval_prevalence_variations_D1_flip_estim(D1, D2, D3, AD1, classifier, estimator, sample_size=500, nprevs=101, nreps=2):
    true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, p_Aeqy = [], [], [], [], []
    direct_estimation = []

    for D2_y0, D2_y1, D3_y0, D3_y1, prev in __var_D1_flip_generator(D1, D2, D3, AD1, classifier, sample_size, nprevs, nreps):
        estimator.fit(D2_y0, D2_y1)
        ig_hat = estimator.predict(D3_y0.instances, D3_y1.instances)
        direct_estimation.append(ig_hat)
        estim_M1_A1.append(0)
        estim_M0_A1.append(0)
        true_M1_A1.append(D3_y1.prevalence()[1])
        true_M0_A1.append(D3_y0.prevalence()[1])
        p_Aeqy.append(prev)

    return Result.with_columns(Protocols.VAR_D1_PREVFLIP, true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1,
                               p_Aeqy=p_Aeqy, direct=direct_estimation)


def eval_size_variations_D2(D1, D2, D3, classifier, model, nreps=10, sample_sizes=None):
    if isinstance(model, BaseQuantifier):
        func = eval_size_variations_D2_quant
    elif isinstance(model, IndependenceGapEstimator):
        func = eval_size_variations_D2_estim
    return func(D1, D2, D3, classifier, model, nreps, sample_sizes)


def eval_size_variations_D2_quant(D1, D2, D3, classifier, quantifier, nreps=10, sample_sizes=None):
    if sample_sizes is None:
        min_size = 1000
        sample_sizes = [int(round(val)) for val in np.geomspace(min_size, len(D2.labels), num=5)]

    f = classifier.fit(*D1.Xy)
    D2_y_hat = classifier.predict(D2.instances)
    D3_y1, D3_y0 = classify(f, D3)

    def vary_and_test(quantifier, D2, D2_y_hat, D3_y0, D3_y1, sample_sizes, nreps):
        M0 = deepcopy(quantifier)
        M1 = deepcopy(quantifier)
        estim_M0_A1, estim_M1_A1  = [], []
        size_D2 = []
        for sample_D2, sample_idcs in tqdm(natural_sampling_generator_varsize(D2, sample_sizes, nreps),
                                           desc='training quantifiers at size variations in D2'):
            assert sample_D2.prevalence().prod() != 0
            sample_D2_y_hat = D2_y_hat[sample_idcs]
            sample_D2_y0 = sample_D2.sampling_from_index(sample_D2_y_hat == 0)
            sample_D2_y1 = sample_D2.sampling_from_index(sample_D2_y_hat == 1)
            M0.fit(sample_D2_y0)
            estim_M0_A1.append(M0.quantify(D3_y0.instances)[1])
            M1.fit(sample_D2_y1)
            estim_M1_A1.append(M1.quantify(D3_y1.instances)[1])
            size_D2.append(len(sample_D2))

        estim_M0_A1 = np.asarray(estim_M0_A1)
        estim_M1_A1 = np.asarray(estim_M1_A1)
        true_M0_A1 = np.asarray([D3_y0.prevalence()[1]] * len(estim_M0_A1))
        true_M1_A1 = np.asarray([D3_y1.prevalence()[1]] * len(estim_M1_A1))

        return true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, size_D2

    true_D3_s0_A1, estim_D3_s0_A1, true_D3_s1_A1, estim_D3_s1_A1, size_D2 = \
        vary_and_test(quantifier, D2, D2_y_hat, D3_y0, D3_y1, sample_sizes, nreps)

    return Result.with_columns(Protocols.VAR_D2_SIZE, true_D3_s0_A1, true_D3_s1_A1, estim_D3_s0_A1, estim_D3_s1_A1,
                               size_D2=size_D2)


def eval_size_variations_D2_estim(D1, D2, D3, classifier, estimator, nreps=10, sample_sizes=None):
    if sample_sizes is None:
        min_size = 1000
        sample_sizes = [int(round(val)) for val in np.geomspace(min_size, len(D2.labels), num=5)]

    f = classifier.fit(*D1.Xy)
    D2_y_hat = classifier.predict(D2.instances)
    D3_y1, D3_y0 = classify(f, D3)

    def vary_and_test(estimator, D2, D2_y_hat, D3_y0, D3_y1, sample_sizes, nreps):
        estim_M0_A1, estim_M1_A1  = [], []
        size_D2 = []
        direct_estimation = []
        for sample_D2, sample_idcs in tqdm(natural_sampling_generator_varsize(D2, sample_sizes, nreps),
                                           desc='training estimators at size variations in D2'):
            assert sample_D2.prevalence().prod() != 0
            sample_D2_y_hat = D2_y_hat[sample_idcs]
            sample_D2_y0 = sample_D2.sampling_from_index(sample_D2_y_hat == 0)
            sample_D2_y1 = sample_D2.sampling_from_index(sample_D2_y_hat == 1)
            estimator.fit(sample_D2_y0, sample_D2_y1)
            ig_hat = estimator.predict(D3_y0.instances, D3_y1.instances)
            direct_estimation.append(ig_hat)
            estim_M1_A1.append(0)
            estim_M0_A1.append(0)
            size_D2.append(len(sample_D2))

        estim_M0_A1 = np.asarray(estim_M0_A1)
        estim_M1_A1 = np.asarray(estim_M1_A1)
        true_M0_A1 = np.asarray([D3_y0.prevalence()[1]] * len(estim_M0_A1))
        true_M1_A1 = np.asarray([D3_y1.prevalence()[1]] * len(estim_M1_A1))

        return true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, size_D2, direct_estimation

    true_D3_s0_A1, estim_D3_s0_A1, true_D3_s1_A1, estim_D3_s1_A1, size_D2, direct_estimation = \
        vary_and_test(estimator, D2, D2_y_hat, D3_y0, D3_y1, sample_sizes, nreps)

    return Result.with_columns(Protocols.VAR_D2_SIZE, true_D3_s0_A1, true_D3_s1_A1, estim_D3_s0_A1, estim_D3_s1_A1,
                               size_D2=size_D2, direct=direct_estimation)


def eval_prevalence_variations_D2(D1, D2, D3, classifier, model, sample_size=1000, nprevs=101, nreps=2):
    if isinstance(model, BaseQuantifier):
        func = eval_prevalence_variations_D2_quant
    elif isinstance(model, IndependenceGapEstimator):
        func = eval_prevalence_variations_D2_estim
    return func(D1, D2, D3, classifier, model, sample_size, nprevs, nreps)


def eval_prevalence_variations_D2_quant(D1, D2, D3, classifier, quantifier, sample_size=1000, nprevs=101, nreps=2):
    f = classifier.fit(*D1.Xy)

    D2_y1, D2_y0 = classify(f, D2)
    D3_y1, D3_y0 = classify(f, D3)

    M0 = deepcopy(quantifier)
    M1 = deepcopy(quantifier)

    def vary_and_test(quantifierVar:BaseQuantifier, quantifierFix:BaseQuantifier,
                      D2var: LabelledCollection, D2fix: LabelledCollection,
                      D3var: LabelledCollection, D3fix: LabelledCollection):

        # artificial samplings in D2var
        trueD2sampVar, estimD3var = [], []
        for D2sampleVar in D2var.artificial_sampling_generator(sample_size=sample_size, n_prevalences=nprevs, repeats=nreps):
            if D2sampleVar.prevalence().prod() == 0: continue
            quantifierVar.fit(D2sampleVar)
            estimD3var.append(quantifierVar.quantify(D3var.instances)[1])
            trueD2sampVar.append(D2sampleVar.prevalence()[1])
        estimD3var = np.asarray(estimD3var)
        trueD2sampVar = np.asarray(trueD2sampVar)
        trueD3var = np.asarray([D3var.prevalence()[1]] * len(estimD3var))

        # natural samplings in D2fix
        trueD2sampFix, estimD3fix = [], []
        for D2sampleFix in D2fix.natural_sampling_generator(sample_size=sample_size, repeats=len(estimD3var)):
            quantifierFix.fit(D2sampleFix)
            estimD3fix.append(quantifierFix.quantify(D3fix.instances)[1])
            trueD2sampFix.append(D2sampleFix.prevalence()[1])  # should be fixed at the original prevalence
        estimD3fix = np.asarray(estimD3fix)
        trueD2sampFix = np.asarray(trueD2sampFix)
        trueD3fix = [D3fix.prevalence()[1]] * len(estimD3fix)

        return trueD3var, estimD3var, trueD3fix, estimD3fix, trueD2sampVar, trueD2sampFix

    true_D3s0var, estim_D3s0var, true_D3s1fix, estim_D3s1fix, true_D2s0var, true_D2s1fix = vary_and_test(M0, M1, D2_y0, D2_y1, D3_y0, D3_y1)
    true_D3s1var, estim_D3s1var, true_D3s0fix, estim_D3s0fix, true_D2s1var, true_D2s0fix = vary_and_test(M1, M0, D2_y1, D2_y0, D3_y1, D3_y0)

    # stack the blocks
    true_D3_s0_A1   = np.concatenate([true_D3s0var, true_D3s0fix])
    true_D3_s1_A1   = np.concatenate([true_D3s1fix, true_D3s1var])
    estim_D3_s0_A1  = np.concatenate([estim_D3s0var, estim_D3s0fix])
    estim_D3_s1_A1  = np.concatenate([estim_D3s1fix, estim_D3s1var])
    true_D2_s0_A1   = np.concatenate([true_D2s0var, true_D2s0fix])
    true_D2_s1_A1   = np.concatenate([true_D2s1fix, true_D2s1var])

    return Result.with_columns(Protocols.VAR_D2_PREV, true_D3_s0_A1, true_D3_s1_A1, estim_D3_s0_A1, estim_D3_s1_A1,
                               var_s=np.asarray([0] * len(true_D3s0var) + [1] * len(true_D3s1var)),
                               D2_s0_prev=true_D2_s0_A1,
                               D2_s1_prev=true_D2_s1_A1,
                               D3_s0_prev=D3_y0.prevalence()[1],
                               D3_s1_prev=D3_y1.prevalence()[1]
                               )

def eval_prevalence_variations_D2_estim(D1, D2, D3, classifier, estimator, sample_size=1000, nprevs=101, nreps=2):
    f = classifier.fit(*D1.Xy)

    D2_y1, D2_y0 = classify(f, D2)
    D3_y1, D3_y0 = classify(f, D3)

    def vary_and_test(estimator:IndependenceGapEstimator,
                      D2var: LabelledCollection, D2fix: LabelledCollection,
                      D3var: LabelledCollection, D3fix: LabelledCollection,
                      order_s0_s1=True):

        # artificial samplings in D2var
        trueD2sampVar, trueD2sampFix = [], []
        direct_estimation = []
        for D2sampleVar in D2var.artificial_sampling_generator(sample_size=sample_size, n_prevalences=nprevs, repeats=nreps):
            D2sampleFix = D2fix.uniform_sampling(sample_size)
            if D2sampleVar.prevalence().prod() == 0: continue
            if order_s0_s1:
                estimator.fit(D2sampleVar, D2sampleFix)
                estim = estimator.predict(D3var.instances, D3fix.instances)
            else:
                estimator.fit(D2sampleFix, D2sampleVar)
                estim = estimator.predict(D3fix.instances, D3var.instances)
            direct_estimation.append(estim)
            trueD2sampVar.append(D2sampleVar.prevalence()[1])
            trueD2sampFix.append(D2sampleFix.prevalence()[1])
        n_results = len(trueD2sampVar)
        trueD2sampVar = np.asarray(trueD2sampVar)
        trueD2sampFix = np.asarray(trueD2sampFix)
        trueD3var = np.asarray([D3var.prevalence()[1]] * n_results)
        trueD3fix = np.asarray([D3fix.prevalence()[1]] * n_results)

        return trueD3var, trueD3fix, trueD2sampVar, trueD2sampFix, direct_estimation

    true_D3s0var, true_D3s1fix, true_D2s0var, true_D2s1fix, direct_estimation1 = \
        vary_and_test(estimator, D2_y0, D2_y1, D3_y0, D3_y1, order_s0_s1=True)
    true_D3s1var, true_D3s0fix, true_D2s1var, true_D2s0fix, direct_estimation2 = \
        vary_and_test(estimator, D2_y1, D2_y0, D3_y1, D3_y0, order_s0_s1=False)

    # stack the blocks
    true_D3_s0_A1   = np.concatenate([true_D3s0var, true_D3s0fix])
    true_D3_s1_A1   = np.concatenate([true_D3s1fix, true_D3s1var])
    estim_D3_s0_A1  = 0
    estim_D3_s1_A1  = 0
    true_D2_s0_A1   = np.concatenate([true_D2s0var, true_D2s0fix])
    true_D2_s1_A1   = np.concatenate([true_D2s1fix, true_D2s1var])
    direct_estimation = np.asarray(direct_estimation1 + direct_estimation2)

    return Result.with_columns(Protocols.VAR_D2_PREV, true_D3_s0_A1, true_D3_s1_A1, estim_D3_s0_A1, estim_D3_s1_A1,
                               var_s=np.asarray([0] * len(true_D3s0var) + [1] * len(true_D3s1var)),
                               D2_s0_prev=true_D2_s0_A1,
                               D2_s1_prev=true_D2_s1_A1,
                               D3_s0_prev=D3_y0.prevalence()[1],
                               D3_s1_prev=D3_y1.prevalence()[1],
                               direct=direct_estimation
                               )


def eval_prevalence_variations_D3(D1, D2, D3, classifier, model, sample_size=500, nprevs=101, nreps=2):
    if isinstance(model, BaseQuantifier):
        func = eval_prevalence_variations_D3_quant
    elif isinstance(model, IndependenceGapEstimator):
        func = eval_prevalence_variations_D3_estim
    return func(D1, D2, D3, classifier, model, sample_size, nprevs, nreps)


def eval_prevalence_variations_D3_quant(D1, D2, D3, classifier, quantifier, sample_size=500, nprevs=101, nreps=2):
    f = classifier.fit(*D1.Xy)

    D2_y1, D2_y0 = classify(f, D2)
    D3_y1, D3_y0 = classify(f, D3)

    M0 = deepcopy(quantifier).fit(D2_y0)
    M1 = deepcopy(quantifier).fit(D2_y1)

    def vary_and_test(quantifierVar:BaseQuantifier, quantifierFix:BaseQuantifier, varySplit: LabelledCollection, fixSplit:LabelledCollection):
        trueVar, estimVar = qp.evaluation.artificial_sampling_prediction(
            quantifierVar,  # the quantification method
            varySplit,  # the test set on which the method will be evaluated
            sample_size=sample_size,  # indicates the size of samples to be drawn
            n_prevpoints=nprevs,  # how many prevalence points will be extracted from the interval [0, 1] for each category
            n_repetitions=nreps,  # number of times each prevalence will be used to generate a test sample
            n_jobs=-1,  # indicates the number of parallel workers (-1 indicates, as in sklearn, all CPUs)
        )

        trueFix, estimFix = qp.evaluation.natural_sampling_prediction(
            quantifierFix,
            fixSplit,
            sample_size=sample_size,
            n_repetitions=trueVar.shape[0],
            n_jobs=-1
        )

        return trueVar[:,1], estimVar[:,1], trueFix[:,1], estimFix[:,1]

    true_s0var, estim_s0var, true_s1fix, estim_s1fix = vary_and_test(M0, M1, D3_y0, D3_y1)
    true_s1var, estim_s1var, true_s0fix, estim_s0fix = vary_and_test(M1, M0, D3_y1, D3_y0)

    # stack the blocks
    true_D3_s0_A1 = np.concatenate([true_s0var, true_s0fix])
    true_D3_s1_A1 = np.concatenate([true_s1fix, true_s1var])
    estim_D3_s0_A1 = np.concatenate([estim_s0var, estim_s0fix])
    estim_D3_s1_A1 = np.concatenate([estim_s1fix, estim_s1var])

    return Result.with_columns(Protocols.VAR_D3_PREV, true_D3_s0_A1, true_D3_s1_A1, estim_D3_s0_A1, estim_D3_s1_A1,
                               var_s=np.asarray([0]*len(true_s0var) + [1]*len(true_s1var)),
                               D2_s0_prev=D2_y0.prevalence()[1],
                               D2_s1_prev=D2_y1.prevalence()[1])


def eval_prevalence_variations_D3_estim(D1, D2, D3, classifier, estimator, sample_size=500, nprevs=101, nreps=2):
    f = classifier.fit(*D1.Xy)

    D2_y1, D2_y0 = classify(f, D2)
    D3_y1, D3_y0 = classify(f, D3)

    estimator.fit(D2_y0, D2_y1)

    def vary_and_test(estimator: IndependenceGapEstimator,
                      varySplit: LabelledCollection,
                      fixSplit: LabelledCollection,
                      order_s0_s1=True):

        trueVar, trueFix, direct_estimations = [],[],[]
        for varSample in varySplit.artificial_sampling_generator(sample_size, nprevs, nreps):
            fixSample = fixSplit.uniform_sampling(sample_size)
            if order_s0_s1:
                estim = estimator.predict(varSample.instances, fixSample.instances)
            else:
                estim = estimator.predict(fixSample.instances, varSample.instances)
            trueVar.append(varSample.prevalence()[1])
            trueFix.append(fixSample.prevalence()[1])
            direct_estimations.append(estim)

        return trueVar, trueFix, direct_estimations

    true_s0var, true_s1fix, direct1 = vary_and_test(estimator, D3_y0, D3_y1, order_s0_s1=True)
    true_s1var, true_s0fix, direct2 = vary_and_test(estimator, D3_y1, D3_y0, order_s0_s1=False)

    # stack the blocks
    true_D3_s0_A1 = np.concatenate([true_s0var, true_s0fix])
    true_D3_s1_A1 = np.concatenate([true_s1fix, true_s1var])
    estim_D3_s0_A1 = 0
    estim_D3_s1_A1 = 0
    direct = np.asarray(direct1+direct2)

    return Result.with_columns(Protocols.VAR_D3_PREV, true_D3_s0_A1, true_D3_s1_A1, estim_D3_s0_A1, estim_D3_s1_A1,
                               var_s=np.asarray([0] * len(true_s0var) + [1] * len(true_s1var)),
                               D2_s0_prev=D2_y0.prevalence()[1],
                               D2_s1_prev=D2_y1.prevalence()[1],
                               direct=direct)

    #def estim_signed_error(true_s0_A1, true_s1_A1, estim_s0_A1, estim_s1_A1):
    """
    Computes the gap between the estimated independence and the true independence.
    Positive (negative) values thus represent a tendency to overestimate (underestimate) the true value.
    """
#    true_inds = independence_gap(true_s0_A1, true_s1_A1)
#    estim_inds = independence_gap(estim_s0_A1, estim_s1_A1)
#    errors = estim_inds - true_inds
#    return errors


# def independence_abs_error(true_s0_A1, true_s1_A1, estim_s0_A1, estim_s1_A1):
#     """
#     Computes the error (absolute value of the gaps) between the estimated independence and the true independence.
#     Errors are always non-zero.
#     """
#     errors = np.abs(estim_signed_error(true_s0_A1, true_s1_A1, estim_s0_A1, estim_s1_A1))
#     return errors


# def independence_sqr_error(true_s0_A1, true_s1_A1, estim_s0_A1, estim_s1_A1):
#     """
#     Computes the error (absolute value of the gaps) between the estimated independence and the true independence.
#     Errors are always non-zero.
#     """
#     errors = estim_signed_error(true_s0_A1, true_s1_A1, estim_s0_A1, estim_s1_A1) ** 2
#     return errors
