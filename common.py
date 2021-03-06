import itertools
import pandas as pd
from sklearn.metrics import f1_score

import quapy as qp
import numpy as np

from method import IndependenceGapEstimator
from quapy.data import LabelledCollection
from sklearn.base import BaseEstimator
from copy import deepcopy
from tqdm import tqdm
from enum import Enum

from quapy.method.aggregative import EMQ
from quapy.method.base import BaseQuantifier


def independence_gap(ps_y0, ps_y1):
    # P(S=1|Y^=0) - P(S=1|Y^=1)
    return ps_y0 - ps_y1


def regular_independence_gap(ps1_y0, ps1_y1, py1, N, ps1_y0_prior, ps1_y1_prior):
    """
    Computes the Demographic Disparity score (regular independence gap):
    DP = P(Y=1|S=1) - P(Y=1|S=0)
    from the (predicted or true) quantities P(S=1|Y=0) and P(S=1|Y=1) (from D3), which are smoothed via additive
    smoothing (Laplace smoothing) and the true P(Y=1). Smoothing is carried out by taking the (true) P(S=1|Y=0) and
    P(S=1|Y=1), as observed in D2, as the known incidence rates. N is the population size in D3.

    """
    py0 = 1-py1

    Ny0 = py0 * N
    Ny1 = py1 * N

    ps1_y0 = smooth(ps1_y0, Ny0, ps1_y0_prior)
    ps1_y1 = smooth(ps1_y1, Ny1, ps1_y1_prior)

    ps1 = ps1_y1 * py1 + ps1_y0 * py0
    ps0 = 1-ps1

    rdp = -1 + ps1_y1 * py1/ps1 + (1 - ps1_y0) * py0/ps0

    assert not np.isnan(rdp).any(), 'nan found in rdp array'

    return rdp


def smooth(ps1_yx, Nyx, mu_1):
    alpha = 1/2  # we assume the pseudocount of the additive smoothing be half a data item
    d = 2  # two classes, S=1 and S=0
    # mu_1 is a prior for ps1_yx; in our case, is ps1_yx as observed in D2
    s1_count = ps1_yx * Nyx
    ps1_yx_smooth = (s1_count + mu_1*alpha*d) / (Nyx + alpha*d)
    # with alpha=1/2 and d=2, this actually amounts to:
    # ps1_yx_smooth = (s1_count + mu_1) / (Nyx + 1)
    return ps1_yx_smooth


def regular_independence_gap_no_smooth(ps_y0, ps_y1, py):
    # P(Y^=1|S=1) - P(Y^=1|S=0)
    ps = ps_y1*py + ps_y0*(1-py)
    rdp = -1 + ps_y1 * py / ps + (1 - ps_y0) * (1 - py) / (1 - ps)
    # if not all(0 < ps_ < 1 for ps_ in ps):
    #     idcs0 = [i for i, ps_ in enumerate(ps) if ps_ == 0]
    #     idcs1 = [i for i, ps_ in enumerate(ps) if ps_ == 1]
    #     rdp = -1 + ps_y1 * py / ps + (1 - ps_y0) * (1 - py) / (1 - ps)
    #     rdp_ = rdp.tolist()
    #     rdp_ps0 = [rdp_[i] for i in idcs0]
    #     rdp_ps1 = [rdp_[i] for i in idcs1]
    #     rdp[rdp.isna()]=3
    #
    #     a=0
    # assert(all(0 < ps_ < 1 for ps_ in ps))
    # assert(all(-1 < rdp_ < 1 for rdp_ in rdp))
    # rdp[rdp.isna()] = 3
    return rdp


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
    def with_columns(cls, protocol, true_D3_y0_A1, true_D3_y1_A1, estim_D3_y0_A1, estim_D3_y1_A1, **kwargs):
        columns = {
            'protocol': protocol,
            'trueD3y0A1': true_D3_y0_A1,
            'trueD3y1A1': true_D3_y1_A1,
            'estimD3y0A1': estim_D3_y0_A1,
            'estimD3y1A1': estim_D3_y1_A1
        }
        columns.update(kwargs)
        return Result(pd.DataFrame(columns))

    @classmethod
    def with_freecolumns(cls, **kwargs):
        return Result(pd.DataFrame(kwargs))

    def independence_gap(self, regDP=False):
        d = self.data
        if regDP:
            ps1_y0_d3 = d['trueD3y0A1'].values
            ps1_y1_d3 = d['trueD3y1A1'].values
            py1_d3 = d['pD3y1'].values
            N_d3 = d['sizeD3'].values
            ps1_y0_prior = d['trueD2y0A1'].values
            ps1_y1_prior = d['trueD3y1A1'].values
            return regular_independence_gap(ps1_y0_d3, ps1_y1_d3, py1_d3, N_d3, ps1_y0_prior, ps1_y1_prior)
        else:
            return independence_gap(d['trueD3y0A1'], d['trueD3y1A1'])


    def estim_independence_gap(self, regDP=False):
        d = self.data
        if 'direct' in d and not d['direct'].isnull().values.any():
            #print('DIRECT IS PRESENT', d['Q_name'].unique())
            return d['direct']
        else:
            if regDP:
                ps1_y0_d3 = d['estimD3y0A1'].values
                ps1_y1_d3 = d['estimD3y1A1'].values
                py1_d3 = d['pD3y1'].values
                N_d3 = d['sizeD3'].values
                ps1_y0_prior = d['trueD2y0A1'].values
                ps1_y1_prior = d['trueD3y1A1'].values
                return regular_independence_gap(ps1_y0_d3, ps1_y1_d3, py1_d3, N_d3, ps1_y0_prior, ps1_y1_prior)
            else:
                return independence_gap(d['estimD3y0A1'], d['estimD3y1A1'])

    def independence_signed_error(self, regDP=False):
        true_inds = self.independence_gap(regDP)
        assert (all(-1 <= ig <= 1 for ig in true_inds))
        estim_inds = self.estim_independence_gap(regDP)
        errors = estim_inds - true_inds
        return errors
        #return estim_signed_error(d['trueD3s0A1'], d['trueD3s1A1'], d['estimD3s0A1'], d['estimD3s1A1'])

    def independence_abs_error(self, regDP=False):
        """
        Computes the error (absolute value of the gaps) between the estimated independence and the true independence.
        Errors are always non-zero.
        """
        return np.abs(self.independence_signed_error(regDP))
        #return independence_abs_error(d['trueD3s0A1'], d['trueD3s1A1'], d['estimD3s0A1'], d['estimD3s1A1'])

    def D3y0_abs_error(self):
        d = self.data
        return qp.error.ae(d['trueD3y0A1'].values.reshape(-1,1), d['estimD3y0A1'].values.reshape(-1,1))

    def D3y1_abs_error(self):
        d = self.data
        return qp.error.ae(d['trueD3y1A1'].values.reshape(-1,1), d['estimD3y1A1'].values.reshape(-1,1))

    def independence_sqr_error(self, regDP=False):
        """
        Computes the error (absolute value of the gaps) between the estimated independence and the true independence.
        Errors are always non-zero.
        """
        return self.independence_signed_error(regDP) ** 2

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


def __train_quantifiers(quantifier, D2_y0, D2_y1, splitD2):
    if splitD2:
        M1 = deepcopy(quantifier).fit(D2_y1)
        M0 = deepcopy(quantifier).fit(D2_y0)
    else:
        M1 = deepcopy(quantifier).fit(D2_y1 + D2_y0)
        M0 = M1
    return M0, M1


def eval_prevalence_variations_D1(D1, D2, D3, AD1, classifier, model, sample_size=500, nprevs=101, nreps=2, splitD2=True, regDP=False):
    if isinstance(model, BaseQuantifier):
        func = eval_prevalence_variations_D1_quant
    elif isinstance(model, IndependenceGapEstimator):
        func = eval_prevalence_variations_D1_estim
    return func(D1, D2, D3, AD1, classifier, model, sample_size, nprevs, nreps, splitD2)


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


def eval_prevalence_variations_D1_quant(D1, D2, D3, AD1, classifier, quantifier, sample_size=500, nprevs=101, nreps=2, splitD2=True):
    trueD3y0A1, trueD3y1A1, estimD3y0A1, estimD3y1A1, p_Aeqy, p_yhat_D2, p_yhat_D3 = [], [], [], [], [], [], []
    trueD2y0A1, trueD2y1A1 = [], []

    for D2_y0, D2_y1, D3_y0, D3_y1, prevAeqY in __var_D1_generator(D1, D2, D3, AD1, classifier, sample_size, nprevs, nreps):
        M0, M1 = __train_quantifiers(quantifier, D2_y0, D2_y1, splitD2)
        estimD3y1A1.append(M1.quantify(D3_y1.instances)[1])
        estimD3y0A1.append(M0.quantify(D3_y0.instances)[1])
        trueD3y1A1.append(D3_y1.prevalence()[1])
        trueD3y0A1.append(D3_y0.prevalence()[1])
        p_Aeqy.append(prevAeqY)
        p_yhat_D2.append(len(D2_y1) / (len(D2_y0) + len(D2_y1)))
        p_yhat_D3.append(len(D3_y1) / (len(D3_y0) + len(D3_y1)))
        trueD2y0A1.append(D2_y0.prevalence()[1])
        trueD2y1A1.append(D2_y1.prevalence()[1])

    sizeD3 = np.full_like(trueD3y0A1, fill_value=len(D3))
    
    return Result.with_columns(Protocols.VAR_D1_PREV, trueD3y0A1, trueD3y1A1, estimD3y0A1, estimD3y1A1,
                               p_Aeqy=p_Aeqy, 
                               pD2y1=p_yhat_D2,
                               pD3y1=p_yhat_D3,
                               sizeD3=sizeD3,
                               trueD2y0A1=trueD2y0A1,
                               trueD2y1A1=trueD2y1A1)


def eval_prevalence_variations_D1_estim(D1, D2, D3, AD1, classifier, estimator:IndependenceGapEstimator, sample_size=500, nprevs=101, nreps=2, splitD2=True):
    if not splitD2: print('warning: calling protocol with Independence Gap Estimator and splitD2=False')
    trueD3y0A1, trueD3y1A1, estimD3y0A1, estimD3y1A1, p_Aeqy, p_yhat_D2, p_yhat_D3 = [], [], [], [], [], [], []
    trueD2y0A1, trueD2y1A1 = [], []
    direct_estimation = []

    for D2_y0, D2_y1, D3_y0, D3_y1, prevAeqY in __var_D1_generator(D1, D2, D3, AD1, classifier, sample_size, nprevs, nreps):
        estimator.fit(D2_y0, D2_y1)
        ig_hat = estimator.predict(D3_y0.instances, D3_y1.instances)
        direct_estimation.append(ig_hat)
        estimD3y1A1.append(0)
        estimD3y0A1.append(0)
        trueD3y1A1.append(D3_y1.prevalence()[1])
        trueD3y0A1.append(D3_y0.prevalence()[1])
        p_Aeqy.append(prevAeqY)
        p_yhat_D2.append(len(D2_y1) / (len(D2_y0) + len(D2_y1)))
        p_yhat_D3.append(len(D3_y1) / (len(D3_y0) + len(D3_y1)))
        trueD2y0A1.append(D2_y0.prevalence()[1])
        trueD2y1A1.append(D2_y1.prevalence()[1])

    sizeD3 = np.full_like(trueD3y0A1, fill_value=len(D3))

    return Result.with_columns(Protocols.VAR_D1_PREV, trueD3y0A1, trueD3y1A1, estimD3y0A1, estimD3y1A1,
                               p_Aeqy=p_Aeqy,
                               pD2y1=p_yhat_D2,
                               pD3y1=p_yhat_D3,
                               sizeD3=sizeD3,
                               trueD2y0A1=trueD2y0A1,
                               trueD2y1A1=trueD2y1A1,
                               direct=direct_estimation)


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


def eval_prevalence_variations_D1_flip(D1, D2, D3, AD1, classifier, model, sample_size=500, nprevs=101, nreps=2, splitD2=True, regDP=False):
    if isinstance(model, BaseQuantifier):
        func = eval_prevalence_variations_D1_flip_quant
    elif isinstance(model, IndependenceGapEstimator):
        func = eval_prevalence_variations_D1_flip_estim
    return func(D1, D2, D3, AD1, classifier, model, sample_size, nprevs, nreps, splitD2)


def eval_prevalence_variations_D1_flip_quant(D1, D2, D3, AD1, classifier, quantifier, sample_size=500, nprevs=101, nreps=2, splitD2=True):
    trueD3y0A1, trueD3y1A1, estimD3y0A1, estimD3y1A1, p_Aeqy, p_yhat_D2, p_yhat_D3 = [], [], [], [], [], [], []
    trueD2y0A1, trueD2y1A1 = [], []

    for D2_y0, D2_y1, D3_y0, D3_y1, prev in __var_D1_flip_generator(D1, D2, D3, AD1, classifier, sample_size, nprevs, nreps):
        M0, M1 = __train_quantifiers(quantifier, D2_y0, D2_y1, splitD2)
        estimD3y1A1.append(M1.quantify(D3_y1.instances)[1])
        estimD3y0A1.append(M0.quantify(D3_y0.instances)[1])
        trueD3y1A1.append(D3_y1.prevalence()[1])
        trueD3y0A1.append(D3_y0.prevalence()[1])
        p_Aeqy.append(prev)
        p_yhat_D2.append(len(D2_y1) / (len(D2_y0) + len(D2_y1)))
        p_yhat_D3.append(len(D3_y1) / (len(D3_y0) + len(D3_y1)))
        trueD2y0A1.append(D2_y0.prevalence()[1])
        trueD2y1A1.append(D2_y1.prevalence()[1])

    sizeD3 = np.full_like(trueD3y0A1, fill_value=len(D3))

    return Result.with_columns(Protocols.VAR_D1_PREVFLIP, trueD3y0A1, trueD3y1A1, estimD3y0A1, estimD3y1A1,
                               p_Aeqy=p_Aeqy,
                               pD2y1=p_yhat_D2,
                               pD3y1=p_yhat_D3,
                               sizeD3=sizeD3,
                               trueD2y0A1=trueD2y0A1,
                               trueD2y1A1=trueD2y1A1)


def eval_prevalence_variations_D1_flip_estim(D1, D2, D3, AD1, classifier, estimator, sample_size=500, nprevs=101, nreps=2, splitD2=True):
    if not splitD2: print('warning: calling protocol with Independence Gap Estimator and splitD2=False')
    trueD3y0A1, trueD3y1A1, estimD3y0A1, estimD3y1A1, p_Aeqy, p_yhat_D2, p_yhat_D3 = [], [], [], [], [], [], []
    trueD2y0A1, trueD2y1A1 = [], []
    direct_estimation = []

    for D2_y0, D2_y1, D3_y0, D3_y1, prev in __var_D1_flip_generator(D1, D2, D3, AD1, classifier, sample_size, nprevs, nreps):
        estimator.fit(D2_y0, D2_y1)
        ig_hat = estimator.predict(D3_y0.instances, D3_y1.instances)
        direct_estimation.append(ig_hat)
        estimD3y1A1.append(0)
        estimD3y0A1.append(0)
        trueD3y1A1.append(D3_y1.prevalence()[1])
        trueD3y0A1.append(D3_y0.prevalence()[1])
        p_Aeqy.append(prev)
        p_yhat_D2.append(len(D2_y1) / (len(D2_y0) + len(D2_y1)))
        p_yhat_D3.append(len(D3_y1) / (len(D3_y0) + len(D3_y1)))
        trueD2y0A1.append(D2_y0.prevalence()[1])
        trueD2y1A1.append(D2_y1.prevalence()[1])

    sizeD3 = np.full_like(trueD3y0A1, fill_value=len(D3))

    return Result.with_columns(Protocols.VAR_D1_PREVFLIP, trueD3y0A1, trueD3y1A1, estimD3y0A1, estimD3y1A1,
                               p_Aeqy=p_Aeqy,
                               pD2y1=p_yhat_D2,
                               pD3y1=p_yhat_D3,
                               sizeD3=sizeD3,
                               trueD2y0A1=trueD2y0A1,
                               trueD2y1A1=trueD2y1A1,
                               direct=direct_estimation)



def eval_size_variations_D2(D1, D2, D3, classifier, model, nreps=10, sample_sizes=None, splitD2=True):
    if isinstance(model, BaseQuantifier):
        func = eval_size_variations_D2_quant
    elif isinstance(model, IndependenceGapEstimator):
        func = eval_size_variations_D2_estim
    return func(D1, D2, D3, classifier, model, nreps, sample_sizes, splitD2)


def eval_size_variations_D2_quant(D1, D2, D3, classifier, quantifier, nreps=10, sample_sizes=None, splitD2=True):
    if sample_sizes is None:
        min_size = 1000
        sample_sizes = [int(round(val)) for val in np.geomspace(min_size, len(D2.labels), num=5)]

    f = classifier.fit(*D1.Xy)
    D2_y_hat = classifier.predict(D2.instances)
    D3_y1, D3_y0 = classify(f, D3)

    def vary_and_test(quantifier, D2, D2_y_hat, D3_y0, D3_y1, sample_sizes, nreps):
        # M0 = deepcopy(quantifier)
        # M1 = deepcopy(quantifier)
        estim_M0_A1, estim_M1_A1  = [], []
        size_D2, p_yhat_D2, p_yhat_D3 = [], [], []
        true_D2_Y0_A1, true_D2_Y1_A1 = [], []
        for sample_D2, sample_idcs in tqdm(natural_sampling_generator_varsize(D2, sample_sizes, nreps),
                                           desc='training quantifiers at size variations in D2'):
            assert sample_D2.prevalence().prod() != 0
            sample_D2_y_hat = D2_y_hat[sample_idcs]
            sample_D2_y0 = sample_D2.sampling_from_index(sample_D2_y_hat == 0)
            sample_D2_y1 = sample_D2.sampling_from_index(sample_D2_y_hat == 1)
            M0, M1 = __train_quantifiers(quantifier, sample_D2_y0, sample_D2_y1, splitD2)
            # M0.fit(sample_D2_y0)
            # M1.fit(sample_D2_y1)
            estim_M0_A1.append(M0.quantify(D3_y0.instances)[1])
            estim_M1_A1.append(M1.quantify(D3_y1.instances)[1])
            size_D2.append(len(sample_D2))
            p_yhat_D2.append(len(sample_D2_y1) / (len(sample_D2_y0) + len(sample_D2_y1)))
            p_yhat_D3.append(len(D3_y1) / (len(D3_y0) + len(D3_y1)))
            true_D2_Y0_A1.append(sample_D2_y0.prevalence()[1])
            true_D2_Y1_A1.append(sample_D2_y1.prevalence()[1])

        estim_M0_A1 = np.asarray(estim_M0_A1)
        estim_M1_A1 = np.asarray(estim_M1_A1)
        true_M0_A1 = np.asarray([D3_y0.prevalence()[1]] * len(estim_M0_A1))
        true_M1_A1 = np.asarray([D3_y1.prevalence()[1]] * len(estim_M1_A1))
        true_D2_Y0_A1 = np.asarray(true_D2_Y0_A1)
        true_D2_Y1_A1 = np.asarray(true_D2_Y1_A1)

        return true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, size_D2, p_yhat_D2, p_yhat_D3, true_D2_Y0_A1, true_D2_Y1_A1

    trueD3y0A1, estimD3y0A1, trueD3y1A1, estimD3y1A1, sizeD2, pD2y1, pD3y1, trueD2y0A1, trueD2y1A1 = \
        vary_and_test(quantifier, D2, D2_y_hat, D3_y0, D3_y1, sample_sizes, nreps)

    sizeD3 = np.full_like(trueD3y0A1, fill_value=len(D3))

    return Result.with_columns(Protocols.VAR_D2_SIZE, trueD3y0A1, trueD3y1A1, estimD3y0A1, estimD3y1A1,
                               sizeD2=sizeD2,
                               pD2y1=pD2y1,
                               sizeD3=sizeD3,
                               pD3y1=pD3y1,
                               trueD2y0A1=trueD2y0A1,
                               trueD2y1A1=trueD2y1A1)


def eval_size_variations_D2_estim(D1, D2, D3, classifier, estimator, nreps=10, sample_sizes=None, splitD2=True):
    if not splitD2: print('warning: calling protocol with Independence Gap Estimator and splitD2=False')
    if sample_sizes is None:
        min_size = 1000
        sample_sizes = [int(round(val)) for val in np.geomspace(min_size, len(D2.labels), num=5)]

    f = classifier.fit(*D1.Xy)
    D2_y_hat = classifier.predict(D2.instances)
    D3_y1, D3_y0 = classify(f, D3)

    def vary_and_test(estimator, D2, D2_y_hat, D3_y0, D3_y1, sample_sizes, nreps):
        estim_M0_A1, estim_M1_A1  = [], []
        size_D2, p_yhat_D2, p_yhat_D3 = [], [], []
        true_D2_Y0_A1, true_D2_Y1_A1 = [], []
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
            p_yhat_D2.append(len(sample_D2_y1) / (len(sample_D2_y0) + len(sample_D2_y1)))
            p_yhat_D3.append(len(D3_y1) / (len(D3_y0) + len(D3_y1)))
            true_D2_Y0_A1.append(sample_D2_y0.prevalence()[1])
            true_D2_Y1_A1.append(sample_D2_y1.prevalence()[1])

        estim_M0_A1 = np.asarray(estim_M0_A1)
        estim_M1_A1 = np.asarray(estim_M1_A1)
        true_M0_A1 = np.asarray([D3_y0.prevalence()[1]] * len(estim_M0_A1))
        true_M1_A1 = np.asarray([D3_y1.prevalence()[1]] * len(estim_M1_A1))

        return true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, size_D2, p_yhat_D2, p_yhat_D3, \
               true_D2_Y0_A1, true_D2_Y1_A1, direct_estimation

    trueD3y0A1, estimD3y0A1, trueD3y1A1, estimD3y1A1, size_D2, p_yhat_D2, p_yhat_D3, trueD2y0A1, trueD2y1A1, direct_estimation = \
        vary_and_test(estimator, D2, D2_y_hat, D3_y0, D3_y1, sample_sizes, nreps)

    sizeD3 = np.full_like(trueD3y0A1, fill_value=len(D3))

    return Result.with_columns(Protocols.VAR_D2_SIZE, trueD3y0A1, trueD3y1A1, estimD3y0A1, estimD3y1A1,
                               sizeD2=size_D2,
                               pD2y1=p_yhat_D2,
                               sizeD3=sizeD3,
                               pD3y1=p_yhat_D3,
                               trueD2y0A1=trueD2y0A1,
                               trueD2y1A1=trueD2y1A1,
                               direct=direct_estimation)


def eval_prevalence_variations_D2(D1, D2, D3, classifier, model, sample_size=1000, nprevs=101, nreps=2, splitD2=True, regDP=False):
    if isinstance(model, BaseQuantifier):
        func = eval_prevalence_variations_D2_quant
    elif isinstance(model, IndependenceGapEstimator):
        func = eval_prevalence_variations_D2_estim
    return func(D1, D2, D3, classifier, model, sample_size, nprevs, nreps, splitD2)


def eval_prevalence_variations_D2_quant(D1, D2, D3, classifier, quantifier, sample_size=1000, nprevs=101, nreps=2, splitD2=True):
    f = classifier.fit(*D1.Xy)

    D2_y1, D2_y0 = classify(f, D2)
    D3_y1, D3_y0 = classify(f, D3)

    def vary_and_test(quantifier,
                      D2var: LabelledCollection, D2fix: LabelledCollection,
                      D3var: LabelledCollection, D3fix: LabelledCollection):

        # artificial samplings in D2var
        trueD2sampVar, estimD3var = [], []
        trueD2sampFix, estimD3fix = [], []
        for D2sampleVar in D2var.artificial_sampling_generator(sample_size=sample_size, n_prevalences=nprevs, repeats=nreps):
            if D2sampleVar.prevalence().prod() == 0: continue
            D2sampleFix = D2fix.uniform_sampling(sample_size)
            quantifierVar, quantifierFix = __train_quantifiers(quantifier, D2sampleVar, D2sampleFix, splitD2)
            estimD3var.append(quantifierVar.quantify(D3var.instances)[1])
            estimD3fix.append(quantifierFix.quantify(D3fix.instances)[1])
            trueD2sampVar.append(D2sampleVar.prevalence()[1])
            trueD2sampFix.append(D2sampleFix.prevalence()[1])  # should be fixed at the original prevalence
        estimD3var = np.asarray(estimD3var)
        estimD3fix = np.asarray(estimD3fix)
        trueD2sampVar = np.asarray(trueD2sampVar)
        trueD2sampFix = np.asarray(trueD2sampFix)
        trueD3var = np.asarray([D3var.prevalence()[1]] * len(estimD3var))
        trueD3fix = [D3fix.prevalence()[1]] * len(estimD3fix)

        return trueD3var, estimD3var, trueD3fix, estimD3fix, trueD2sampVar, trueD2sampFix

    true_D3s0var, estim_D3s0var, true_D3s1fix, estim_D3s1fix, true_D2s0var, true_D2s1fix = \
        vary_and_test(quantifier, D2_y0, D2_y1, D3_y0, D3_y1)
    true_D3s1var, estim_D3s1var, true_D3s0fix, estim_D3s0fix, true_D2s1var, true_D2s0fix = \
        vary_and_test(quantifier, D2_y1, D2_y0, D3_y1, D3_y0)

    # stack the blocks
    true_D3_y0_A1   = np.concatenate([true_D3s0var, true_D3s0fix])
    true_D3_y1_A1   = np.concatenate([true_D3s1fix, true_D3s1var])
    estim_D3_y0_A1  = np.concatenate([estim_D3s0var, estim_D3s0fix])
    estim_D3_y1_A1  = np.concatenate([estim_D3s1fix, estim_D3s1var])
    true_D2_y0_A1   = np.concatenate([true_D2s0var, true_D2s0fix])
    true_D2_y1_A1   = np.concatenate([true_D2s1fix, true_D2s1var])
    p_yhat_D2 = 0.5  # sample_size / (2*sample_size)
    p_yhat_D3 = len(D3_y1) / (len(D3_y0) + len(D3_y1))

    pD2y1 = np.full_like(true_D3_y0_A1, fill_value=p_yhat_D2)
    pD3y1 = np.full_like(true_D3_y0_A1, fill_value=p_yhat_D3)
    sizeD3 = np.full_like(true_D3_y0_A1, fill_value=len(D3))

    return Result.with_columns(Protocols.VAR_D2_PREV, true_D3_y0_A1, true_D3_y1_A1, estim_D3_y0_A1, estim_D3_y1_A1,
                               var_s=np.asarray([0] * len(true_D3s0var) + [1] * len(true_D3s1var)),
                               trueD2y0A1=true_D2_y0_A1,
                               trueD2y1A1=true_D2_y1_A1,
                               trueD3y0A1=D3_y0.prevalence()[1],
                               trueD3y1A1=D3_y1.prevalence()[1],
                               pD2y1=pD2y1,
                               pD3y1=pD3y1,
                               sizeD3=sizeD3
                               )

def eval_prevalence_variations_D2_estim(D1, D2, D3, classifier, estimator, sample_size=1000, nprevs=101, nreps=2, splitD2=True):
    if not splitD2: print('warning: calling protocol with Independence Gap Estimator and splitD2=False')
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
    trueD3y0A1   = np.concatenate([true_D3s0var, true_D3s0fix])
    trueD3y1A1   = np.concatenate([true_D3s1fix, true_D3s1var])
    estimD3y0A1  = 0
    estimD3y1A1  = 0
    true_D2_y0_A1   = np.concatenate([true_D2s0var, true_D2s0fix])
    true_D2_y1_A1   = np.concatenate([true_D2s1fix, true_D2s1var])
    direct_estimation = np.asarray(direct_estimation1 + direct_estimation2)
    p_yhat_D2 = 0.5  # sample_size / (2*sample_size)
    p_yhat_D3 = len(D3_y1) / (len(D3_y0) + len(D3_y1))
    sizeD3 = np.full_like(trueD3y0A1, fill_value=len(D3))

    return Result.with_columns(Protocols.VAR_D2_PREV, trueD3y0A1, trueD3y1A1, estimD3y0A1, estimD3y1A1,
                               var_s=np.asarray([0] * len(true_D3s0var) + [1] * len(true_D3s1var)),
                               trueD2y0A1=true_D2_y0_A1,
                               trueD2y1A1=true_D2_y1_A1,
                               trueD3y0A1=D3_y0.prevalence()[1],
                               trueD3y1A1=D3_y1.prevalence()[1],
                               pD2y1=[p_yhat_D2] * len(true_D2_y0_A1),
                               pD3y1=[p_yhat_D3] * len(true_D2_y0_A1),
                               sizeD3=sizeD3,
                               direct=direct_estimation
                               )


def eval_clf_prevalence_variations_D2(D2split, D3split, clf, quantifiers, sample_size=1000, nprevs=101, nreps=2):
    # artificial samplings in D2split
    trueD2sample, accs, f1s, emq_accs, emq_f1s = [], [], [], [], []
    quantif_results = {q_name:[] for q_name,_ in quantifiers}
    for D2sample in D2split.artificial_sampling_generator(sample_size=sample_size, n_prevalences=nprevs, repeats=nreps):
        if D2sample.prevalence().prod() == 0: continue
        clf.fit(*D2sample.Xy)
        predictions = clf.predict(D3split.instances)
        accs.append((predictions == D3split.labels).mean())
        f1s.append(f1_score(D3split.labels, predictions))
        trueD2sample.append(D2sample.prevalence()[1])
        for q_name, q in quantifiers:
            q.fit(D2sample)
            prev = q.quantify(D3split.instances)[1]
            quantif_results[q_name].append(prev)
            if isinstance(q, EMQ):
                emq_predictions = q.predict_proba(D3split.instances)[:,1]>0.5
                emq_accs.append((emq_predictions == D3split.labels).mean())
                emq_f1s.append(f1_score(D3split.labels, emq_predictions))

    trueD2sample = np.asarray(trueD2sample)
    trueD3sample = np.asarray([D3split.prevalence()[1]] * len(trueD2sample))
    return Result.with_freecolumns(trueD2sample=trueD2sample, trueD3sample=trueD3sample,
                                   accs=accs, f1s=f1s,
                                   emq_accs=emq_accs, emq_f1s=emq_f1s,
                                   **quantif_results)


def eval_prevalence_variations_D3(D1, D2, D3, classifier, model, sample_size=500, nprevs=101, nreps=2, splitD2=True, regDP=False):
    if isinstance(model, BaseQuantifier):
        func = eval_prevalence_variations_D3_quant
    elif isinstance(model, IndependenceGapEstimator):
        func = eval_prevalence_variations_D3_estim
    return func(D1, D2, D3, classifier, model, sample_size, nprevs, nreps, splitD2)


def eval_prevalence_variations_D3_quant(D1, D2, D3, classifier, quantifier, sample_size=500, nprevs=101, nreps=2, splitD2=True):
    f = classifier.fit(*D1.Xy)

    D2_y1, D2_y0 = classify(f, D2)
    D3_y1, D3_y0 = classify(f, D3)

    M0, M1 = __train_quantifiers(quantifier, D2_y0, D2_y1, splitD2)

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
    trueD3y0A1 = np.concatenate([true_s0var, true_s0fix])
    trueD3y1A1 = np.concatenate([true_s1fix, true_s1var])
    estimD3y0A1 = np.concatenate([estim_s0var, estim_s0fix])
    estimD3y1A1 = np.concatenate([estim_s1fix, estim_s1var])
    p_yhat_D2 = len(D2_y1) / (len(D2_y0) + len(D2_y1))
    p_yhat_D3 = 0.5  # sample_size / (2*sample_size)
    sizeD3 = np.full_like(trueD3y0A1, fill_value=sample_size*2)  # one sample_size for varying sets plus one sample_size for fixed ones

    return Result.with_columns(Protocols.VAR_D3_PREV, trueD3y0A1, trueD3y1A1, estimD3y0A1, estimD3y1A1,
                               var_s=np.asarray([0]*len(true_s0var) + [1]*len(true_s1var)),
                               trueD2y0A1=D2_y0.prevalence()[1],
                               trueD2y1A1=D2_y1.prevalence()[1],
                               pD2y1=[p_yhat_D2] * len(trueD3y0A1),
                               pD3y1=[p_yhat_D3] * len(trueD3y0A1),
                               sizeD3=sizeD3,
                               )


def eval_prevalence_variations_D3_estim(D1, D2, D3, classifier, estimator, sample_size=500, nprevs=101, nreps=2, splitD2=True):
    if not splitD2: print('warning: calling protocol with Independence Gap Estimator and splitD2=False')
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
    trueD3y0A1 = np.concatenate([true_s0var, true_s0fix])
    trueD3y1A1 = np.concatenate([true_s1fix, true_s1var])
    estimD3y0A1 = 0
    estimD3y1A1 = 0
    direct = np.asarray(direct1+direct2)
    p_yhat_D2 = len(D2_y1) / (len(D2_y0) + len(D2_y1))
    p_yhat_D3 = 0.5  # sample_size / (2*sample_size)
    sizeD3 = np.full_like(trueD3y0A1, fill_value=sample_size * 2)  # one sample_size for varying sets plus one sample_size for fixed ones

    return Result.with_columns(Protocols.VAR_D3_PREV, trueD3y0A1, trueD3y1A1, estimD3y0A1, estimD3y1A1,
                               var_s=np.asarray([0] * len(true_s0var) + [1] * len(true_s1var)),
                               trueD2y0A1=D2_y0.prevalence()[1],
                               trueD2y1A1=D2_y1.prevalence()[1],
                               pD2y1=[p_yhat_D2] * len(trueD3y0A1),
                               pD3y1=[p_yhat_D3] * len(trueD3y0A1),
                               sizeD3=sizeD3,
                               direct=direct)


def eval_clf_prevalence_variations_D3(D2split, D3split, clf, quantifiers, sample_size=1000, nprevs=101, nreps=2):
    clf.fit(*D2split.Xy)
    for _,q in quantifiers:
        q.fit(D2split)

    trueD3sample, accs, f1s, emq_accs, emq_f1s = [], [], [], [], []
    quantif_results = {q_name:[] for q_name,_ in quantifiers}
    for D3sample in D3split.artificial_sampling_generator(sample_size=sample_size, n_prevalences=nprevs, repeats=nreps):
        predictions = clf.predict(D3sample.instances)
        accs.append((predictions == D3sample.labels).mean())
        f1s.append(f1_score(D3sample.labels, predictions))
        trueD3sample.append(D3sample.prevalence()[1])
        for q_name, q in quantifiers:
            prev = q.quantify(D3sample.instances)[1]
            quantif_results[q_name].append(prev)
            if isinstance(q, EMQ):
                emq_predictions = q.predict_proba(D3sample.instances)[:, 1] > 0.5
                emq_accs.append((emq_predictions == D3sample.labels).mean())
                emq_f1s.append(f1_score(D3sample.labels, emq_predictions))


    trueD3sample = np.asarray(trueD3sample)
    trueD2sample = np.asarray([D2split.prevalence()[1]] * len(trueD3sample))
    return Result.with_freecolumns(trueD2sample=trueD2sample, trueD3sample=trueD3sample,
                                   accs=accs, f1s=f1s,
                                   emq_accs=emq_accs, emq_f1s=emq_f1s,
                                   **quantif_results)