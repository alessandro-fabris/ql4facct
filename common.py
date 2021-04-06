import quapy as qp
import numpy as np
from quapy.data import LabelledCollection
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
from tqdm import tqdm


def uniform_sampling_with_indices(D: LabelledCollection, size):
    unif_index = D.uniform_sampling_index(size)
    return D.sampling_from_index(unif_index), unif_index


def natural_sampling_generator_varsize(D: LabelledCollection, sample_sizes, repeats):
    for _ in range(repeats):
        for sample_size in sample_sizes:
            yield uniform_sampling_with_indices(D, sample_size)

def get_prevalences(D2_y0, D2_y1, D3_y0, D3_y1):
    d = {"prev_A1_D2_y1": sum(D2_y1.labels) / len(D2_y1.labels),
         "prev_A1_D3_y1": sum(D3_y1.labels) / len(D3_y1.labels),
         "prev_A1_D2_y0": sum(D2_y0.labels) / len(D2_y0.labels),
         "prev_A1_D3_y0": sum(D3_y0.labels) / len(D3_y0.labels),
         "perc_pred_y1_D2": len(D2_y1) / len(D2_y0 + D2_y1),
         "perc_pred_y1_D3": len(D3_y1) / len(D3_y0 + D3_y1)}
    return d

def joint_code_yA(y,A):
    return y + 2 * A

def split_data(X, y, A, seed):

    # jointly codify (y,A) as 0,0->0; 1,0->1; 0,1->2; 1,1->3 to stratify
    yA = joint_code_yA(y,A)

    data = LabelledCollection(X, yA)
    D1, data = data.split_stratified(train_prop=1/3, random_state=seed)
    D2, D3 = data.split_stratified(train_prop=1/2, random_state=seed)

    # recodify targets in D1 for y, and D2,D3 for A
    AD1 = np.logical_or(D1.labels==2, D1.labels==3).astype(int)
    D1 = LabelledCollection(D1.instances, np.logical_or(D1.labels==1, D1.labels==3).astype(int), n_classes=2)
    D2 = LabelledCollection(D2.instances, np.logical_or(D2.labels==2, D2.labels==3).astype(int), n_classes=2)
    D3 = LabelledCollection(D3.instances, np.logical_or(D3.labels==2, D3.labels==3).astype(int), n_classes=2)

    return D1, D2, D3, AD1


def classify(classifier:BaseEstimator, data:LabelledCollection):
    y_hat = classifier.predict(data.instances)
    pred_positives = LabelledCollection(data.instances[y_hat == 1], data.labels[y_hat == 1], n_classes=data.n_classes)
    pred_negatives = LabelledCollection(data.instances[y_hat == 0], data.labels[y_hat == 0], n_classes=data.n_classes)
    return pred_positives, pred_negatives


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

            # sanity check: prevs below should be similar, i.e.
            # prev_A1_D2_y1 \simeq prev_A1_D3_y1,
            # prev_A1_D2_y0 \simeq prev_A1_D3_y0.
            # unless we change this, MLPE should be quite accurate
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
            size_D2.append(len(sample_D2.labels))
        true_M0_A1 = [D3_y0.prevalence()[1]] * len(estim_M0_A1)
        true_M1_A1 = [D3_y1.prevalence()[1]] * len(estim_M1_A1)
        return true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, size_D2

    return vary_and_test(quantifier, D2, D2_y_hat, D3_y0, D3_y1, sample_sizes, nreps)




def eval_prevalence_variations_D3(D1, D2, D3, classifier, quantifier, sample_size=500, nprevs=101, nreps=2):
    f = classifier.fit(*D1.Xy)

    D2_y1, D2_y0 = classify(f, D2)
    D3_y1, D3_y0 = classify(f, D3)

    M1 = deepcopy(quantifier).fit(D2_y1)
    M0 = deepcopy(quantifier).fit(D2_y0)

    dict_prev = get_prevalences(D2_y0, D2_y1, D3_y0, D3_y1)
    print('Original prevalences:')
    print(f'    prev A1 in D3_y1: = {dict_prev["prev_A1_D3_y1"]:.2f}')
    print(f'    prev A1 in D3_y0: = {dict_prev["prev_A1_D3_y0"]:.2f}')

    def build_ql_report(quantifier, data: LabelledCollection):
        report = qp.evaluation.artificial_sampling_report(
            quantifier,  # the quantification method
            data,  # the test set on which the method will be evaluated
            sample_size=sample_size,  # indicates the size of samples to be drawn
            n_prevpoints=nprevs,  # how many prevalence points will be extracted from the interval [0, 1] for each category
            n_repetitions=nreps,  # number of times each prevalence will be used to generate a test sample
            n_jobs=-1,  # indicates the number of parallel workers (-1 indicates, as in sklearn, all CPUs)
            random_seed=42,  # setting a random seed allows to replicate the test samples across runs
            error_metrics='mae',
            verbose=True  # set to True to show some standard-line outputs
        )
        true_prevs = np.asarray([prev[1] for prev in report['true-prev']])
        estim_prevs = np.asarray([prev[1] for prev in report['estim-prev']])
        return true_prevs, estim_prevs

    true_M0_A1, estim_M0_A1 = build_ql_report(M0, D3_y0)
    true_M1_A1, estim_M1_A1 = build_ql_report(M1, D3_y1)

    return true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, dict_prev
