import quapy as qp
import numpy as np
from quapy.data import LabelledCollection
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
from tqdm import tqdm



def split_data(X, y, A, seed):

    # jointly codify (y,A) as 0,0->0; 1,0->1; 0,1->2; 1,1->3 to stratify
    yA = y + 2 * A

    data = LabelledCollection(X, yA)
    D1, data = data.split_stratified(train_prop=1/3, random_state=seed)
    D2, D3 = data.split_stratified(train_prop=1/2, random_state=seed)

    # recodify targets in D1 for y, and D2,D3 for A
    D1 = LabelledCollection(D1.instances, np.logical_or(D1.labels==1, D1.labels==3).astype(int), n_classes=2)
    D2 = LabelledCollection(D2.instances, np.logical_or(D2.labels==2, D2.labels==3).astype(int), n_classes=2)
    D3 = LabelledCollection(D3.instances, np.logical_or(D3.labels==2, D3.labels==3).astype(int), n_classes=2)

    return D1, D2, D3


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

def eval_prevalence_variations_D1(D1, D2, D3, classifier, quantifier, sample_size=500, nprevs=101):

    true_M0_A1, true_M1_A1 = [], []
    estim_M0_A1, estim_M1_A1 = [], []

    for sample_D1 in tqdm(D1.artificial_sampling_generator(sample_size=sample_size, n_prevalences=nprevs),
                          total=nprevs, desc='training quantifiers at prevalence variations in D1'):
        if sample_D1.prevalence().prod() == 0: continue

        f = classifier.fit(*sample_D1.Xy)

        D2_y1, D2_y0 = classify(f, D2)
        D3_y1, D3_y0 = classify(f, D3)

        M1 = deepcopy(quantifier).fit(D2_y1)
        M0 = deepcopy(quantifier).fit(D2_y0)

        estim_M1_A1.append(M1.quantify(D3_y1.instances)[1])
        estim_M0_A1.append(M0.quantify(D3_y0.instances)[1])

        true_M1_A1.append(D3_y1.prevalence()[1])
        true_M0_A1.append(D3_y0.prevalence()[1])

    return compute_bias_error(true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1)


def eval_prevalence_variations_D2(D1, D2, D3, classifier, quantifier, sample_size=500, nprevs=101):
    f = classifier.fit(*D1.Xy)

    D2_y1, D2_y0 = classify(f, D2)
    D3_y1, D3_y0 = classify(f, D3)

    def vary_and_test(quantifier, D2split, D3split):
        M = deepcopy(quantifier)

        estim_M_A1 = []
        for sample_D2 in tqdm(D2split.artificial_sampling_generator(sample_size=sample_size, n_prevalences=nprevs),
                              total=nprevs, desc='training quantifiers at prevalence variations in D2'):
            if sample_D2.prevalence().prod() == 0: continue
            M.fit(sample_D2)
            estim_M_A1.append(M.quantify(D3split.instances)[1])

        true_M_A1 = [D3split.prevalence()[1]] * len(estim_M_A1)

        return true_M_A1, estim_M_A1

    true_M1_A1, estim_M1_A1 = vary_and_test(quantifier, D2_y1, D3_y1)
    true_M0_A1, estim_M0_A1 = vary_and_test(quantifier, D2_y0, D3_y0)

    return compute_bias_error(true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1)


def eval_prevalence_variations_D3(D1, D2, D3, classifier, quantifier, sample_size=500, nprevs=101):
    f = classifier.fit(*D1.Xy)

    D2_y1, D2_y0 = classify(f, D2)
    D3_y1, D3_y0 = classify(f, D3)

    M1 = deepcopy(quantifier).fit(D2_y1)
    M0 = deepcopy(quantifier).fit(D2_y0)

    def build_ql_report(quantifier, data: LabelledCollection):
        report = qp.evaluation.artificial_sampling_report(
            quantifier,  # the quantification method
            data,  # the test set on which the method will be evaluated
            sample_size=sample_size,  # indicates the size of samples to be drawn
            n_prevpoints=nprevs,  # how many prevalence points will be extracted from the interval [0, 1] for each category
            n_repetitions=1,  # number of times each prevalence will be used to generate a test sample
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

    return compute_bias_error(true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1)