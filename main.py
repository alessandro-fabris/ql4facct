import quapy as qp
import numpy as np
from data import adultcsv_loader
from common import *
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from quapy.method.aggregative import CC, ACC, PACC, EMQ, HDy

qp.environ['SAMPLE_SIZE'] = 100

def new_cls():
    return LogisticRegression()

# def new_quantifier():
    # return qp.method.aggregative.CC(new_classifier())
    # return qp.method.aggregative.ACC(new_classifier())

def update_dict_vary_D3_prev(d, vary, true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, orig_prev, Q_name):
    if vary == "D31":
        original_preval_fixed = orig_prev["prev_A1_D3_y0"]
        true_fixed = true_M0_A1
        true_variab = true_M1_A1
        estim_fixed = estim_M0_A1
        estim_variab = estim_M1_A1
    elif vary == "D30":
        original_preval_fixed = orig_prev["prev_A1_D3_y1"]
        true_fixed = true_M1_A1
        true_variab = true_M0_A1
        estim_fixed = estim_M1_A1
        estim_variab = estim_M0_A1
    else:
        raise ValueError
    idx_ref_fixed = np.argsort([abs(t - original_preval_fixed) for t in true_fixed])[0]
    idcs_ref_fixed = [i for i, t in enumerate(true_M0_A1) if t == true_fixed[idx_ref_fixed]]
    ref_true_fixed = true_fixed[idx_ref_fixed]
    ref_estim_fixed = estim_fixed[idcs_ref_fixed] #despite being fixed, we have multiple estimates at the fixed value
    error = [(r - e) - (ref_true_fixed - t) for r in ref_estim_fixed for t,e in zip(true_variab, estim_variab)]
    if vary == "D30":
        error = [e*-1 for e in error]

    d["true_val"].extend(list(true_variab)*len(idcs_ref_fixed))
    d["estim_error"].extend(error)
    d["quant"].extend([Q_name]*len(true_variab)*len(idcs_ref_fixed))

def update_dict_vary_D2_size(d, true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, size_D2, Q_name):
    assert len(true_M0_A1) == len(estim_M0_A1) == len(true_M1_A1) == len(estim_M1_A1) == len(size_D2)
    error = [(e0 - e1) - (t0 - t1) for e0, e1, t0, t1 in  zip(estim_M0_A1, estim_M1_A1, true_M0_A1, true_M1_A1)]
    d["size_D2"].extend(size_D2)
    d["quant"].extend([Q_name] * len(size_D2))
    d["estim_error"].extend(error)

def boxplot_from_dict(d, filename, xlab='true_val', tit='', xvar=None, preproc=None):
    plt.figure()
    if preproc is None:
        df = pd.DataFrame(d)
    elif preproc == "D3_prev":
        d["true_val_q"] = [np.round(t * 100) / 100 for t in d["true_val"]]
        df = pd.DataFrame(d)
        df.sort_values("true_val_q", inplace=True)
        xvar = "true_val_q"

    sns.boxplot(x=xvar, y="estim_error", hue="quant", data=df).set(xlabel=xlab)
    sns.despine(offset=10, trim=True)
    plt.title(tit)
    plt.savefig(filename)


# protocol = "vary_D3_prev"
protocol = "vary_D2_size"


X, y, A = adultcsv_loader("./adult.csv", protected_attr='gender')
# Notice P(A=0, y=1)<0.04, i.e. rich women are rare.
# This is a bottleneck for quantification, e.g. in the vary_D2_size protocol, when |D2|=1000 there are ~37 women in D21
# making life pretty hard for M1.
D1, D2, D3 = split_data(X, y, A, seed=0)

f = new_cls()

if protocol == "vary_D3_prev":
    vary_prev_D31 = {"estim_error": [], "true_val": [], "quant": []}
    vary_prev_D30 = {"estim_error": [], "true_val": [], "quant": []}
elif protocol == "vary_D2_size":
    # not stratified in any way (challenging but realistic)
    # currently slow TODO: make faster
    vary_size_D2 = {"estim_error": [], "size_D2": [], "quant": []}

for Q in [CC(new_cls()), ACC(new_cls()), PACC(new_cls()), EMQ(new_cls()), HDy(new_cls())]:
    Q_name = Q.__class__.__name__
    if protocol == "vary_D3_prev":
        true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, orig_prev = eval_prevalence_variations_D3(D1, D2, D3, f, Q,
                                                                                                    nprevs=11, nreps=20)
        update_dict_vary_D3_prev(vary_prev_D31, "D31", true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, orig_prev, Q_name)
        update_dict_vary_D3_prev(vary_prev_D30, "D30", true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, orig_prev, Q_name)
    elif protocol == "vary_D2_size":
        true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, size_D2 = eval_size_variations_D2(D1, D2, D3, f, Q,
                                                                                            nreps=10, sample_sizes=None)
        update_dict_vary_D2_size(vary_size_D2, true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, size_D2, Q_name)

    # true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, orig_prev = eval_prevalence_variations_D1(D1, D2, D3, f, Q, nprevs=11)
    # true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, orig_prev = eval_prevalence_variations_D2(D1, D2, D3, f, Q, nprevs=11)
    bias_ave, bias_std, error_ave, error_std = compute_bias_error(true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1)
    print('Classifier: ', f.__class__.__name__)
    print('Quantifier: ', Q_name)
    print(f'bias = {bias_ave:.5f}+-{bias_std:.5f}')
    print(f'error = {error_ave:.5f}+-{error_std:.5f}')

# plots
if protocol == "vary_D3_prev":
    title = "Training ($\mathcal{D}_2$) prevalence for quantifier: "
    boxplot_from_dict(vary_prev_D30, "./plots/protD3_vary_prev_d30.pdf", xlab='$\mathbb{P}(A=1|\hat{Y}=0)$ in $\mathcal{D}_3$',
                      tit=title+f'{orig_prev["prev_A1_D2_y0"]:.2f}', preproc="D3_prev")
    boxplot_from_dict(vary_prev_D31, "./plots/protD3_vary_prev_d31.pdf", xlab='$\mathbb{P}(A=1|\hat{Y}=1)$ in $\mathcal{D}_3$',
                      tit=title+f'{orig_prev["prev_A1_D2_y1"]:.2f}', preproc="D3_prev")
if protocol == "vary_D2_size":
    boxplot_from_dict(vary_size_D2, "./plots/protD2_vary_size.pdf", xlab=r'$|\mathcal{D}_2|$', tit='',  xvar="size_D2")





