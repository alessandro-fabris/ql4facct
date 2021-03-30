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

def update_dict(d, vary, true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, orig_prev, Q_name):
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

    d["true_val"].extend(list(true_variab)*len(idcs_ref_fixed))
    d["estim_error"].extend([ref_true_fixed - t - (r - e) for r in ref_estim_fixed for t,e in zip(true_variab, estim_variab)])
    d["quant"].extend([Q_name]*len(true_variab)*len(idcs_ref_fixed))


def boxplot_from_dict(d, filename, xlab='true_val', tit=''):
    plt.figure()
    d["true_val_q"] = [np.round(t * 100) / 100 for t in d["true_val"]]
    df = pd.DataFrame(d)
    df.sort_values("true_val_q", inplace=True)

    sns.boxplot(x="true_val_q", y="estim_error", hue="quant", data=df).set(xlabel=xlab)
    sns.despine(offset=10, trim=True)
    plt.title(tit)
    plt.savefig(filename)


protocol = "vary_prev_D3"

X, y, A = adultcsv_loader("./adult.csv", protected_attr='gender')

D1, D2, D3 = split_data(X, y, A, seed=0)

f = new_cls()

if protocol == "vary_prev_D3":
    vary_prev_D31 = {"estim_error": [], "true_val": [], "quant": []}
    vary_prev_D30 = {"estim_error": [], "true_val": [], "quant": []}

for Q in [CC(new_cls()), ACC(new_cls()), PACC(new_cls()), EMQ(new_cls()), HDy(new_cls())]:
    # true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, orig_prev = eval_prevalence_variations_D1(D1, D2, D3, f, Q, nprevs=11)
    # true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, orig_prev = eval_prevalence_variations_D2(D1, D2, D3, f, Q, nprevs=11)
    bias_ave, bias_std, error_ave, error_std = compute_bias_error(true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1)
    print('Classifier: ', f.__class__.__name__)
    print('Quantifier: ', Q.__class__.__name__)
    print(f'bias = {bias_ave:.5f}+-{bias_std:.5f}')
    print(f'error = {error_ave:.5f}+-{error_std:.5f}')

    if protocol == "vary_prev_D3":
        true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, orig_prev = eval_prevalence_variations_D3(D1, D2, D3, f, Q,
                                                                                                    nprevs=11, nreps=20)
        update_dict(vary_prev_D31, "D31", true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, orig_prev, Q.__class__.__name__)
        update_dict(vary_prev_D30, "D30", true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, orig_prev, Q.__class__.__name__)

# plots
if protocol == "vary_prev_D3":
    title = "Training (D2) prevalence for quantifier: "
    boxplot_from_dict(vary_prev_D30, "./plots/protD3_vary_prev_d30.pdf", xlab=r'$\mathbb{P}(A=1|\hat{Y}=0)$',
                      tit=title+f'{orig_prev["prev_A1_D2_y0"]:.2f}')
    boxplot_from_dict(vary_prev_D31, "./plots/protD3_vary_prev_d31.pdf", xlab=r'$\mathbb{P}(A=1|\hat{Y}=1)$',
                      tit=title+f'{orig_prev["prev_A1_D2_y1"]:.2f}')






