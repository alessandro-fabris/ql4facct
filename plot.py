from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from common import Protocol, Result


def boxplot_from_dict(d, filename, xlab='x_axis', tit=''):
    plt.figure()
    df = pd.DataFrame(d)
    df.sort_values("x_axis")
    ax = plt.gca()
    ax.axhline(0, 0, 1, linestyle='--', color='k', label='optimal', zorder=0)
    sns.boxplot(x="x_axis", y="estim_error", hue="quant", data=df).set(xlabel=xlab, ylabel='Independence Gap')
    sns.despine(offset=10, trim=True)
    plt.title(tit)
    plt.savefig(filename)

"""
def boxplot_from_dict_(d, filename, xlab='x_axis', tit='', xvar=None, preproc=None):
    plt.figure()
    if preproc is None:
        df = pd.DataFrame(d)
        xvar = "x_axis"
    elif preproc == "D3_prev" or preproc == "D1_prev" or preproc == "D1_prev_flip":
        if preproc == "D1_prev_flip":
            d["x_axis_q"] = [np.round(t * 10) / 10 for t in d["x_axis"]]
        else:
            d["x_axis_q"] = [np.round(t * 100) / 100 for t in d["x_axis"]]
        df = pd.DataFrame(d)
        df.sort_values("x_axis_q", inplace=True)
        xvar = "x_axis_q"

    ax = plt.gca()
    ax.axhline(0, 0, 1, linestyle='--', color='k', label='optimal', zorder=0)
    sns.boxplot(x=xvar, y="estim_error", hue="quant", data=df).set(xlabel=xlab)
    sns.despine(offset=10, trim=True)
    plt.title(tit)
    plt.savefig(filename)
"""


def generate_plots(protocol: Protocol, outs:List[Result]):

    def init_result_dict():
        return {"estim_error": [], "x_axis": [], "quant": []}

    """
    if protocol == "vary_D1_prev":
        vary_prev_D1 = init_result_dict()
        for true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, p_Aeqy, Q_name in outs:
            update_dict_prev_D1_prev(vary_prev_D1, true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, p_Aeqy, Q_name)
        boxplot_from_dict(vary_prev_D1,
                          "./plots/protD1_vary_prev_d1.pdf",
                          xlab='$\mathbb{P}(A=Y)$ in $\mathcal{D}_1$',
                          preproc="D1_prev")

    elif protocol == "vary_D1_prev_flip":
        vary_prev_D1 = init_result_dict()
        for true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, p_Aeqy, Q_name in outs:
            update_dict_prev_D1_prev_flip(vary_prev_D1, true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, p_Aeqy,
                                          Q_name)
        boxplot_from_dict(vary_prev_D1,
                          "./plots/protD1_vary_prev_d1_flip.pdf",
                          xlab='$\mathbb{P}(A=1|\hat{Y}=0) - \mathbb{P}(A=1|\hat{Y}=1)$ in $\mathcal{D}_3$ i.e. true indep. gap',
                          preproc="D1_prev_flip")

    elif protocol == "vary_D2_size":
        vary_size_D2 = init_result_dict()
        for true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, size_D2, Q_name in outs:
            update_dict_vary_D2_size(vary_size_D2, true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, size_D2, Q_name)
        boxplot_from_dict(vary_size_D2,
                          "./plots/protD2_vary_size.pdf",
                          xlab=r'$|\mathcal{D}_2|$',
                          tit='')

    elif protocol == "vary_D2_prev":
        vary_prev_D20 = init_result_dict()
        vary_prev_D21 = init_result_dict()
        for true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, prev_D2_y0, prev_D2_y1, orig_prev, Q_name in outs:
            update_dicts_vary_D2_prev(vary_prev_D20, vary_prev_D21, true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1,
                                      prev_D2_y0, prev_D2_y1, orig_prev, Q_name)
        title = "Test ($\mathcal{D}_3$) prevalence for quantifier: "
        boxplot_from_dict(vary_prev_D20,
                          "./plots/protD2_vary_prev_d20.pdf",
                          xlab='$\mathbb{P}(A=1|\hat{Y}=0)$ in $\mathcal{D}_2$',
                          tit=title + f'{orig_prev["prev_A1_D3_y0"]:.2f}',
                          preproc="D3_prev")
        boxplot_from_dict(vary_prev_D21,
                          "./plots/protD2_vary_prev_d21.pdf",
                          xlab='$\mathbb{P}(A=1|\hat{Y}=1)$ in $\mathcal{D}_2$',
                          tit=title + f'{orig_prev["prev_A1_D3_y1"]:.2f}',
                          preproc="D3_prev")
    """
    if protocol == Protocol.VAR_D3_PREV:
        vary_prev_D30 = init_result_dict()
        vary_prev_D31 = init_result_dict()
        for out_i in outs:
            if out_i.run > 0: continue
            Q_name = out_i.Q_name

            var_s0 = out_i.from_slice(out_i.var_s0)
            gap = var_s0.independence_gap()
            vary_prev_D30["x_axis"].extend(list(var_s0.true_D3_s0_A1))
            vary_prev_D30["estim_error"].extend(list(gap))
            vary_prev_D30["quant"].extend([Q_name] * len(gap))
            orig_prev_D20 = out_i.D2_s0_prev

            var_s1 = out_i.from_slice(out_i.var_s1)
            gap = var_s1.independence_gap()
            vary_prev_D31["x_axis"].extend(list(var_s1.true_D3_s1_A1))
            vary_prev_D31["estim_error"].extend(list(gap))
            vary_prev_D31["quant"].extend([Q_name] * len(gap))
            orig_prev_D21 = out_i.D2_s1_prev

        title = "Training ($\mathcal{D}_2$) prevalence for quantifier: "
        boxplot_from_dict(vary_prev_D30,
                          "./plots/protD3_vary_prev_d30_refactor.pdf",
                          xlab='$\mathbb{P}(A=1|\hat{Y}=0)$ in $\mathcal{D}_3$',
                          tit=title + f'{orig_prev_D20:.2f}')
        boxplot_from_dict(vary_prev_D31,
                          "./plots/protD3_vary_prev_d31_refactor.pdf",
                          xlab='$\mathbb{P}(A=1|\hat{Y}=1)$ in $\mathcal{D}_3$',
                          tit=title + f'{orig_prev_D21:.2f}')


#def update_dicts_vary_D3_prev(d30, d31, true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, orig_prev, Q_name):
#    update_dict_vary_D3_prev(d31, "D31", true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, orig_prev, Q_name)
#    update_dict_vary_D3_prev(d30, "D30", true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, orig_prev, Q_name)

"""
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
    idcs_ref_fixed = [i for i, t in enumerate(true_fixed) if t == true_fixed[idx_ref_fixed]]
    ref_true_fixed = true_fixed[idx_ref_fixed]
    ref_estim_fixed = estim_fixed[idcs_ref_fixed]  # despite being fixed, we have multiple estimates at the fixed value
    error = [(r - e) - (ref_true_fixed - t) for r in ref_estim_fixed for t, e in zip(true_variab, estim_variab)]
    if vary == "D30":
        error = [e * -1 for e in error]
    d["x_axis"].extend(list(true_variab) * len(idcs_ref_fixed))
    d["estim_error"].extend(error)
    d["quant"].extend([Q_name] * len(true_variab) * len(idcs_ref_fixed))
"""

def update_dict_vary_D2_size(d, true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, size_D2, Q_name):
    assert len(true_M0_A1) == len(estim_M0_A1) == len(true_M1_A1) == len(estim_M1_A1) == len(size_D2)
    error = [(e0 - e1) - (t0 - t1) for e0, e1, t0, t1 in zip(estim_M0_A1, estim_M1_A1, true_M0_A1, true_M1_A1)]
    d["x_axis"].extend(size_D2)
    d["quant"].extend([Q_name] * len(size_D2))
    d["estim_error"].extend(error)


def update_dict_prev_D1_prev(d, true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, p_Aeqy, Q_name):
    assert len(true_M0_A1) == len(estim_M0_A1) == len(true_M1_A1) == len(estim_M1_A1) == len(p_Aeqy)
    error = [(e0 - e1) - (t0 - t1) for e0, e1, t0, t1 in zip(estim_M0_A1, estim_M1_A1, true_M0_A1, true_M1_A1)]
    d["x_axis"].extend(p_Aeqy)
    d["quant"].extend([Q_name] * len(p_Aeqy))
    d["estim_error"].extend(error)


def update_dict_prev_D1_prev_flip(d, true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, p_Aeqy, Q_name):
    assert len(true_M0_A1) == len(estim_M0_A1) == len(true_M1_A1) == len(estim_M1_A1) == len(p_Aeqy)
    error = [(e0 - e1) - (t0 - t1) for e0, e1, t0, t1 in zip(estim_M0_A1, estim_M1_A1, true_M0_A1, true_M1_A1)]
    true_indep_gap = [t0 - t1 for t0, t1 in zip(true_M0_A1, true_M1_A1)]
    d["x_axis"].extend(true_indep_gap)
    d["quant"].extend([Q_name] * len(p_Aeqy))
    d["estim_error"].extend(error)


def update_dicts_vary_D2_prev(d20, d21, true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, prev_D2_y0, prev_D2_y1,
                              orig_prev, Q_name):
    update_dict_vary_D2_prev(d20, "D20", true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, prev_D2_y0,
                             prev_D2_y1, orig_prev, Q_name)
    update_dict_vary_D2_prev(d21, "D21", true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, prev_D2_y0,
                             prev_D2_y1, orig_prev, Q_name)


def update_dict_vary_D2_prev(d, vary, true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, prev_D2_y0_sample,
                             prev_D2_y1_sample, orig_prev, Q_name):
    assert (all(true_M1_A1 == true_M1_A1[0]))
    assert (all(true_M0_A1 == true_M0_A1[0]))
    if vary == "D21":
        original_preval_train_fixed = orig_prev["prev_A1_D2_y0"]
        train_fixed = prev_D2_y0_sample
        train_variab = prev_D2_y1_sample
        test_true_fixed = true_M0_A1[0]
        test_true_variab = true_M1_A1[0]
        test_estim_fixed = estim_M0_A1
        test_estim_variab = estim_M1_A1
    elif vary == "D20":
        original_preval_train_fixed = orig_prev["prev_A1_D2_y1"]
        train_fixed = prev_D2_y1_sample
        train_variab = prev_D2_y0_sample
        test_true_fixed = true_M1_A1[0]
        test_true_variab = true_M0_A1[0]
        test_estim_fixed = estim_M1_A1
        test_estim_variab = estim_M0_A1
    else:
        raise ValueError
    idx_ref_fixed = np.argsort([abs(t - original_preval_train_fixed) for t in train_fixed])[0]
    idcs_ref_fixed = [i for i, t in enumerate(train_fixed) if t == train_fixed[idx_ref_fixed]]
    ref_estim_fixed = [test_estim_fixed[i] for i in idcs_ref_fixed]
    error = [(ef - ev) - (test_true_fixed - test_true_variab) for ef in ref_estim_fixed for ev in test_estim_variab]
    if vary == "D20":
        error = [e * -1 for e in error]
    d["x_axis"].extend(list(train_variab) * len(idcs_ref_fixed))
    d["estim_error"].extend(error)
    d["quant"].extend([Q_name] * len(train_variab) * len(idcs_ref_fixed))