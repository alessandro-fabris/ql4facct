from typing import List
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from common import Protocol, Result
import os


def boxplot_from_dict(d, filename, xlab='x_axis', tit=''):
    plt.figure()
    df = pd.DataFrame(d)
    df.sort_values("x_axis")
    ax = plt.gca()
    ax.axhline(0, 0, 1, linestyle='--', color='k', label='optimal', zorder=0)
    sns.boxplot(x="x_axis", y="y_axis", hue="quant", data=df).set(xlabel=xlab, ylabel='Independence Gap')
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
    sns.boxplot(x=xvar, y="y_axis", hue="quant", data=df).set(xlabel=xlab)
    sns.despine(offset=10, trim=True)
    plt.title(tit)
    plt.savefig(filename)
"""


def init_result_dict():
    return {"y_axis": [], "x_axis": [], "quant": []}

def generate_plots(protocol: Protocol, outs:List[Result], plotdir='./plots'):

    os.makedirs(plotdir, exist_ok=True)
    """
    if protocol == "vary_D1_prev":
        vary_prev_D1 = init_result_dict()
        for true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, p_Aeqy, Q_name in outs:
            update_dict_prev_D1_prev(vary_prev_D1, true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, p_Aeqy, Q_name)
        boxplot_from_dict(vary_prev_D1,
                          "./plots/protD1_vary_prev_d1.pdf",
                          xlab='$\mathbb{P}(A=Y)$ in $\mathcal{D}_1$',
                          preproc="D1_prev")

    el"""
    if protocol == "vary_D1_prev_flip":
        vary_prev_D1 = init_result_dict()
        for true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, p_Aeqy, Q_name in outs:
            update_dict_prev_D1_prev_flip(vary_prev_D1, true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, p_Aeqy,
                                          Q_name)
        boxplot_from_dict(vary_prev_D1,
                          "./plots/protD1_vary_prev_d1_flip.pdf",
                          xlab='$\mathbb{P}(A=1|\hat{Y}=0) - \mathbb{P}(A=1|\hat{Y}=1)$ in $\mathcal{D}_3$ i.e. true indep. gap',
                          preproc="D1_prev_flip")

    elif protocol == Protocol.VAR_D2_SIZE:
        plot_prot2size(outs, plotdir)
    elif protocol == Protocol.VAR_D2_PREV:
        plot_prot2prev(outs, plotdir)
    elif protocol == Protocol.VAR_D3_PREV:
        plot_prot3prev(outs, plotdir)

def plot_prot1flip(outs:List[Result], plotdir='./plots'):
    vary_prev_D1 = init_result_dict()
    for true_M0_A1, true_M1_A1, estim_M0_A1, estim_M1_A1, p_Aeqy, Q_name in outs:
        update_dict_prev_D1_prev_flip(vary_prev_D1, true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, p_Aeqy,
                                      Q_name)
    boxplot_from_dict(vary_prev_D1,
                      "./plots/protD1_vary_prev_d1_flip.pdf",
                      xlab='$\mathbb{P}(A=1|\hat{Y}=0) - \mathbb{P}(A=1|\hat{Y}=1)$ in $\mathcal{D}_3$ i.e. true indep. gap',
                      preproc="D1_prev_flip")


def plot_prot2size(outs:List[Result], plotdir='./plots'):
    vary_size_D2 = init_result_dict()

    df = Result.concat(outs).data
    method_names = df['Q_name'].unique()

    for Q_name in method_names:
        Q_df = Result(df.loc[df['Q_name'] == Q_name])
        gap = Q_df.independence_gap()
        vary_size_D2["x_axis"].extend(list(Q_df.data['size_D2']))
        vary_size_D2["y_axis"].extend(list(gap))
        vary_size_D2["quant"].extend([Q_name] * len(gap))

    boxplot_from_dict(vary_size_D2,
                      os.path.join(plotdir, 'protD2_vary_size.pdf'),
                      xlab=r'$|\mathcal{D}_2|$',
                      tit='')


def plot_prot2prev(outs:List[Result], plotdir='./plots'):
    vary_prev_D20 = init_result_dict()
    vary_prev_D21 = init_result_dict()

    df = Result.concat(outs).data
    method_names = df['Q_name'].unique()

    for Q_name in method_names:
        Q_df = df.loc[df['Q_name'] == Q_name]

        var_s0 = Result(Q_df.loc[Q_df['var_s'] == 0])
        gap = var_s0.independence_gap()
        vary_prev_D20["x_axis"].extend(list(var_s0.data['D2_s0_prev']))
        vary_prev_D20["y_axis"].extend(list(gap))
        vary_prev_D20["quant"].extend([Q_name] * len(gap))
        orig_prev_D30 = var_s0.data['D3_s0_prev'].mean()

        var_s1 = Result(Q_df.loc[Q_df['var_s'] == 1])
        gap = var_s1.independence_gap()
        vary_prev_D21["x_axis"].extend(list(var_s1.data['D2_s1_prev']))
        vary_prev_D21["y_axis"].extend(list(gap))
        vary_prev_D21["quant"].extend([Q_name] * len(gap))
        orig_prev_D31 = var_s0.data['D3_s1_prev'].mean()
    title = "Test ($\mathcal{D}_3$) prevalence for quantifier: "
    boxplot_from_dict(vary_prev_D20,
                      os.path.join(plotdir, 'protD2_vary_prev_d20.pdf'),
                      xlab='$\mathbb{P}(A=1|\hat{Y}=0)$ in $\mathcal{D}_2$',
                      tit=title + f'{orig_prev_D30:.2f}')
    boxplot_from_dict(vary_prev_D21,
                      os.path.join(plotdir, 'protD2_vary_prev_d21.pdf'),
                      xlab='$\mathbb{P}(A=1|\hat{Y}=1)$ in $\mathcal{D}_2$',
                      tit=title + f'{orig_prev_D31:.2f}')


def plot_prot3prev(outs:List[Result], plotdir='./plots'):
    vary_prev_D30 = init_result_dict()
    vary_prev_D31 = init_result_dict()

    df = Result.concat(outs).data
    method_names = df['Q_name'].unique()

    for Q_name in method_names:
        Q_df = df.loc[df['Q_name'] == Q_name]

        var_s0 = Result(Q_df.loc[Q_df['var_s'] == 0])
        gap = var_s0.independence_gap()
        vary_prev_D30["x_axis"].extend(list(var_s0.data['trueD3s0A1']))
        vary_prev_D30["y_axis"].extend(list(gap))
        vary_prev_D30["quant"].extend([Q_name] * len(gap))
        orig_prev_D20 = var_s0.data['D2_s0_prev'].mean()

        var_s1 = Result(Q_df.loc[Q_df['var_s'] == 1])
        gap = var_s1.independence_gap()
        vary_prev_D31["x_axis"].extend(list(var_s1.data['trueD3s1A1']))
        vary_prev_D31["y_axis"].extend(list(gap))
        vary_prev_D31["quant"].extend([Q_name] * len(gap))
        orig_prev_D21 = var_s0.data['D2_s1_prev'].mean()

    title = "Training ($\mathcal{D}_2$) prevalence for quantifier: "
    boxplot_from_dict(vary_prev_D30,
                      os.path.join(plotdir, 'protD3_vary_prev_d30.pdf'),
                      xlab='$\mathbb{P}(A=1|\hat{Y}=0)$ in $\mathcal{D}_3$',
                      tit=title + f'{orig_prev_D20:.2f}')
    boxplot_from_dict(vary_prev_D31,
                      os.path.join(plotdir, 'protD3_vary_prev_d31.pdf'),
                      xlab='$\mathbb{P}(A=1|\hat{Y}=1)$ in $\mathcal{D}_3$',
                      tit=title + f'{orig_prev_D21:.2f}')


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
    d["y_axis"].extend(error)
    d["quant"].extend([Q_name] * len(true_variab) * len(idcs_ref_fixed))
"""

def update_dict_prev_D1_prev(d, true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, p_Aeqy, Q_name):
    assert len(true_M0_A1) == len(estim_M0_A1) == len(true_M1_A1) == len(estim_M1_A1) == len(p_Aeqy)
    error = [(e0 - e1) - (t0 - t1) for e0, e1, t0, t1 in zip(estim_M0_A1, estim_M1_A1, true_M0_A1, true_M1_A1)]
    d["x_axis"].extend(p_Aeqy)
    d["quant"].extend([Q_name] * len(p_Aeqy))
    d["y_axis"].extend(error)


def update_dict_prev_D1_prev_flip(d, true_M0_A1, estim_M0_A1, true_M1_A1, estim_M1_A1, p_Aeqy, Q_name):
    assert len(true_M0_A1) == len(estim_M0_A1) == len(true_M1_A1) == len(estim_M1_A1) == len(p_Aeqy)
    error = [(e0 - e1) - (t0 - t1) for e0, e1, t0, t1 in zip(estim_M0_A1, estim_M1_A1, true_M0_A1, true_M1_A1)]
    true_indep_gap = [t0 - t1 for t0, t1 in zip(true_M0_A1, true_M1_A1)]
    d["x_axis"].extend(true_indep_gap)
    d["quant"].extend([Q_name] * len(p_Aeqy))
    d["y_axis"].extend(error)

