from typing import List
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from common import Protocols, Result
import os


def _boxplot_from_dict(d, filename, xlab='x_axis', tit=''):
    plt.figure()
    df = pd.DataFrame(d)
    df.sort_values("x_axis")
    ax = plt.gca()
    ax.axhline(0, 0, 1, linestyle='--', color='k', label='optimal', zorder=0)
    sns.boxplot(x="x_axis", y="y_axis", hue="quant", data=df).set(xlabel=xlab, ylabel='Independence Gap')
    sns.despine(offset=10, trim=True)
    plt.title(tit)
    plt.savefig(filename)


def _init_result_dict():
    return {"y_axis": [], "x_axis": [], "quant": []}


def generate_plots(protocol: Protocols, outs:List[Result], plotdir='./plots'):
    os.makedirs(plotdir, exist_ok=True)

    results = Result.concat(outs).select_protocol(protocol)

    if protocol == Protocols.VAR_D1_PREV:
        plot_prot1prev(results, plotdir)
    elif protocol == Protocols.VAR_D1_PREVFLIP:
        plot_prot1flip(results, plotdir)
    elif protocol == Protocols.VAR_D2_SIZE:
        plot_prot2size(results, plotdir)
    elif protocol == Protocols.VAR_D2_PREV:
        plot_prot2prev(results, plotdir)
    elif protocol == Protocols.VAR_D3_PREV:
        plot_prot3prev(results, plotdir)


def plot_prot1prev(results: Result, plotdir='./plots'):
    vary_prev_D1 = _init_result_dict()

    method_names = results.data['Q_name'].unique()
    for Q_name in method_names:
        Q_df = results.filter('Q_name', Q_name)
        gap = Q_df.independence_gap()

        pAeqY = np.around(Q_df.data['p_Aeqy'], decimals=2)
        vary_prev_D1["x_axis"].extend(pAeqY)
        vary_prev_D1["quant"].extend([Q_name] * len(gap))
        vary_prev_D1["y_axis"].extend(gap)

    _boxplot_from_dict(vary_prev_D1,
                       os.path.join(plotdir, "protD1_vary_prev_d1.pdf"),
                       xlab='$\mathbb{P}(A=Y)$ in $\mathcal{D}_1$')


def plot_prot1flip(results:Result, plotdir='./plots'):
    vary_prev_D1 = _init_result_dict()

    method_names = results.data['Q_name'].unique()
    for Q_name in method_names:
        Q_df = results.filter('Q_name', Q_name)
        gap = Q_df.independence_gap()
        indep = Q_df.independence_true()

        indep = np.around(indep, decimals=1)
        vary_prev_D1["x_axis"].extend(list(indep))
        vary_prev_D1["y_axis"].extend(list(gap))
        vary_prev_D1["quant"].extend([Q_name] * len(gap))

    _boxplot_from_dict(vary_prev_D1,
                       os.path.join(plotdir, 'protD1_vary_prev_d1_flip.pdf'),
                       xlab='$\mathbb{P}(A=1|\hat{Y}=0) - \mathbb{P}(A=1|\hat{Y}=1)$ in $\mathcal{D}_3$ '
                           'i.e. true indep. gap')


def plot_prot2size(results:Result, plotdir='./plots'):
    vary_size_D2 = _init_result_dict()

    method_names = results.data['Q_name'].unique()

    for Q_name in method_names:
        Q_df = results.filter('Q_name', Q_name)
        gap = Q_df.independence_gap()
        vary_size_D2["x_axis"].extend(list(Q_df.data['size_D2']))
        vary_size_D2["y_axis"].extend(list(gap))
        vary_size_D2["quant"].extend([Q_name] * len(gap))

    _boxplot_from_dict(vary_size_D2,
                       os.path.join(plotdir, 'protD2_vary_size.pdf'),
                       xlab=r'$|\mathcal{D}_2|$',
                       tit='')


def plot_prot2prev(results:Result, plotdir='./plots'):
    vary_prev_D20 = _init_result_dict()
    vary_prev_D21 = _init_result_dict()

    method_names = results.data['Q_name'].unique()

    for Q_name in method_names:
        Q_df = results.filter('Q_name', Q_name)

        var_s0 = Q_df.filter('var_s', 0)
        gap = var_s0.independence_gap()
        vary_prev_D20["x_axis"].extend(list(var_s0.data['D2_s0_prev']))
        vary_prev_D20["y_axis"].extend(list(gap))
        vary_prev_D20["quant"].extend([Q_name] * len(gap))
        orig_prev_D30 = var_s0.data['D3_s0_prev'].mean()

        var_s1 = Q_df.filter('var_s', 1)
        gap = var_s1.independence_gap()
        vary_prev_D21["x_axis"].extend(list(var_s1.data['D2_s1_prev']))
        vary_prev_D21["y_axis"].extend(list(gap))
        vary_prev_D21["quant"].extend([Q_name] * len(gap))
        orig_prev_D31 = var_s0.data['D3_s1_prev'].mean()

    title = "Test ($\mathcal{D}_3$) prevalence for quantifier: "
    _boxplot_from_dict(vary_prev_D20,
                       os.path.join(plotdir, 'protD2_vary_prev_d20.pdf'),
                       xlab='$\mathbb{P}(A=1|\hat{Y}=0)$ in $\mathcal{D}_2$',
                       tit=title + f'{orig_prev_D30:.2f}')
    _boxplot_from_dict(vary_prev_D21,
                       os.path.join(plotdir, 'protD2_vary_prev_d21.pdf'),
                       xlab='$\mathbb{P}(A=1|\hat{Y}=1)$ in $\mathcal{D}_2$',
                       tit=title + f'{orig_prev_D31:.2f}')


def plot_prot3prev(results:Result, plotdir='./plots'):
    vary_prev_D30 = _init_result_dict()
    vary_prev_D31 = _init_result_dict()

    method_names = results.data['Q_name'].unique()

    for Q_name in method_names:
        Q_df = results.filter('Q_name', Q_name)

        var_s0 = Q_df.filter('var_s', 0)
        gap = var_s0.independence_gap()
        vary_prev_D30["x_axis"].extend(list(var_s0.data['trueD3s0A1']))
        vary_prev_D30["y_axis"].extend(list(gap))
        vary_prev_D30["quant"].extend([Q_name] * len(gap))
        orig_prev_D20 = var_s0.data['D2_s0_prev'].mean()

        var_s1 = Q_df.filter('var_s', 1)
        gap = var_s1.independence_gap()
        vary_prev_D31["x_axis"].extend(list(var_s1.data['trueD3s1A1']))
        vary_prev_D31["y_axis"].extend(list(gap))
        vary_prev_D31["quant"].extend([Q_name] * len(gap))
        orig_prev_D21 = var_s0.data['D2_s1_prev'].mean()

    title = "Training ($\mathcal{D}_2$) prevalence for quantifier: "
    _boxplot_from_dict(vary_prev_D30,
                       os.path.join(plotdir, 'protD3_vary_prev_d30.pdf'),
                       xlab='$\mathbb{P}(A=1|\hat{Y}=0)$ in $\mathcal{D}_3$',
                       tit=title + f'{orig_prev_D20:.2f}')
    _boxplot_from_dict(vary_prev_D31,
                       os.path.join(plotdir, 'protD3_vary_prev_d31.pdf'),
                       xlab='$\mathbb{P}(A=1|\hat{Y}=1)$ in $\mathcal{D}_3$',
                       tit=title + f'{orig_prev_D21:.2f}')

