from typing import List
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from common import Protocols, Result
import os
import quapy as qp

def _boxplot_from_dict(d, filename, xlab='x_axis', tit='', highlight_prevalence=None, prevalence_label=None, x2int=False):
    # sns.set(style="ticks")
    # sns.set_style("darkgrid")
    plt.figure()
    df = pd.DataFrame(d)
    df.sort_values("x_axis")
    ax = plt.gca()
    ax.axhline(0, 0, 1, linestyle='--', linewidth=2, color='k', label='optimal', zorder=0)

    ax.set_axisbelow(True)
    if x2int:
        df=df.astype({'x_axis':int})
    sns.boxplot(x="x_axis", y="y_axis", hue="quant", data=df).set(xlabel=xlab, ylabel='Estimation error')
    if highlight_prevalence is not None:
        # the training prevalence (a value in [0, 1]) has to be (linearly) transformed into the active range of the
        # sns boxplot coordinates, which can be consulted with ax.get_xlim()
        xs = df['x_axis'].unique()
        xmin,xmax = ax.get_xlim()
        xmin_real, xmax_real = min(xs) - 0.05, max(xs) + 0.05
        a = (xmax-xmin)/(xmax_real-xmin_real)
        b = xmin-a*xmin_real
        plt.axvline(highlight_prevalence * a + b, 0, 1, linestyle='-', linewidth=3, color='g', label=prevalence_label, zorder=0, alpha=0.5)
    plt.legend(loc=2, bbox_to_anchor=(1.05, 0.7), borderaxespad=0.)
    plt.title(tit)
    plt.grid(True, axis='both')
    print(f'saving figure in {filename}')
    plt.savefig(filename, bbox_inches='tight')


def _init_result_dict():
    return {"y_axis": [], "x_axis": [], "quant": []}


def generate_plots(protocol: Protocols, outs:List[Result], plotdir='./plots'):
    os.makedirs(plotdir, exist_ok=True)

    results = Result.concat(outs).select_protocol(protocol)
    results.data['Q_name'] = results.data['Q_name'].str.replace('EMQ', 'SLD')

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
        err = Q_df.independence_signed_error()

        pAeqY = np.around(Q_df.data['p_Aeqy'], decimals=2)

        vary_prev_D1["x_axis"].extend(pAeqY)
        # indep_gap = Q_df.independence_gap()
        # print(f'                indep gap: max={max(indep_gap)}, min={min(indep_gap)}')
        # indep_gap = np.around(indep_gap, decimals=1)
        # vary_prev_D1["x_axis"].extend(list(indep_gap))


        vary_prev_D1["quant"].extend([Q_name] * len(err))
        vary_prev_D1["y_axis"].extend(err)

    _boxplot_from_dict(vary_prev_D1,
                       os.path.join(plotdir, "protD1_vary_prev_d1.pdf"),
                       xlab='$Pr(Y=S)$ in $\mathcal{D}_1$')


def plot_prot1flip(results:Result, plotdir='./plots'):
    vary_prev_D1 = _init_result_dict()

    method_names = results.data['Q_name'].unique()
    for Q_name in method_names:
        Q_df = results.filter('Q_name', Q_name)
        err = Q_df.independence_signed_error()
        indep_gap = Q_df.independence_gap()

        indep_gap = np.around(indep_gap, decimals=1)
        vary_prev_D1["x_axis"].extend(list(indep_gap))
        vary_prev_D1["y_axis"].extend(list(err))
        vary_prev_D1["quant"].extend([Q_name] * len(err))

    _boxplot_from_dict(vary_prev_D1,
                       os.path.join(plotdir, 'protD1_vary_prev_d1_flip.pdf'),
                       xlab='$\Delta(s)$')


def plot_prot2size(results:Result, plotdir='./plots'):
    vary_size_D2 = _init_result_dict()

    method_names = results.data['Q_name'].unique()

    for Q_name in method_names:
        Q_df = results.filter('Q_name', Q_name)
        err = Q_df.independence_signed_error()
        vary_size_D2["x_axis"].extend(list(Q_df.data['size_D2']))
        vary_size_D2["y_axis"].extend(list(err))
        vary_size_D2["quant"].extend([Q_name] * len(err))

    _boxplot_from_dict(vary_size_D2,
                       os.path.join(plotdir, 'protD2_vary_size.pdf'),
                       xlab=r'$|\mathcal{D}_2|$',
                       tit='',
                       x2int=True)


def plot_prot2prev(results:Result, plotdir='./plots'):
    vary_prev_D20 = _init_result_dict()
    vary_prev_D21 = _init_result_dict()

    method_names = results.data['Q_name'].unique()

    for Q_name in method_names:
        Q_df = results.filter('Q_name', Q_name)

        var_s0 = Q_df.filter('var_s', 0)
        err = var_s0.independence_signed_error()
        vary_prev_D20["x_axis"].extend(list(var_s0.data['D2_s0_prev']))
        vary_prev_D20["y_axis"].extend(list(err))
        vary_prev_D20["quant"].extend([Q_name] * len(err))
        orig_prev_D30 = var_s0.data['D3_s0_prev'].mean()

        var_s1 = Q_df.filter('var_s', 1)
        err = var_s1.independence_signed_error()
        vary_prev_D21["x_axis"].extend(list(var_s1.data['D2_s1_prev']))
        vary_prev_D21["y_axis"].extend(list(err))
        vary_prev_D21["quant"].extend([Q_name] * len(err))
        orig_prev_D31 = var_s0.data['D3_s1_prev'].mean()

    # title = "Test ($\mathcal{D}_3$) prevalence for quantifier: "
    _boxplot_from_dict(vary_prev_D20,
                       os.path.join(plotdir, 'protD2_vary_prev_d20.pdf'),
                       xlab='$Pr(S=1|\hat{Y}=\ominus)$ in $\\breve{\mathcal{D}}_2^\ominus$',
                       # tit=title + f'{orig_prev_D30:.2f}',
                       highlight_prevalence=orig_prev_D30,
                       prevalence_label='$p_{\mathcal{D}_{3}^\ominus}(S=1)$')
    _boxplot_from_dict(vary_prev_D21,
                       os.path.join(plotdir, 'protD2_vary_prev_d21.pdf'),
                       xlab='$Pr(S=1|\hat{Y}=\oplus)$ in $\\breve{\mathcal{D}}_2^\oplus$',
                       # tit=title + f'{orig_prev_D31:.2f}',
                       highlight_prevalence=orig_prev_D31,
                       prevalence_label='$p_{\mathcal{D}_{3}^\oplus}(S=1)$')


def plot_prot3prev(results:Result, plotdir='./plots'):
    vary_prev_D30 = _init_result_dict()
    vary_prev_D31 = _init_result_dict()

    method_names = results.data['Q_name'].unique()

    for Q_name in method_names:
        Q_df = results.filter('Q_name', Q_name)

        var_s0 = Q_df.filter('var_s', 0)
        err = var_s0.independence_signed_error()
        vary_prev_D30["x_axis"].extend(list(var_s0.data['trueD3s0A1']))
        vary_prev_D30["y_axis"].extend(list(err))
        vary_prev_D30["quant"].extend([Q_name] * len(err))
        orig_prev_D20 = var_s0.data['D2_s0_prev'].mean()

        var_s1 = Q_df.filter('var_s', 1)
        err = var_s1.independence_signed_error()
        vary_prev_D31["x_axis"].extend(list(var_s1.data['trueD3s1A1']))
        vary_prev_D31["y_axis"].extend(list(err))
        vary_prev_D31["quant"].extend([Q_name] * len(err))
        orig_prev_D21 = var_s0.data['D2_s1_prev'].mean()

    # title = "Training ($\mathcal{D}_2$) prevalence for quantifier: "
    _boxplot_from_dict(vary_prev_D30,
                       os.path.join(plotdir, 'protD3_vary_prev_d30.pdf'),
                       xlab='$P(S=1|\hat{Y}=\ominus)$ in $\\breve{\mathcal{D}}_3^\ominus$',
                       # tit=title + f'{orig_prev_D20:.2f}',
                       highlight_prevalence=orig_prev_D20,
                       prevalence_label='$p_{\mathcal{D}_{2}^\ominus}(S=1)$')
    _boxplot_from_dict(vary_prev_D31,
                       os.path.join(plotdir, 'protD3_vary_prev_d31.pdf'),
                       xlab='$Pr(S=1|\hat{Y}=\oplus)$ in $\\breve{\mathcal{D}}_3^\oplus$',
                       # tit=title + f'{orig_prev_D21:.2f}',
                       highlight_prevalence=orig_prev_D21,
                       prevalence_label='$p_{\mathcal{D}_{2}^\oplus}(S=1)$')


def generate_plots_clf(protocol: Protocols, outs:List[Result], plotdir='./plots'):
    os.makedirs(plotdir, exist_ok=True)

    results = Result.concat(outs)
    classifiers = results.data['Clf_name'].unique()

    for clf in classifiers:
        if protocol == Protocols.VAR_D2_PREV:
            plot_prot2prev_clf(results, 'D2y0', clf, plotdir)
            plot_prot2prev_clf(results, 'D2y1', clf, plotdir)
        elif protocol == Protocols.VAR_D3_PREV:
            plot_prot3prev_clf(results, 'D3y0', clf, plotdir)
            plot_prot3prev_clf(results, 'D3y1', clf, plotdir)


def get_series(results, col_name, digitized, bins):
    means = [(results.data[col_name].values)[digitized == i].mean() for i in range(1, len(bins)+1)]
    stds  = [(results.data[col_name].values)[digitized == i].std() for i in range(1, len(bins)+1)]
    return means, stds


def get_errors(results, q_name, ground_truth, digitized, bins):
    estimations = results.data[q_name].values
    true_values = results.data[ground_truth].values
    errors = qp.error.ae(true_values.reshape(-1, 1), estimations.reshape(-1, 1))
    means = [errors[digitized == i].mean() for i in range(1, len(bins)+1)]
    stds  = [errors[digitized == i].std() for i in range(1, len(bins)+1)]
    return means, stds


def plot_prot2prev_clf(results:Result, prefix, clf, plotdir='./plots'):
    results = results.filter('prefix', prefix).filter('Clf_name', clf)

    x = results.data['trueD2sample'].values
    bins = sorted(np.unique(x))
    digitized = np.digitize(x, bins)
    x_means = [x[digitized == i].mean() for i in range(1, len(bins)+1)]

    series = [
        (f'{clf} accuracy', get_series(results, 'accs', digitized, bins)),
        (f'{clf} $F_1$', get_series(results, 'f1s', digitized, bins)),
        (f'SLD accuracy', get_series(results, 'emq_accs', digitized, bins)),
        (f'SLD $F_1$', get_series(results, 'emq_f1s', digitized, bins)),
    ]
    quantifiers = results.data.columns[6:-4].values
    for q_name in quantifiers:
        q_means, q_std = get_errors(results, q_name, 'trueD3sample', digitized, bins)
        q_name = q_name.replace('EMQ','SLD')
        series.append((f'{q_name} MAE', (q_means, q_std)))

    d3prev = np.mean(results.data['trueD3sample'].values)
    # d3prevlabel = f"{prefix.replace('2', '3')} prev"
    yindex='\oplus' if prefix[-1]=='1' else '\ominus'
    d2prevlabel = '$p_{\\breve{\mathcal{D}}_{2}^'+yindex+'}(S=1)$'
    d3prevlabel = '$p_{\mathcal{D}_{3}^'+yindex+'}(S=1)$'
    gen_plot(x_means, series,
             #title=clf + ' performance on protocol sample-prev-$\mathcal{D}_{2}^'+yindex+'$',
             title=None,
             xlabel=f'{d2prevlabel}',
             ylabel='',
             path=os.path.join(plotdir, prefix)+'.pdf',
             vline=d3prev, vlinelabel=d3prevlabel)


def plot_prot3prev_clf(results:Result, prefix, clf, plotdir='./plots'):
    results = results.filter('prefix', prefix).filter('Clf_name', clf)

    x = results.data['trueD3sample'].values
    bins = sorted(np.unique(x))
    digitized = np.digitize(x, bins)
    x_means = [x[digitized == i].mean() for i in range(1, len(bins)+1)]

    series = [
        (f'{clf} accuracy', get_series(results, 'accs', digitized, bins)),
        (f'{clf} $F_1$', get_series(results, 'f1s', digitized, bins)),
        (f'SLD accuracy', get_series(results, 'emq_accs', digitized, bins)),
        (f'SLD $F_1$', get_series(results, 'emq_f1s', digitized, bins)),
    ]
    quantifiers = results.data.columns[6:-4].values
    for q_name in quantifiers:
        q_means, q_std = get_errors(results, q_name, 'trueD3sample', digitized, bins)
        q_name = q_name.replace('EMQ', 'SLD') 
        series.append((f'{q_name} MAE', (q_means, q_std)))

    d2prev = np.mean(results.data['trueD2sample'].values)
    # d2prevlabel = f"{prefix.replace('3','2')} prev"
    yindex='\oplus' if prefix[-1]=='1' else '\ominus'
    d2prevlabel = '$p_{\mathcal{D}_{2}^' + yindex + '}(S=1)$'
    d3prevlabel = '$p_{\\breve{\mathcal{D}}_{3}^' + yindex + '}(S=1)$'
    gen_plot(x_means, series,
             # title=clf + ' performance on protocol sample-prev-$\mathcal{D}_{3}^'+yindex+'$',
             title=None,
             xlabel=f'{d3prevlabel}',
             ylabel='',
             path=os.path.join(plotdir, prefix)+'.pdf',
             vline=d2prev, vlinelabel=d2prevlabel)


def gen_plot(x_means, series, title, xlabel='', ylabel='', path=None, vline=None, vlinelabel=None):
    plt.figure()
    for label, (means,stds) in series:
        means = np.asarray(means)
        stds = np.asarray(stds)
        plt.errorbar(x_means, means, label=label)
        plt.fill_between(x_means, means - stds, means + stds, alpha=0.25)
    if vline is not None:
        plt.vlines(vline, -0.1, 1, colors='k', linestyles='dotted', label=vlinelabel)
        # plt.axvline(vline, -0.1, 1, linestyle='-', linewidth=3, color='g', label=vlinelabel, zorder=0, alpha=0.5)
    if title is not None:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(np.min(x_means), np.max(x_means))
    plt.ylim(-0.1, 1)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.legend(loc=2, bbox_to_anchor=(1.05, 0.7), borderaxespad=0.)
    plt.grid(True, axis='both')
    plt.savefig(path, bbox_inches='tight')

    # plt.savefig(path)
