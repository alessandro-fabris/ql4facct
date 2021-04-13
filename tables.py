import numpy as np
import pandas as pd
from common import Protocol, Result
from tabular import Table
import os
import pathlib


def generate_tables(protocol: Protocol, outs):
    tab_path = './tables/results.tex'

    #todo filter by protocol?

    df = Result.concat(outs).data
    method_names = df['Q_name'].unique()

    columns = ['Gap', '$|\mathrm{Error}|$', 'Error$^2$']
    table = Table(columns, list(method_names), lower_is_better=True, ttest='wilcoxon', prec_mean=3, show_std=True,
                  prec_std=3, average=False, color=False)

    for Q_name in method_names:
        results = Result(df.loc[df['Q_name'] == Q_name])
        print(f'{Q_name} has {len(results)} outs')
        gap = results.independence_gap()
        abs_error = results.independence_abs_error()
        sqr_error = results.independence_sqr_error()
        table.add('Gap', Q_name, values=gap)
        table.add('$|\mathrm{Error}|$', Q_name, values=abs_error)
        table.add('Error$^2$', Q_name, values=sqr_error)
        print(f'Q-method: {Q_name}')
        print(f'gap={gap.mean():.3f}(+-{gap.std():.3f})')
        print(f'|error|={abs_error.mean():.3f}(+-{abs_error.std():.3f})')
        print(f'error^2={sqr_error.mean():.3f}(+-{sqr_error.std():.3f})\n')

    os.makedirs(pathlib.Path(tab_path).parent, exist_ok=True)
    table.latexTable(tab_path, average=False, caption='Protocol 3', resizebox=False, method_by_benchmark=True)


