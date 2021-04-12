import numpy as np
import pandas as pd
from common import Protocol, Result


def generate_tables(protocol: Protocol, outs):

    #todo filter by protocol?

    method_names = frozenset(r.Q_name for r in outs)
    methods = {
        Q_name: Result.concatenate([r for r in outs if r.Q_name==Q_name]) for Q_name in method_names
    }

    for Q_name, results in methods.items():
        print(f'{Q_name} has {len(results)} outs')
        gap = results.independence_gap()
        abs_error = results.independence_abs_error()
        sqr_error = results.independence_sqr_error()
        print(f'Q-method: {Q_name}')
        print(f'gap={gap.mean():.3f}(+-{gap.std():.3f})')
        print(f'|error|={abs_error.mean():.3f}(+-{abs_error.std():.3f})')
        print(f'error^2={sqr_error.mean():.3f}(+-{sqr_error.std():.3f})')


