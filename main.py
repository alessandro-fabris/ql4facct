import pathlib
import os
from os.path import join
from sklearn.linear_model import LogisticRegression
from data import adultcsv_loader
from common import *
from quapy.method.aggregative import CC, ACC, PACC, EMQ, HDy
from common import Protocols
from tabular import generate_tables
from plot import generate_plots



# Define all hyper-parameters here
# --------------------------------------------

qp.environ['SAMPLE_SIZE'] = 100

model_selection = False
datasplit_repetitions = 1
data_path = "datasets/adult.csv"
protected_attr = "gender"
options = {
    'nprevs': 11,
    'nreps': 5,
    'sample_size': 500
}
result_dir = './results'
table_dir = './tables'
plot_dir = './plots'

skip_already_computed = True  # set to False to force re-generation of experiments

f = LogisticRegression()
fname = 'LR'


# Define the classifiers we would like to test
# --------------------------------------------
def classifiers():
    hyperparams = {'C': np.logspace(-3,3,7), 'class_weight': ['balanced', None]}
    yield 'LR', LogisticRegression(), hyperparams


# Define the quantifiers we would like to test
# --------------------------------------------
def quantifiers():
    yield 'CC', CC
    # yield 'ACC', ACC
    yield 'PACC', PACC
    yield 'EMQ', EMQ
    # yield 'HDy', HDy


# instantiate all quantiiers x classifiers (wrapped also within model selection if requested)
def iter_quantifiers(model_selection=True):
    for (c_name, c, hyper), (q_name, q) in itertools.product(classifiers(), quantifiers()):
        name = f'{q_name}({c_name})'
        q = q(c)
        if model_selection:
            q = qp.model_selection.GridSearchQ(
                model=q,
                param_grid=hyper,
                sample_size=qp.environ['SAMPLE_SIZE'],
                eval_budget=500,
                error='mae',
                refit=True,  # retrain on the whole labelled set
                val_split=0.4,
                verbose=False  # show information as the process goes on
            )
        yield name, q


def dataset_name():
    return pathlib.Path(data_path).name.replace('.csv', '')+'_'+protected_attr


def run_name():
    options_par = '_'.join(f'{key}{options[key]}' for key in sorted(options.keys()))
    return f'{dataset_name()}_{Q_name}_f{fname}_Run{run}_{protocol}_{options_par}.pkl'


# --------------------------------------------
# Compute the experiments
# --------------------------------------------

X, y, A = adultcsv_loader(data_path, protected_attr=protected_attr)

os.makedirs(result_dir, exist_ok=True)
os.makedirs(table_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

results = []
for run, (D1, D2, D3, AD1) in enumerate(gen_split_data(X, y, A, repetitions=datasplit_repetitions)):
    pbar = tqdm(itertools.product(iter_quantifiers(model_selection), Protocols))
    for (Q_name, Q), protocol in pbar:
        pbar.set_description(f'{protocol}: {Q_name}')

        result_path = os.path.join(result_dir, run_name())
        if skip_already_computed and os.path.exists(result_path):
            print(f'skipping {result_path}; already computed!')
            results.append(Result.load(result_path))
            continue

        if protocol == Protocols.VAR_D1_PREV:
            outs = eval_prevalence_variations_D1(D1, D2, D3, AD1, f, Q, **options)
        elif protocol == Protocols.VAR_D1_PREVFLIP:
            outs = eval_prevalence_variations_D1_flip(D1, D2, D3, AD1, f, Q, **options)
        elif protocol == Protocols.VAR_D2_SIZE:
            outs = eval_size_variations_D2(D1, D2, D3, f, Q, sample_sizes=None, nreps=options['nreps'])
        elif protocol == Protocols.VAR_D2_PREV:
            outs = eval_prevalence_variations_D2(D1, D2, D3, f, Q, **options)
        elif protocol == Protocols.VAR_D3_PREV:
            outs = eval_prevalence_variations_D3(D1, D2, D3, f, Q, **options)

        outs.set('Q_name', Q_name)
        outs.set('run', run)
        outs.save(result_path)
        results.append(outs)


# -------------------------
# Generate tables and plots
# -------------------------
for protocol in Protocols:
    generate_tables(protocol, results, table_path=join(table_dir, f'tab_{protocol}_{dataset_name()}.tex'))
    generate_plots(protocol, results, plotdir=join(plot_dir, dataset_name()))

