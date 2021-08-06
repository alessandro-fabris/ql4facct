import pathlib
import os
from os.path import join

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from data import adultcsv_loader, compascsv_loader, ccdefaultcsv_loader
from common import *
from method import *
from quapy.method.aggregative import CC, PCC, ACC, PACC, EMQ, HDy
from common import Protocols
from tabular import generate_tables_joindatasets, generate_tables
from plot import generate_plots


# Define all hyper-parameters here
# --------------------------------------------

qp.environ['SAMPLE_SIZE'] = 100
model_selection = False
datasplit_repetitions = 5
options = {
    'nprevs': 11,
    'nreps': 10,
    'sample_size': 500,
    'splitD2': True
}
result_dir = './results'
table_dir = './tables'
plot_dir = './plots'

skip_already_computed = True  # set to False to force re-generation of experiments

fclassweight='balanced'
f = LogisticRegression(class_weight=fclassweight)
fname = 'LR'
include_noSplitD2 = True
if include_noSplitD2:
    plot_dir+='-ablation'


# classifier_name = 'LR'
classifier_name = 'SVM'

if classifier_name=='LR':
    classifier = LogisticRegression()
elif classifier_name=='SVM':
    classifier = LinearSVC()
else:
    raise ValueError('unknown classifier name', classifier_name)

plot_dir = join(plot_dir, classifier_name)


# Define the quantifiers we would like to test
# --------------------------------------------
def quantifiers():
    yield 'CC', CC
    # yield 'PCC', PCC
    # yield 'ACC', ACC
    yield 'PACC', PACC
    yield 'EMQ', EMQ
    # yield 'HDy', HDy


# Define the independence-gap Estimators we would like to test
# --------------------------------------------
def estimators():
    yield 'Dummy', DummyIGE()
    # yield 'NSA', NaturalSamplingAdjustment(PACC(LogisticRegression()))
    #yield 'ASA', ArtificialSamplingAdjustment(PACC(LogisticRegression()))
    #yield 'ASAem', ArtificialSamplingAdjustment(EMQ(LogisticRegression()))
    # yield 'ASAemGP', ArtificialSamplingAdjustment(EMQ(LogisticRegression())) # with k-nn regression
    # yield 'ASAemSVR_gs', ArtificialSamplingAdjustment(EMQ(LogisticRegression()))  # with k-nn regression
    #yield 'ASAem2in', ArtificialSamplingAdjustment(EMQ(LogisticRegression()))
    # yield 'ASE2GP', ArtificialSamplingEnsambleAdjustment() #pacc+emq
    # yield 'ASE2SVR_new', ArtificialSamplingEnsambleAdjustment()  # pacc+emq more efficient
    # yield 'ASE2SVRrefit', ArtificialSamplingEnsambleAdjustment(n_repetitions=5000, refit=True)  # pacc+emq
    yield 'ASE2SVRNoRefit', ArtificialSamplingEnsambleAdjustment(n_repetitions=5000, refit=False)  # pacc+emq
    #yield 'ASEm', ArtificialSamplingEnsambleAdjustment() # with all aggregatives
    #yield 'ASAem5', ArtificialSamplingAdjustment(EMQ(LogisticRegression()), n_repetitions=5000)
    #yield 'MSA', MixSamplingAdjustment(EMQ(LogisticRegression()), n_repetitions=1000)


# Define the datasets we would like to test
# --------------------------------------------
def datasets():
    yield 'adult', "datasets/adult2.csv", adultcsv_loader, "gender"
    # yield 'compas', "datasets/compas-scores-two-years.csv", compascsv_loader, "race"
    # yield 'cc_default', "datasets/default of credit card clients.csv", ccdefaultcsv_loader, "SEX"


# instantiate all quantifiers x classifiers (wrapped also within model selection if requested)
def iter_methods():
    for (q_name, q) in quantifiers():
        c_name = classifier_name
        c = classifier
        name = f'{q_name}({c_name})'
        q = q(c)
        yield name, q
    # for (e_name, e) in estimators():
    #     yield e_name, e


def run_name():
    options_par = '_'.join(f'{key}{options[key]}' for key in sorted(options.keys()))
    options_par = options_par.replace('_splitD2True', '')
    options_par = options_par.replace('_splitD2False', '')
    fmode = '' if fclassweight is None else '_fclassweight=balanced'
    # splitD2str = '-nosD2' if (not options['splitD2'] and isinstance(Q, BaseQuantifier)) else ''
    return f'{dataset_name}_{Q_name}_f{fname}_Run{run}_{protocol}_{options_par}_modsel{False}{fmode}.pkl'


# --------------------------------------------
# Compute the experiments
# --------------------------------------------

os.makedirs(result_dir, exist_ok=True)
os.makedirs(table_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

if include_noSplitD2:
    options_splitD2 = [True, False]
else:
    options_splitD2 = [True]

test_protocols = [Protocols.VAR_D2_SIZE]
# test_protocols = Protocols

all_results = []
for dataset_name, data_path, loader, protected in datasets():
    dataset_name = f'{dataset_name}_{protected}'
    X, y, A = loader(data_path, protected_attr=protected)
    scaler = sklearn.preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    results = []
    for run, (D1, D2, D3, AD1) in enumerate(gen_split_data(X, y, A, repetitions=datasplit_repetitions)):
        pbar = tqdm(itertools.product(iter_methods(), test_protocols, options_splitD2))
        for (Q_name, Q), protocol, splitD2 in pbar:
            pbar.set_description(f'{dataset_name} - {protocol}: {Q_name}')
            options['splitD2'] = splitD2
            if not options['splitD2']:
                if isinstance(Q, BaseQuantifier):
                    Q_name += '-nosD2'
                else:
                    continue

            result_path = os.path.join(result_dir, run_name())
            if skip_already_computed and os.path.exists(result_path):
                print(f'skipping {result_path}; already computed!')
                results_ = Result.load(result_path)
                results_.set('dataset', dataset_name) #retrospective
                results.append(results_)
                continue
            print(f'Running: {result_path}')

            if protocol == Protocols.VAR_D1_PREV:
                outs = eval_prevalence_variations_D1(D1, D2, D3, AD1, f, Q, **options)
            elif protocol == Protocols.VAR_D1_PREVFLIP:
                outs = eval_prevalence_variations_D1_flip(D1, D2, D3, AD1, f, Q, **options)
            elif protocol == Protocols.VAR_D2_SIZE:
                outs = eval_size_variations_D2(D1, D2, D3, f, Q, sample_sizes=None, nreps=options['nreps'], splitD2=options['splitD2'])
            elif protocol == Protocols.VAR_D2_PREV:
                outs = eval_prevalence_variations_D2(D1, D2, D3, f, Q, **options)
            elif protocol == Protocols.VAR_D3_PREV:
                outs = eval_prevalence_variations_D3(D1, D2, D3, f, Q, **options)

            outs.set('Q_name', Q_name)
            outs.set('run', run)
            outs.set('dataset', dataset_name)

            outs.save(result_path)
            results.append(outs)
    all_results.extend(results)

    # -------------------------------------------------
    # Generate plots and tables specific to a dataset
    # -------------------------------------------------
    for protocol in test_protocols:
        generate_tables(protocol, results, table_path=join(table_dir, f'tab_{protocol}_{dataset_name}.tex'))
        generate_plots(protocol, results, plotdir=join(plot_dir, dataset_name))

# -------------------------------------------------
# Generate general tables
# -------------------------------------------------
for protocol in test_protocols:
    generate_tables_joindatasets(protocol, all_results, table_path=join(table_dir, f'tab_{protocol}.tex'))
