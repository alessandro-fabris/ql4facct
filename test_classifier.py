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
from plot import generate_plots_clf


# Define all hyper-parameters here
# --------------------------------------------

qp.environ['SAMPLE_SIZE'] = 100
model_selection = False
datasplit_repetitions = 5
options = {
    'nprevs': 11,
    'nreps': 10,
    'sample_size': 500,
}
result_dir = './results_clf'
plot_dir = './plots_clf'

skip_already_computed = True  # set to False to force re-generation of experiments

fclassweight='balanced'
f = LogisticRegression(class_weight=fclassweight)
fname = 'LR'


classifier_name = 'LR'
# classifier_name = 'SVM'

if classifier_name=='LR':
    classifier = LogisticRegression()
elif classifier_name=='SVM':
    classifier = LinearSVC()
else:
    raise ValueError('unknown classifier name', classifier_name)



# Define the datasets we would like to test
# --------------------------------------------
def datasets():
    yield 'adult', "datasets/adult.csv", adultcsv_loader, "gender"
    yield 'compas', "datasets/compas-scores-two-years.csv", compascsv_loader, "race"
    yield 'cc_default', "datasets/default of credit card clients.csv", ccdefaultcsv_loader, "SEX"


def run_name():
    options_par = '_'.join(f'{key}{options[key]}' for key in sorted(options.keys()))
    fmode = '' if fclassweight is None else '_fclassweight=balanced'
    return f'{dataset_name}_{Clf_name}_f{fname}_Run{run}_{protocol}_{options_par}_{fmode}.pkl'


# --------------------------------------------
# Compute the experiments
# --------------------------------------------

os.makedirs(result_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

test_protocols = [Protocols.VAR_D2_PREV, Protocols.VAR_D3_PREV]

def run_and_save(eval_func, D2s, D3s, clf, quantifiers, options, Clf_name, run, dataset_name, prefix, runname, results):
    result_path = os.path.join(result_dir, prefix+'_'+runname)
    if not skip_already_computed or not os.path.exists(result_path):
        print(f'Running: {result_path}')
        outs = eval_func(D2s, D3s, clf, quantifiers, **options)
        outs.set('Clf_name', Clf_name)
        outs.set('run', run)
        outs.set('dataset', dataset_name)
        outs.set('prefix', prefix)
        outs.save(result_path)
    else:
        print(f'\tskipping {result_path}; already computed!')
        outs = Result.load(result_path)
    results.append(outs)


all_results = []
for dataset_name, data_path, loader, protected in datasets():
    dataset_name = f'{dataset_name}_{protected}'
    X, y, A = loader(data_path, protected_attr=protected)
    scaler = sklearn.preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    Clf_name, clf = classifier_name, classifier
    results = []
    for run, (D1, D2, D3, AD1) in enumerate(gen_split_data(X, y, A, repetitions=datasplit_repetitions)):
        for protocol in test_protocols:
            quantifiers = [
                (f'CC({Clf_name})', CC(deepcopy(clf))),
                (f'PACC({Clf_name})', PACC(deepcopy(clf))),
                (f'SLD({Clf_name})', EMQ(deepcopy(clf)))
            ]
            runname = run_name()

            f = f.fit(*D1.Xy)
            D2_y1, D2_y0 = classify(f, D2)
            D3_y1, D3_y0 = classify(f, D3)

            if protocol == Protocols.VAR_D2_PREV:
                run_and_save(eval_clf_prevalence_variations_D2, D2_y0, D3_y0, clf, quantifiers, options, Clf_name, run,
                             dataset_name, 'D2y0', runname, results)
                run_and_save(eval_clf_prevalence_variations_D2, D2_y1, D3_y1, clf, quantifiers, options, Clf_name, run,
                             dataset_name, 'D2y1', runname, results)

            elif protocol == Protocols.VAR_D3_PREV:
                run_and_save(eval_clf_prevalence_variations_D3, D2_y0, D3_y0, clf, quantifiers, options, Clf_name, run,
                             dataset_name, 'D3y0', runname, results)
                run_and_save(eval_clf_prevalence_variations_D3, D2_y1, D3_y1, clf, quantifiers, options, Clf_name, run,
                             dataset_name, 'D3y1', runname, results)


    # -------------------------------------------------
    # Generate plots for this dataset
    # -------------------------------------------------
    generate_plots_clf(Protocols.VAR_D2_PREV, results, plotdir=join(plot_dir, dataset_name, Clf_name))
    generate_plots_clf(Protocols.VAR_D3_PREV, results, plotdir=join(plot_dir, dataset_name, Clf_name))

