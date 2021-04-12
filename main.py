from data import adultcsv_loader
from common import *
from sklearn.linear_model import LogisticRegression
from plot import *
from quapy.method.aggregative import CC, ACC, PACC, EMQ, HDy
import itertools
from common import Protocol
from tables import generate_tables


qp.environ['SAMPLE_SIZE'] = 100

nprevs=21
nreps=10
sample_size=1000
model_selection=False
datasplit_repetitions = 1
data_path = "./adult.csv"
protected_attr = "gender"
# Notice P(A=0, y=1)<0.04, i.e. rich women are rare.
# This is a bottleneck for quantification, e.g. in the vary_D2_size protocol, when |D2|=1000 there are ~37 women in D21
# making life pretty hard for M1.

f = LogisticRegression()

protocol = Protocol.VAR_D3_PREV


def classifiers():
    hyperparams = {'C': np.logspace(-3,3,7), 'class_weight': ['balanced', None]}
    yield 'LR', LogisticRegression(), hyperparams


def quantifiers():
    yield 'CC', CC
    #yield 'ACC', ACC
    yield 'PACC', PACC
    yield 'EMQ', EMQ
    #yield 'HDy', HDy


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


X, y, A = adultcsv_loader(data_path, protected_attr=protected_attr)

results = []
for run, (D1, D2, D3, AD1) in enumerate(gen_split_data(X, y, A, repetitions=datasplit_repetitions)):
    pbar = tqdm(iter_quantifiers(model_selection))
    for Q_name, Q in pbar:
        pbar.set_description(f'{protocol}: {Q_name}')

        """"
        if protocol == Protocols.VAR_D1_PREV:
            outs = eval_prevalence_variations_D1(D1, D2, D3, AD1, f, Q, nprevs=nprevs, nreps=nreps, sample_size=sample_size)
        elif protocol == Protocols.VAR_D1_PREVFLIP:
        if protocol == Protocol.VAR_D1_PREVFLIP:
            outs = eval_prevalence_variations_D1_flip(D1, D2, D3, AD1, f, Q, nprevs=nprevs, nreps=nreps, sample_size=sample_size)
        elif protocol == Protocol.VAR_D2_SIZE:
            outs = eval_size_variations_D2(D1, D2, D3, f, Q, nreps=nreps, sample_sizes=None)
        elif protocol == Protocol.VAR_D2_PREV:
            # TODO: should we have different sample_size for D21, D20 to reflect the prevalence of hat_y in the data?
            outs = eval_prevalence_variations_D2(D1, D2, D3, f, Q, nprevs=nprevs, nreps=nreps, sample_size=sample_size)
        el"""
        if protocol == Protocol.VAR_D3_PREV:
            # TODO: should we have different sample_size for D31, D30 to reflect the prevalence of hat_y in the data?
            outs = eval_prevalence_variations_D3(D1, D2, D3, f, Q, nprevs=nprevs, nreps=nreps, sample_size=sample_size)

        outs.Q_name = Q_name
        outs.run = run
        results.append(outs)




# ------------------------------------
# tables
# ------------------------------------
generate_tables(protocol, results)


# ------------------------------------
# plots
# ------------------------------------
generate_plots(protocol, results)





