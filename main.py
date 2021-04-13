from data import adultcsv_loader
from common import *
from sklearn.linear_model import LogisticRegression
from plot import *
from quapy.method.aggregative import CC, ACC, PACC, EMQ, HDy
import itertools
from common import Protocol
import pathlib


qp.environ['SAMPLE_SIZE'] = 100

protocol = Protocol.VAR_D3_PREV
model_selection=False
datasplit_repetitions = 1
data_path = "./adult.csv"
protected_attr = "gender"
debug = True
options = {
    'nprevs': 21,
    'nreps': 10,
    'sample_size': 50
}
result_dir = './results'
os.makedirs(result_dir, exist_ok=True)

skip_already_computed = True  # set to False to force re-generate them
dataset_name = pathlib.Path(data_path).name.replace('.','_')

# Notice P(A=0, y=1)<0.04, i.e. rich women are rare.
# This is a bottleneck for quantification, e.g. in the vary_D2_size protocol, when |D2|=1000 there are ~37 women in D21
# making life pretty hard for M1.

# TODO: should we have different sample_size for D21, D20 to reflect the prevalence of hat_y in the data?
# TODO: should we have different sample_size for D31, D30 to reflect the prevalence of hat_y in the data?

f = LogisticRegression()
# if debug:
#     f = LogisticRegression()
# else:
#     f = GridSearchCV(estimator=LogisticRegression(),
#                      param_grid={'C': np.logspace(-3,3,7), 'class_weight': ['balanced', None]}, cv=5,
#                      n_jobs=-1)

def classifiers():
    hyperparams = {'C': np.logspace(-3,3,7), 'class_weight': ['balanced', None]}
    yield 'LR', LogisticRegression(), hyperparams


def quantifiers():
    yield 'CC', CC
    # yield 'ACC', ACC
    yield 'PACC', PACC
    # yield 'EMQ', EMQ
    # yield 'HDy', HDy


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

        result_path = os.path.join(result_dir, f'{dataset_name}_{Q_name}_{run}_Prot{protocol}.pkl')

        if skip_already_computed and os.path.exists(result_path):
            print(f'skipping {result_path}; already computed!')
            continue

        """"
        if protocol == Protocols.VAR_D1_PREV:
            outs = eval_prevalence_variations_D1(D1, D2, D3, AD1, f, Q, nprevs=nprevs, nreps=nreps, sample_size=sample_size)
        elif protocol == Protocols.VAR_D1_PREVFLIP:
        """
        if protocol == Protocol.VAR_D1_PREVFLIP:
            outs = eval_prevalence_variations_D1_flip(D1, D2, D3, AD1, f, Q, **options)
        elif protocol == Protocol.VAR_D2_SIZE:
            outs = eval_size_variations_D2(D1, D2, D3, f, Q, sample_sizes=None, nreps=options['nreps'])
        elif protocol == Protocol.VAR_D2_PREV:
            outs = eval_prevalence_variations_D2(D1, D2, D3, f, Q, **options)
        elif protocol == Protocol.VAR_D3_PREV:
            outs = eval_prevalence_variations_D3(D1, D2, D3, f, Q, **options)

        outs.set('Q_name', Q_name)
        outs.set('run', run)
        outs.save(result_path)

    if debug:
        break


