import quapy as qp
import pandas as pd



def adultcsv_loader(path, protected_attr, covariates=None, dummies=None, drop_first_dummy=False, remove_missing=True):

    if not remove_missing:
        raise NotImplementedError()

    if covariates is None:
        covariates = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'capital-gain',
                      'capital-loss', 'hours-per-week', 'native-country', 'income']

    if dummies is None:
        dummies = ['workclass', 'education', 'marital-status', 'occupation', 'native-country',
                   'income']

    privileged = {
        'gender': 'Male',
        'race': 'White'
    }
    assert protected_attr in privileged, f'unknown protected attribute; valid are {privileged.keys()}'

    if not all(d in covariates for d in dummies):
        print('warning: some dummy-features are not in covariates')
        dummies = [d for d in dummies if d in covariates]

    df = pd.read_csv(path)
    if remove_missing:
        # removing entries with missig values, may want to keep and treat as separate class instead
        df = df[(df != '?').all(axis=1)]
        df.reset_index(drop=True, inplace=True)

    # process covariates
    X = df[covariates]
    for dummy in dummies:
        df_dum = pd.get_dummies(X[dummy], prefix=dummy, drop_first=drop_first_dummy)
        X = pd.concat([X, df_dum], axis=1)
        X.drop(dummy, axis=1, inplace=True)
    X = X.values

    # process predictor
    y = qp.data.binarize(df['income'], pos_class='>50K')

    # process protected attribute
    A = qp.data.binarize(df[protected_attr], pos_class=privileged[protected_attr])

    return X, y, A






