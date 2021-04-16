import quapy as qp
import pandas as pd


def adultcsv_loader(path, protected_attr, covariates=None, dummies=None, drop_first_dummy=True, remove_missing=True, verbose=True):

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
        # removing entries with missing values, may want to keep and treat as separate class instead
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
    pos_y_cl = '>50K'
    y = qp.data.binarize(df['income'], pos_class=pos_y_cl)

    # process protected attribute
    A = qp.data.binarize(df[protected_attr], pos_class=privileged[protected_attr])

    if verbose:
        print(f'A=1 is {privileged[protected_attr]}; y=1 is {pos_y_cl}')

    return X, y, A


def compascsv_loader(path, protected_attr, covariates=None, dummies=None, races_keep=None, drop_first_dummy=True, verbose=True):

    if covariates is None:
        covariates = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree']

    if dummies is None:
        dummies = ['c_charge_degree']

    if races_keep is None:
        races_keep = ['African-American', 'Caucasian']

    privileged = {
        'sex': 'Male',
        'race': 'Caucasian'
    }
    assert protected_attr in privileged, f'unknown protected attribute; valid are {privileged.keys()}'

    if not all(d in covariates for d in dummies):
        print('warning: some dummy-features are not in covariates')
        dummies = [d for d in dummies if d in covariates]

    def compas_strandard_preproc(df):
        df = df[df['days_b_screening_arrest'] <= 30]
        df = df[df['days_b_screening_arrest'] >= -30]
        df = df[df['is_recid'] != -1]
        df = df[df['c_charge_degree'] != 'O']
        df = df[df['score_text'] != 'N/A']
        df.reset_index(drop=True, inplace=True)
        return df

    df = pd.read_csv(path)
    df = compas_strandard_preproc(df)
    df = df[df['race'].isin(races_keep)]
    df.reset_index(drop=True, inplace=True)

    # process covariates
    X = df[covariates]
    for dummy in dummies:
        df_dum = pd.get_dummies(X[dummy], prefix=dummy, drop_first=drop_first_dummy)
        X = pd.concat([X, df_dum], axis=1)
        X.drop(dummy, axis=1, inplace=True)
    X = X.values

    # process predictor
    pos_y_cl = 0
    y_col = 'is_recid'
    y = qp.data.binarize(df[y_col], pos_class=pos_y_cl)

    # process protected attribute
    A = qp.data.binarize(df[protected_attr], pos_class=privileged[protected_attr])

    if verbose:
        print(f'A=1 is {protected_attr}:{privileged[protected_attr]}; y=1 is {y_col}:{pos_y_cl}')

    return X, y, A


def ccdefaultcsv_loader(path, protected_attr, covariates=None, dummies=None, drop_first_dummy=True, verbose=True):

    if covariates is None:
        covariates = ['LIMIT_BAL', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                      'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                      'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    if dummies is None:
        dummies = ['EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    privileged = {'SEX': 1}
    assert protected_attr in privileged, f'unknown protected attribute; valid are {privileged.keys()}'

    if not all(d in covariates for d in dummies):
        print('warning: some dummy-features are not in covariates')
        dummies = [d for d in dummies if d in covariates]
    df = pd.read_csv(path)

    # process covariates
    X = df[covariates]
    for dummy in dummies:
        df_dum = pd.get_dummies(X[dummy], prefix=dummy, drop_first=drop_first_dummy)
        X = pd.concat([X, df_dum], axis=1)
        X.drop(dummy, axis=1, inplace=True)
    X = X.values

    # process predictor
    pos_y_cl = 0
    y_col = 'default payment next month'
    y = qp.data.binarize(df[y_col], pos_class=pos_y_cl)

    # process protected attribute
    A = qp.data.binarize(df[protected_attr], pos_class=privileged[protected_attr])

    if verbose:
        print(f'A=1 is {protected_attr}:{privileged[protected_attr]}; y=1 is {y_col}:{pos_y_cl}')

    return X, y, A



