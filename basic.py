import quapy as qp
from sklearn.linear_model import LogisticRegression
from quapy.data.reader import binarize
from quapy.data.base import LabelledCollection, Dataset
import pandas as pd
from sklearn.linear_model import LogisticRegression
import itertools


'''
This is a (rather dirty) minimum working sample to demonstrate the basics of our idea in action. 
Dataset splits and algorithms employed will definitely change.
'''

''''
    From df, return X (covariates) and y (response) as specified by caller. Categorical variables are one hot encoded
'''
def adult_df_to_xy(df, covariates=None, response=None):
    dummies = ['workclass', 'education', 'marital-status', 'occupation', 'race', 'gender', 'native-country', 'income']
    df_x = df[covariates]
    for dummy in dummies:
        if dummy in covariates:
            # df_dum = pd.get_dummies(df_x[dummy], prefix=dummy)
            df_dum = pd.get_dummies(df_x[dummy], prefix=dummy, drop_first=True)
            df_x = pd.concat([df_x, df_dum], axis=1)
            df_x.drop(dummy, axis=1, inplace=True)
    if response == "gender":
        pos_cl = "Male"
    elif response == "race":
        pos_cl = "White"
    elif response == "income":
        pos_cl = ">50K"
    else:
        raise ValueError
    df_y = (df[response])
    return df_x.values, binarize(df_y, pos_class=pos_cl), df_x, df_y

def build_ql_report(quantifier, dataset):
    return qp.evaluation.artificial_sampling_report(
        quantifier,  # the quantification method
        dataset.test,  # the test set on which the method will be evaluated
        sample_size=qp.environ['SAMPLE_SIZE'],  # indicates the size of samples to be drawn
        n_prevpoints=11,  # how many prevalence points will be extracted from the interval [0, 1] for each category
        n_repetitions=1,  # number of times each prevalence will be used to generate a test sample
        n_jobs=-1,  # indicates the number of parallel workers (-1 indicates, as in sklearn, all CPUs)
        random_seed=42,  # setting a random seed allows to replicate the test samples across runs
        error_metrics=['mae', 'mrae', 'mkld'],  # specify the evaluation metrics
        verbose=True  # set to True to show some standard-line outputs
    )

if __name__ == "__main__":
    df = pd.read_csv("./adult.csv")
    # removing entries with missig values, may want to keep and treat as separate class instead
    df = df[(df != '?').all(axis=1)]
    df.reset_index(drop=True, inplace=True)
    # dropping: fnlwgt, educational-num, relationship (may want to keep it but should unify husband/wife?)
    df = df[['age', 'workclass', 'education', 'marital-status', 'occupation', 'race', 'gender', 'capital-gain',
             'capital-loss', 'hours-per-week', 'native-country', 'income']]


    # get X,y
    n = df.shape[0]
    d_size = n // 3
    X,y,_,_ = adult_df_to_xy(df,
                         covariates=['age', 'workclass', 'education', 'marital-status', 'occupation',
                                       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'],
                          response='income')
    # fit y classifier on D1
    logreg = LogisticRegression()
    X1 = X[:d_size,:]
    y1 = y[:d_size]
    logreg.fit(X1, y1)

    # predict y on D2, D3
    X23 = X[d_size:d_size*3, :]
    y23 = y[d_size:d_size*3]
    y23_pred = logreg.predict(X23)
    accuracy_y23_pred = sum(y23==y23_pred) / len(y23)

    # partition D23 based on y predictions
    X, A, _, _ = adult_df_to_xy(df,
                                  covariates=['age', 'workclass', 'education', 'marital-status', 'occupation',
                                              'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'],
                                  response='gender')

    X23 = X[d_size:d_size * 3, :]
    A23 = A[d_size:d_size * 3]
    X23_0 = X23[y23_pred == 0, :]
    X23_1 = X23[y23_pred == 1, :]
    assert(X23_0.shape[0] + X23_1.shape[0] == X23.shape[0])
    A23_0 = A23[y23_pred == 0]
    A23_1 = A23[y23_pred == 1]

    # fit ACC and CC quantifiers, separately for positive predicted ad negative predicted data points
    train_prop = 0.5
    data_ql_23_0 = LabelledCollection(X23_0, A23_0)
    dataset_23_0 = Dataset(*data_ql_23_0.split_stratified(train_prop, random_state=0))
    quantifier_0_ACC = qp.method.aggregative.ACC(LogisticRegression())
    quantifier_0_ACC.fit(dataset_23_0.training)
    quantifier_0_CC = qp.method.aggregative.CC(LogisticRegression())
    quantifier_0_CC.fit(dataset_23_0.training)

    data_ql_23_1 = LabelledCollection(X23_1, A23_1)
    dataset_23_1 = Dataset(*data_ql_23_1.split_stratified(train_prop, random_state=0))
    quantifier_1_ACC = qp.method.aggregative.ACC(LogisticRegression())
    quantifier_1_ACC.fit(dataset_23_1.training)
    quantifier_1_CC = qp.method.aggregative.CC(LogisticRegression())
    quantifier_1_CC.fit(dataset_23_1.training)

    # evaluate quantifiers
    qp.environ['SAMPLE_SIZE'] = 500
    report_class0_ACC = build_ql_report(quantifier_0_ACC, dataset_23_0)
    report_class1_ACC = build_ql_report(quantifier_1_ACC, dataset_23_1)
    report_class0_CC = build_ql_report(quantifier_0_CC, dataset_23_0)
    report_class1_CC = build_ql_report(quantifier_1_CC, dataset_23_1)

    # build independence_report, a DataFrame with all possible combinations of true prevalences for women in
    # D30 and D31 and the respective estimates by CC and ACC
    idcs_01 = itertools.product(range(report_class0_ACC.shape[0]), range(report_class1_ACC.shape[0]))
    true_prev_A0 = []
    true_prev_A1 = []
    estim_prev_A0_CC = []
    estim_prev_A1_CC = []
    estim_prev_A0_ACC = []
    estim_prev_A1_ACC = []
    for idx0, idx1 in idcs_01:
        true_prev_A0.append(report_class0_ACC["true-prev"][idx0])
        true_prev_A1.append(report_class1_ACC["true-prev"][idx1])
        estim_prev_A0_CC.append(report_class0_CC["estim-prev"][idx0])
        estim_prev_A1_CC.append(report_class1_CC["estim-prev"][idx1])
        estim_prev_A0_ACC.append(report_class0_ACC["estim-prev"][idx0])
        estim_prev_A1_ACC.append(report_class1_ACC["estim-prev"][idx1])
    true_deltas = [p0[0] - p1[0] for p0,p1 in zip(true_prev_A0, true_prev_A1)]
    estim_deltas_CC = [p0[0] - p1[0] for p0,p1 in zip(estim_prev_A0_CC, estim_prev_A1_CC)]
    estim_deltas_ACC = [p0[0] - p1[0] for p0,p1 in zip(estim_prev_A0_ACC, estim_prev_A1_ACC)]
    data = {"true-prev0": true_prev_A0,
            "true-prev1": true_prev_A1,
            "true-delta": true_deltas,
            "estim-delta-CC-error": [td-ed for td,ed in zip(true_deltas, estim_deltas_CC)],
            "estim-delta-ACC-error": [td-ed for td,ed in zip(true_deltas, estim_deltas_ACC)]}
    independence_report = pd.DataFrame(data)

