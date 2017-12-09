from ._package import np
from ._package import learning_curve
from ._package import Counter
from ._decorator import log
from ._package import roc_curve
from ._package import plt
from ._decorator import runtime_log
from ._package import pd


__all__ = ['calc_max_ks', 'calc_ks',
           'sample_weight', 'plot_ks_threshold', 'plot_learning_curve']


@runtime_log
@log
def calc_max_ks(estimator, X, y):
    if hasattr(estimator, 'predict_proba'):
        pred = estimator.predict_proba(X)
        if len(pred[0]) == 2:
            prob = [p[1] for p in pred]
        else:
            prob = pred
    else:
        prob = estimator.predict(X)
    fpr, tpr, _ = roc_curve(y, prob)
    return max(tpr - fpr)


@runtime_log
@log
def do_oversamping(self, df):
    '''保存xgboost模型时用,对数据做上采样'''
    df_p = df[df['label'] == 1.0]
    df_n = df[df['label'] == 0.0]
    assert len(df_p) + len(df_n) == len(df)
    if len(df_p) < len(df_n):
        df_p = pd.concat(
            [df_p, df_p.sample(len(df_n) - len(df_p), replace=True)])
    elif len(df_n) < len(df_p):
        df_n = pd.concat(
            [df_n, df_n.sample(len(df_p) - len(df_n), replace=True)])
    assert len(df_p) == len(df_n)
    return pd.concat([df_p, df_n]).sample(frac=1.)


@runtime_log
@log
def calc_ks(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return max(tpr - fpr)


def sample_weight(y):
    cnt = Counter(y)
    cnt0, cnt1 = cnt.get(0), cnt.get(1)
    upsample = int(cnt0 / cnt1)
    sample = [upsample if label == 1 else 1 for label in y]
    return sample


def plot_ks_threshold(y_true, y_pred):
    assert len(y_pred) == len(y_true), 'wrong value...'
    if len(y_pred[0]) == 2:
        prob = [p[1] for p in y_pred]
    else:
        prob = y_pred
    fpr, tpr, threshold = roc_curve(y_true, prob)
    ks = tpr - fpr
    plt.figure()
    plt.title('KS -- Threshold')
    plt.ylabel('KS')
    plt.xlabel('Threshold')
    plt.plot(threshold, ks, 'o-', color='r')
    return plt


def plot_learning_curve(estimator=None, X=None, y=None, **kw):
    '''return learning curve'''
    keys, val = kw.keys(), kw.values()
    if estimator is None:
        raise ValueError(
            'to plot the learning curve that the estimator must be included...')
    assert hasattr(estimator, 'fit') and hasattr(estimator, 'predict'), 'The estimator \
    need implement fit and predict method...'
    title = kw.get('title', 'Learning Curve(%s)' % estimator.__class__)
    if X is None:
        raise ValueError('The train data is needed...')
    if y is None:
        raise ValueError('The train label is needed...')
    ylim = kw.get('ylim', None)
    cv = kw.get('cv', 5)
    train_sizes = kw.get('train_sizes', np.linspace(0.1, 1.0, 5))
    scoring = kw.get('scoring', calc_max_ks)
    n_jobs = kw.get('n_jobs', -1)
    verbose = kw.get('verbose', 1)
    train_sizes, train_score, test_score = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs,
        train_sizes=train_sizes, verbose=verbose, scoring=scoring)
    train_score_mean = np.mean(train_score, axis=1)
    train_score_std = np.std(train_score, axis=1)
    test_score_mean = np.mean(test_score, axis=1)
    test_score_std = np.std(test_score, axis=1)
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.grid()
    plt.fill_between(train_sizes, train_score_mean - train_score_std,
                     train_score_mean + train_score_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_score_mean - test_score_std,
                     test_score_mean + test_score_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_score_mean, 'o-',
             color='r', label='Training score')
    plt.plot(train_sizes, test_score_mean, 'o-',
             color='g', label='Cross-validation score')
    plt.legend(loc='best')
    return plt
