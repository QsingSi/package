from ._package import np
from ._package import learning_curve
from ._package import Counter
from ._decorator import log
from ._package import roc_curve
from ._package import plt
from ._decorator import runtime_log
from ._package import pd
from ._package import math
from ._package import mean_squared_log_error
from ._package import make_scorer


__all__ = ['calc_max_ks', 'calc_ks', 'RMSLE',
           'sample_weight', 'plot_ks_threshold', 'plot_learning_curve', 'dis_lat_lon']


def _MSLE(y_true, y_pred):
    loss = mean_squared_log_error(y_true, y_pred)
    return math.sqrt(loss)


RMSLE = make_scorer(_MSLE, greater_is_better=False)


def _rad(degree):
    return degree * math.pi / 180.0


def dis_lat_lon(lat1, lon1, lat2, lon2):
    '''根据两点的经度和纬度，计算其之间的距离，单位:km'''
    radLat1 = [_rad(lat) for lat in lat1]
    radLat2 = [_rad(lat) for lat in lat2]
    radLon1 = [_rad(lon) for lon in lon1]
    radLon2 = [_rad(lon) for lon in lon2]
    a = np.array(radLat1) - np.array(radLat2)
    b = np.array(radLon1) - np.array(radLon2)
    EARTH_RADIUS = 6378.137
    res = []
    for i in range(len(a)):
        s = 2 * math.asin(math.sqrt(math.pow(math.sin(a[i] / 2), 2) + math.cos(
            radLat1[i]) * math.cos(radLat2[i]) * math.pow(math.sin(b[i] / 2), 2)))
        s *= EARTH_RADIUS
        res.append(s)
    return np.array(res)


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


def sample_weight(y):
    cnt = Counter(y)
    cnt0, cnt1 = cnt.get(0), cnt.get(1)
    upsample = int(cnt0 / cnt1 + 0.5)
    sample = [upsample if label == 1 else 1 for label in y]
    return sample


def plot_ks_threshold(estimator, X, y, drop=True):
    if hasattr(estimator, 'predict_proba'):
        pred = estimator.predict_proba(X)
        if len(pred[0]) == 2:
            prob = [p[1] for p in pred]
        else:
            prob = pred
    else:
        prob = estimator.predict(X)
    fpr, tpr, _ = roc_curve(y, prob)
    fpr, tpr, threshold = roc_curve(y, prob, drop_intermediate=drop)
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
