from ._package import GridSearchCV
from ._package import cross_val_score
from ._decorator import runtime_log, log
from ._package import (RandomForestClassifier,
                       VotingClassifier, lgb, xgb, np, KFold, pd)
from ._evaluate import sample_weight, calc_max_ks

__all__ = ['Config', 'vote']


class Config:
    xgb_best_param = {'learning_rate': 0.25,
                      'max_depth': 5,
                      'objective': 'binary:logistic',
                      'eval_metric': 'error'}
    xgb_best_param2 = {'learning_rate': 0.25,
                       'max_depth': 5,
                       'objective': 'binary:logistic',
                       'n_estimators': 90}
    lgb_best_param = dict(subsample=0.7,
                          reg_lambda=0.1,
                          reg_alpha=0.05,
                          num_leaves=63,
                          n_estimators=280,
                          min_child_weight=4,
                          min_child_samples=80,
                          max_bin=200,
                          learning_rate=0.05,
                          colsample_bytree=0.9,
                          n_jobs=-1,
                          verbose=1)
    lgb_best_param2 = dict(colsample_bytree=0.7,
                           learning_rate=0.05,
                           max_bin=170,
                           max_depth=-1,
                           min_child_samples=30,
                           min_child_weight=6,
                           min_split_gain=0.0,
                           n_estimator=350,
                           n_jobs=10,
                           num_leaves=63,
                           objective=None,
                           random_state=0,
                           reg_alpha=0.05,
                           reg_lambda=0.1,
                           silent=True,
                           subsample=0.9,
                           subsample_for_bin=90000,
                           subsample_freq=1)
    rf_best_param = dict(n_estimators=90,
                         max_depth=10,
                         n_jobs=-1,
                         verbose=1)


@runtime_log
@log
def vote(data, target):
    rf = RandomForestClassifier(**(Config.rf_best_param))
    xg = xgb.XGBClassifier(**(Config.xgb_best_param2))
    lg = lgb.LGBMClassifier(**(Config.lgb_best_param))
    eclf = VotingClassifier(
        estimators=[('rf', rf), ('xgboost', xg), ('lgb', lg)],
        voting='soft',
        weights=[1, 2, 3])
    scores = []
    names = []
    for clf, name in zip([rf, xg, lg,  eclf],
                         ['Random Forest', 'XGBoost', 'LightGBM', 'Ensemble']):
        score = np.zeros(5)
        step = 0
        kf = KFold(n_splits=5, shuffle=True, random_state=125)
        for train, test in kf.split(data, target):
            train_data, train_target, test_data, test_target = data[
                train], target[train], data[test], target[test]
            unit_score = np.zeros(10)
            for i in range(10):
                aa = np.arange(train_data.shape[0])
                np.random.shuffle(aa)
                train_data, train_target = train_data[aa], train_target[aa]
                clf.fit(train_data, train_target,
                        sample_weight=sample_weight(train_target))
                y_pred = clf.predict_proba(test_data)
                unit_score[i] = calc_max_ks(clf, test_data, test_target)
            score[step] = unit_score.mean()
            step += 1
        scores.append(score)
        names.append(name)
        print('KS: %.5f (+/- %.5f)[%s]' %
              (score.mean(), score.std(), name))
    res = pd.DataFrame(scores, index=names,
                       columns=list(range(1, 5 + 1, 1)))
    return res
