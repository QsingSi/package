from ._package import GridSearchCV
from ._package import cross_val_score
from ._decorator import runtime_log, log

__all__ = ['Config']


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
