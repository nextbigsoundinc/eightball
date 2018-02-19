import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, KFold
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time


def _proba_score_proxy(y_true, y_probs, class_idx, proxied_func, **kwargs):
    return proxied_func(y_true, y_probs[:, class_idx], **kwargs)


brier_scorer = metrics.make_scorer(
    _proba_score_proxy,
    greater_is_better=False,
    needs_proba=True,
    class_idx=1,
    proxied_func=metrics.brier_score_loss
)


def simple_scorer(model, df, target_column, scoring=None, proba=True):
    scores = {'time': []}
    if scoring is None:
        scoring = {'roc_auc': metrics.roc_auc_score, 'brier': metrics.brier_score_loss}
    # X, y = model.split_and_impute(df, target_column, fit_imputer=False)
    X, y = model.split(df, target_column)
    X = model.preprocessor.process(X)
    X = model.imputer.impute(X)

    start_time = time.time()
    if proba:
        p = model.clf.predict_proba(X)
        ypred = [pi[1] for pi in p]
    else:
        ypred = model.clf.predict(X)
    end_time = time.time()
    scores['time'] = (end_time - start_time) / len(X)
    for k, scorer in scoring.iteritems():
        scores[k] = scorer(y, ypred)
    return scores


def evaluate(model, df, target_column, scoring=None, cv=5, nruns=5, proba=True, seed=None):
    """Performs n split/fits scores the results to generate avg and std
        scores and feature importance.
        Note: feature importance is only available for random forest. Currently
            only generate brier and roc_auc scores.
    Args:
        df - dataframe of features and target variables
        target_column - column name for target variable
        nruns - number of times to split and fit (default 10)

    Returns:
        a Scores object dictionary of 'scores' and 'feature_importance'. 'scores' is a dictinoary
        with keys 'roc_auc' and 'brier' where the values are the tuple
        (mean_score, std_score). 'feature_importance' is a dataframe with columns
        'feature' (the feature name) and 'score' (the importance score).

    """
    clf = copy.copy(model.clf)
    if seed is not None:
        clf.set_params(random_state=seed)
    scores = {'time': []}
    if scoring == 'accuracy':
        scoring = {'accuracy': metrics.accuracy_score}
    if scoring == 'roc_auc':
        scoring = {'roc_auc': metrics.roc_auc_score}
    if scoring == 'brier':
        scoring = {'brier': metrics.brier_score_loss}
    if scoring is None:
        scoring = {'roc_auc': metrics.roc_auc_score, 'brier': metrics.brier_score_loss}
    for k, v in scoring.iteritems():
        scores[k] = {'scores': []}
    feature_score_sums = {}
    X, y = model.split(df, target_column)
    X = model.preprocess_and_impute(X)

    for i in range(0, int(np.ceil(1.0 * nruns / cv))):
        kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
        for train_index, test_index in kf.split(X):
            Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
            clf.fit(Xtrain, ytrain)
            # get predicted scores on test set
            start_time = time.time()
            if proba:
                p = clf.predict_proba(Xtest)
                ypred = [pi[1] for pi in p]
            else:
                ypred = clf.predict(Xtest)
            end_time = time.time()
            scores['time'].append((end_time - start_time) / len(Xtest))
            for k, scorer in scoring.iteritems():
                scores[k]['scores'].append(scorer(ytest, ypred))

            # TODO add more classifiers that include feature importance attribute to this list
            if clf.__class__.__name__ in ['RandomForestClassifier', 'XGBClassifier']:
                features = X.columns
                importance_scores = clf.feature_importances_
                for j in range(len(features)):
                    if features[j] in feature_score_sums:
                        feature_score_sums[features[j]] += importance_scores[j]
                    else:
                        feature_score_sums[features[j]] = importance_scores[j]
    feature_importance_df = pd.DataFrame({
        'feature': feature_score_sums.keys(),
        'score': feature_score_sums.values(),
    }).sort_values(by='score', ascending=False).reset_index(drop=True)
    for k, v in scoring.iteritems():
            scores[k]['mean'] = np.mean(scores[k]['scores'])
            scores[k]['std'] = np.std(scores[k]['scores'])
    return Eval(scores, feature_importance_df)


def feature_drop(model, df, target_column, scorer='brier', min_features=1):
    """ranks features by feature importance and successively drops features
        and fits/scores on the remaining feature set until there are
        only min_features left. Scores are the average of 10 runs (currently
        hardcoded).

    Args:
        df - dataframe of features and target variables
        target_column - column name for target variable
        nruns - number of times to split and fit (default 10)
        scorer (str): must be one of the scorers available for the score_model
            method (brier by default)

    Returns:
        dataframe with columns 'n_features' (number of features used to fit/score),
            'avg_score', and 'std_score'
        TODO: update score_model method to allow for more scorers
    """
    all_scores = {'n_features': [], 'avg_score': [], 'std_score': []}
    scores = evaluate(model, df, target_column, nruns=10)
    feature_importance_df = scores._feature_importance
    for i in range(0, len(feature_importance_df) - min_features + 1):
        features_to_drop = feature_importance_df\
            .sort_values(by='score').head(i)['feature'].tolist()
        df_dropped = df.drop(features_to_drop, axis=1).copy()
        scores = evaluate(model, df_dropped, target_column, nruns=10)
        all_scores['n_features'].append(len(df_dropped.columns))
        all_scores['avg_score'].append(scores._scores[scorer]['mean'])
        all_scores['std_score'].append(scores._scores[scorer]['std'])
    all_scores_df = pd.DataFrame(all_scores).set_index('n_features')
    return all_scores_df


def plot_feature_drop(feature_drop_scores_df):
    """pass the output of feature_drop generate a plot of dropping features"""
    feature_drop_scores_df['avg_score'].plot(yerr=feature_drop_scores_df['std_score'])


def grid_search(model, traing_df, target_column, param_grid, scoring, n_jobs=-1):
    """
        param_grid = {
            'n_estimators': [30, 100, 300],
            'min_samples_split': [8, 16, 24, 32, 64],
            'max_features': ['sqrt', 1, 0.1, 0.2, 0.3]
        }
        scoring = brier_scorer
    """
    clf_grid = GridSearchCV(model.clf, param_grid, n_jobs=n_jobs, scoring=scoring)
    X, y = model.split(traing_df, target_column)
    X = model.preprocessor.process(X)
    X = model.imputer.impute(X)
    clf_grid.fit(X, y)
    return GridScores(clf_grid)


class GridScores(object):
    def __init__(self, clf_grid):
        self.clf_grid = clf_grid

    def plot_heat_map(self, metric, title='', reduce_by='best', margins=False, ax=None):
        """
            metric: 'mean_test_score'
        """
        values = self.clf_grid.cv_results_[metric]
        params = self.clf_grid.param_grid
        if reduce_by is not None:
            if reduce_by == 'avg':
                scores = self._get_score_margins(metric)
            elif reduce_by == 'best':
                scores = self._reduce_to_best_reults(metric)
            # TODO: number of subplots should be equal to size of scores
            f, ax = plt.subplots(3, figsize=(5, 12))
            i = 0
            for s in scores:
                self._plot_heat_map(
                    s['scores'], s['params'],
                    title=s['title'], ax=ax[i], margins=margins
                )
                plt.tight_layout()
                i += 1
            return None
        self._plot_heat_map(values, params, title, margins, ax)

    def _plot_heat_map(self, values, params, title='', margins=False, ax=None):
        """
            metric: 'mean_test_score'
        """
        params = copy.deepcopy(params)
        xlabs = copy.deepcopy(params.keys())
        xlabs.sort(reverse=True)
        x = []
        for xlab in xlabs:
            x.append(params[xlab])
        values = values.copy()
        if values.ndim == 1:
            xlen = [len(xi) for xi in x]
            xlen.reverse()
            values = values.reshape(xlen)
        if values.ndim == 1:
            values = np.array([values])
            x.append([])
            xlabs.append('')
        if margins:
            values = np.r_[values, [values.mean(axis=0)]]
            values = np.c_[values, values.mean(axis=1)]
            x[0].append('avg')
            x[1].append('avg')
        if ax is None:
            f, (ax) = plt.subplots(1)
        values_df = pd.DataFrame(values, index=x[1], columns=x[0])
        values_df.index.name = xlabs[1]
        values_df.columns.name = xlabs[0]
        sns.heatmap(values_df, annot=True, cmap=plt.cm.hot, ax=ax)
        ax.set_title(title)

    def _get_score_margins(self, metric):
        values = copy.deepcopy(self.clf_grid.cv_results_[metric])
        params = copy.deepcopy(self.clf_grid.param_grid)
        xlabs = np.flipud(np.array(params.keys()))
        x = np.flipud(np.array(params.values()))

        if values.ndim == 1:
            xlen = [len(xi) for xi in x]
            scores = values.reshape(xlen)
        else:
            scores = values
        scores_marg = []
        for i in range(scores.ndim):
            paramsi = {}
            for j in range(scores.ndim):
                if j != i:
                    if type(x[j]) == list:
                        paramsi[xlabs[j]] = x[j]
                    else:
                        paramsi[xlabs[j]] = x[j].tolist()
            scoresi = scores.mean(axis=i).T
            scores_marg.append({
                'scores': scoresi,
                'params': paramsi,
                'title': "avg'd over %s" % xlabs[i]
            })
        return scores_marg

    def plot(self, metric, title='', reduce_by=None, margins=True, ax=None):
        """
            metric: 'mean_score_time', 'n_estimators', etc.
        """
        if len(self.clf_grid.param_grid) != 1:
            return self.plot_heat_map(metric, title, reduce_by, margins, ax)
        param = self.clf_grid.param_grid.keys()[0]
        metrics_df = pd.DataFrame({
            param: self.clf_grid.param_grid[param],
            'mean_score_time': self.clf_grid.cv_results_['mean_score_time'],
            'mean_test_score': self.clf_grid.cv_results_['mean_test_score'],
            'std_test_score': self.clf_grid.cv_results_['std_test_score']
        })
        ylim = self._get_custom_plot_range(metrics_df, 'mean_test_score', 'std_test_score')
        metrics_df.plot(
            kind='scatter', y='mean_test_score', x=metric, yerr='std_test_score', ylim=ylim
        )

    def _get_custom_plot_range(self, df, field, err_field=None, margin=0.1):
        if err_field is not None:
            ymin = (df[field] - df[err_field]).min()
            ymax = (df[field] + df[err_field]).max()
        else:
            ymin = df[field].min()
            ymax = df[field].max()
        span = ymax - ymin
        return [ymin - margin * span, ymax + margin * span]

    def _reduce_to_best_reults(self, metric):
        best_params = self.clf_grid.best_params_
        scores = []
        for param, value in best_params.iteritems():
            mask = self.clf_grid.cv_results_['param_' + param].data == value
            scoresi = self.clf_grid.cv_results_[metric][mask].copy()
            paramsi = {}
            for p in self.clf_grid.param_grid.keys():
                if p != param:
                    paramsi[p] = self.clf_grid.param_grid[p]
            scores.append({
                'scores': scoresi,
                'params': paramsi,
                'title': '%s=%s' % (param, value)
            })
        return scores


class Eval(object):
    def __init__(self, scores, feature_importance=None):
        self._scores = scores
        self._feature_importance = feature_importance

    def print_scores(self):
        for k, score in self._scores.iteritems():
            if k != 'time':
                print('%s: %s +/- %s' % (k, score['mean'], score['std']))

    def plot_feature_importance(self, topn=15, part_idxs=None):
        if part_idxs is not None:
            parts = self._reduce_featuers(self._feature_importance, part_idxs)
            reduced_scores = []
            for t in parts.keys():
                cond = self._feature_importance['feature'].isin(parts[t])
                reduced_scores.append(self._feature_importance.loc[cond, 'score'].mean())
            fetures_reduced_df = pd.DataFrame({'feature': parts.keys(), 'score': reduced_scores})
        else:
            fetures_reduced_df = self._feature_importance

        fetures_reduced_df\
            .sort_values(by='score')\
            .tail(topn)\
            .set_index('feature')\
            .plot(kind='barh', figsize=(8, int(topn / 4)))

    def _reduce_featuers(self, feature_importance_df, part_idxs=[0]):
        parts_dict = {}
        for f in feature_importance_df['feature']:
            parts = f.split('_')
            if len(parts) < 3:
                continue
            t = []
            for i in part_idxs:
                t.append(parts[i])
            t = '_'.join(t)
            if t in parts_dict:
                parts_dict[t].append(f)
            else:
                parts_dict[t] = [f]
        return parts_dict
