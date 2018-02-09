import pandas as pd
import pickle
import copy

import imputer
import evaluation


class Model(object):
    """
        Model object for binary classifiers
    """
    def __init__(self, clf, training_data, target_column, _imputer='zero'):
        self.clf = clf
        self.training_data = training_data
        self.target_column = target_column
        if _imputer == 'zero':
            self._imputer = imputer.ZeroFillImputer()
        else:
            self._imputer = _imputer

    def evaluate(self, scoring=None, cv=5, nruns=5):
        """returns scores object for the given dataframe

        Args:
            scoring: scoring methods. should be a dictionary of {'scorer_name': scorer}
                (i.e. {'roc_auc': metrics.roc_auc_score}). By default roc_auc and brier
                scorers are used.
            nruns: number of times runs (should be a multiple of cv)
            cv: number of cross validation folds
        """
        self.eval = evaluation.evaluate(
            self, self.training_data, self.target_column, scoring=scoring, cv=cv, nruns=nruns
        )
        return self.eval

    def score(self, df, target_column=None, scoring=None):
        """score method for held out data set. fit must be called first. see evaluate
            method for cross validation scores.
        """
        if target_column is None:
            target_column = self.target_column
        return evaluation.simple_scorer(self, df, target_column, scoring=scoring)

    def grid_search(self, param_grid, scoring, n_jobs=-1, use_best=False):
        self.grid_scores = evaluation.grid_search(
            self, self.training_data, self.target_column,
            param_grid, scoring, n_jobs=-1
        )
        if use_best:
            self.clf = self.grid_scores.clf_grid.best_estimator_
        return self.grid_scores

    def fit(self):
        """
        """
        (Xtrain, ytrain) = self.split_and_impute(
            self.training_data, self.target_column, fit_imputer=True)
        self.clf.fit(Xtrain, ytrain)

    def predict(self, features):
        features = self._imputer.impute(features)
        p = self.clf.predict_proba(features)
        prediction_df = pd.DataFrame(index=features.index)
        prediction_df['prob'] = [pi[1] for pi in p]
        return prediction_df

    def save(self, path, store_data=False):
        model_copy = copy.copy(self)
        if store_data is False:
            del model_copy.training_data
        pickle.dump(model_copy, open(path, "wb"))
        del model_copy

    def split_and_impute(self, df, target_column, fit_imputer=False):
        """split dataframe into featuers and target and impute missing values"""
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        if fit_imputer:
            self._imputer.fit(X)
            self.feature_list = X.columns.tolist()
        X = self._imputer.impute(X)
        return (X, y)
