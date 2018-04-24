import pandas as pd
import pickle
import copy

import imputer
import evaluation
import preprocessor
import param_optimization


def load(path):
    with open(path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model


class Model(object):
    """
        Model object for binary classifiers
        Args:
            clf: a scikit learn style classifier object
            training_data: a pandas data frame of training features and target variable
            target_column: a string indicating the column name of the target variable
            imputer_: can be a string ('mean', 'median', 'most_frequent', or 'zero') for
                simple imputation, or an imputation object
            preprocessor_: a preprocessor object or list of preprocessor objects
            warn_level: 'quiet', 'warn', or 'error'
    """
    def __init__(
        self, clf, training_data, target_column, imputer_=None, preprocessor_=None,
        warn_level='warn'
    ):
        self.clf = clf
        self.training_data = training_data
        self.target_column = target_column
        self._warn_level = warn_level
        self.set_preprocessor(preprocessor_)
        self.set_imputer(imputer_)
        self.set_training_set(training_data, target_column)

    def set_training_set(self, training_data, target_column):
        self.training_data = training_data
        self.target_column = target_column
        Xtrain, ytrain = self.split(self.training_data, self.target_column)
        for pp in self.preprocessors:
            pp.fit(Xtrain)
            Xtrain = self._preprocess_one(Xtrain, pp)
        self.imputer.fit(Xtrain)

    def set_preprocessor(self, preprocessor_):
        if preprocessor_ is None:
            preprocessor_ = [preprocessor.Preprocessor()]
        elif type(preprocessor_) is not list:
            preprocessor_ = [preprocessor_]
        self.preprocessors = preprocessor_
        Xtrain, ytrain = self.split(self.training_data, self.target_column)
        for pp in self.preprocessors:
            pp.fit(Xtrain)
            Xtrain = self._preprocess_one(Xtrain, pp)

    def set_imputer(self, imputer_):
        if imputer_ is None:
            self.imputer = imputer.NoImpute(on_missing=self._warn_level)
        elif imputer_ in ['zero', 'mean', 'median', 'most_frequent']:
            self.imputer = imputer.SimpleImputer(imputer_, on_missing=self._warn_level)
        else:
            self.imputer = imputer_  # should be a custom imputer class
        Xtrain, ytrain = self.split(self.training_data, self.target_column)
        Xtrain = self._preprocess_all(Xtrain)
        self.imputer.fit(Xtrain)

    def set_target_column(self, target_column):
        self.target_column = target_column

    def set_warn_level(self, warn_level):
        self._warn_level = warn_level
        if type(self.imputer) == imputer.SimpleImputer:
            self.imputer.set_on_missing(warn_level)

    def set_clf(self, clf):
        self.clf = clf

    def fit(self):
        """
        """
        Xtrain, ytrain = self.split(self.training_data, self.target_column)
        Xtrain = self.preprocess_and_impute(Xtrain)
        self.feature_list = Xtrain.columns.tolist()

        # fit classifier
        self.clf.fit(Xtrain, ytrain)

    def predict(self, X, proba=True):
        X = self.preprocess_and_impute(X)
        X = X[self.feature_list]

        prediction_df = pd.DataFrame(index=X.index)
        if proba:
            p = self.clf.predict_proba(X)
            prediction_df['prob'] = [pi[1] for pi in p]
        else:
            prediction_df[self.target_column] = self.clf.predict(X)
        return prediction_df

    def save(self, path, store_data=False):
        model_copy = copy.copy(self)
        if store_data is False:
            del model_copy.training_data
        pickle.dump(model_copy, open(path, "wb"), protocol=2)
        del model_copy

    def _preprocess_all(self, X):
        for pp in self.preprocessors:
            X = self._preprocess_one(X, pp)
        return X

    def _preprocess_one(self, X, pp):
        cols = X.columns
        idxs = X.index
        X = pp.transform(X)
        if type(X) != pd.core.frame.DataFrame:
            if self._warn_level.lower() in ['warn', 'error']:
                print(
                    'preprocessors should return a dataframe. '
                    'column names not guaranteed to be correct.'
                )
            X = pd.DataFrame(X, columns=cols, index=idxs)
        return X

    def preprocess_and_impute(self, X):
        """preprocess and impute missing values"""
        # converting to pandas dataframe to support scikit learn transformers
        # may deprecate in future
        X = self._preprocess_all(X)
        cols = X.columns
        idxs = X.index
        X = self.imputer.transform(X)
        if type(X) != pd.core.frame.DataFrame:
            if self._warn_level.lower() in ['warn', 'error']:
                print(
                    'imputers should return a dataframe.'
                    'column names are not guaranteed to be correct.'
                )
            X = pd.DataFrame(X, columns=cols, index=idxs)
        return X

    def split(self, df, target_column=None):
        """split dataframe into featuers and target"""
        if target_column is None:
            target_column = self.target_column
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        return (X, y)

    def evaluate(self, scoring=None, cv=5, nruns=5, proba=True, seed=None):
        """returns scores object for the given dataframe

        Args:
            scoring: scoring methods. should be a dictionary of {'scorer_name': scorer}
                (i.e. {'roc_auc': metrics.roc_auc_score}). By default roc_auc and brier
                scorers are used.
            nruns: number of times runs (should be a multiple of cv)
            cv: number of cross validation folds
        """
        self.eval_results = evaluation.evaluate(
            self, self.training_data, self.target_column,
            scoring=scoring, cv=cv, nruns=nruns, proba=proba, seed=seed
        )
        return self.eval_results

    def score(self, df, target_column=None, scoring=None, proba=True):
        """score method for held out data set. fit must be called first. see evaluate
            method for cross validation scores.
        """
        if target_column is None:
            target_column = self.target_column
        return evaluation.simple_scorer(self, df, target_column, scoring=scoring, proba=proba)

    def grid_search(self, param_grid, scoring, n_jobs=-1, use_best=False):
        self.grid_scores = evaluation.grid_search(
            self, self.training_data, self.target_column,
            param_grid, scoring, n_jobs=-1
        )
        if use_best:
            self.clf = self.grid_scores.clf_grid.best_estimator_
            self.fit()
        return self.grid_scores

    def param_optimization(self, param_bounds, scoring, n_jobs=-1, use_best=False, verbose=False):
        """
            params_bounds is a dictionary that defines the paramaters of the search. the key
            is the parameter name and the value is a tuple of:
                (minimum param value, maximum param value,
                    minimum step size (optional), start value(optional)
                )

            Ex:
                param_bounds = {
                    'n_estimators': (1, 50, 5),
                    'min_samples_split': (1, 32, 1),
                    'max_features': (0.05, 0.95, 0.05)
                }

            This will start by searching a cube of param values with n_estimators=(1, 25, 50),
            min_samples_split=(1, 16, 32), and max_features=(0.05,0.5,0.95). The method will
            continue search smaller and smaller cubes of param values until it obtains the best
            score value for after searching with the min step size (5, 1, and 0.05 respectivley).
        """
        self.param_optimization_results = param_optimization.ParamSearch(self, scoring)
        self.param_optimization_results.search(param_bounds, verbose=verbose)
        return self.param_optimization_results
