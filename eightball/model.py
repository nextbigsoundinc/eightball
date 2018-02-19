import pandas as pd
import pickle
import copy

import imputer
import evaluation
import preprocessor
import param_search


class Model(object):
    """
        Model object for binary classifiers
    """
    def __init__(self, clf, training_data, target_column, imputer_='zero', preprocessor_=None):
        self.clf = clf
        self.training_data = training_data
        self.target_column = target_column
        Xtrain, ytrain = self.split(self.training_data, self.target_column)
        if preprocessor_ is None:
            preprocessor_ = preprocessor.Preprocessor()
            self.preprocessor = preprocessor_
        else:
            self.preprocessor = preprocessor_
        self.preprocessor.fit(Xtrain)
        Xtrain = self.preprocessor.process(Xtrain)

        if imputer_ is None:
            self.imputer = imputer.NoImpute()
        elif imputer_ == 'zero':
            self.imputer = imputer.ZeroFillImputer()
        else:
            self.imputer = imputer_
        self.imputer.fit(Xtrain)

    def evaluate(self, scoring=None, cv=5, nruns=5, proba=True, seed=None):
        """returns scores object for the given dataframe

        Args:
            scoring: scoring methods. should be a dictionary of {'scorer_name': scorer}
                (i.e. {'roc_auc': metrics.roc_auc_score}). By default roc_auc and brier
                scorers are used.
            nruns: number of times runs (should be a multiple of cv)
            cv: number of cross validation folds
        """
        self.eval = evaluation.evaluate(
            self, self.training_data, self.target_column,
            scoring=scoring, cv=cv, nruns=nruns, proba=proba, seed=seed
        )
        return self.eval

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

    def param_search(self, param_bounds, scoring, n_jobs=-1, use_best=False, verbose=False):
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
        self.param_search_results = param_search.ParamSearch(self, scoring)
        self.param_search_results.search(param_bounds, verbose=verbose)
        return self.param_search_results

    def fit(self):
        """
        """
        Xtrain, ytrain = self.split(self.training_data, self.target_column)
        Xtrain = self.preprocess_and_impute(Xtrain)
        self.feature_list = Xtrain.columns.tolist()

        # fit classifier
        self.clf.fit(Xtrain, ytrain)

    def predict(self, X, proba=True):
        X = self.preprocessor.process(X)
        X = self.imputer.impute(X)
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
        pickle.dump(model_copy, open(path, "wb"))
        del model_copy

    def preprocess_and_impute(self, X):
        """preprocess and impute missing values"""
        X = self.preprocessor.process(X)
        X = self.imputer.impute(X)
        return X

    def split(self, df, target_column=None):
        """split dataframe into featuers and target"""
        if target_column is None:
            target_column = self.target_column
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        return (X, y)
