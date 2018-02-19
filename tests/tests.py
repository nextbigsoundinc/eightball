from eightball.model import Model
from eightball import evaluation

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import pandas as pd
import numpy as np

from unittest import TestCase


class TestLogit(TestCase):
    def setUp(self):
        iris = datasets.load_iris()
        y = iris.target
        y[y != 1] = 0
        training_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        training_df['is_versicolor'] = y
        self.training_data = training_df

        log_clf = LogisticRegression()
        logit_model = Model(clf=log_clf, training_data=training_df, target_column='is_versicolor')
        logit_model.fit()
        self.logit_model = logit_model

    def test_coefs(self):
        self.assertEqual(np.round(self.logit_model.clf.coef_[0][0], 2), 0.42)

    def test_eval_simple_scorer(self):
        train_score = self.logit_model.score(self.training_data)
        assert(np.round(train_score['brier'], 2) == 0.18)
        assert(np.round(train_score['roc_auc'], 2) == 0.81)

    def test_model_scorer(self):
        train_score = evaluation.simple_scorer(
            self.logit_model, self.training_data, 'is_versicolor')
        assert(np.round(train_score['brier'], 2) == 0.18)
        assert(np.round(train_score['roc_auc'], 2) == 0.81)

    def test_grid_search(self):
        param_grid = {'penalty': ['l1', 'l2'], 'C': [0.5, 1.0, 2.0]}
        self.logit_model.grid_search(scoring='roc_auc', param_grid=param_grid)
        margin_scores = self.logit_model.grid_scores._get_score_margins('mean_test_score')
        self.assertEqual(len(margin_scores), 2)
        for s in margin_scores:
            if 'penalty' in s['params']:
                self.assertEqual(np.round(s['scores'][0], 2), 0.58)
                self.assertEqual(len(s['scores']), 2)

        best_scores = self.logit_model.grid_scores._reduce_to_best_reults('mean_test_score')
        self.assertEqual(len(best_scores), 2)
        for s in best_scores:
            if 'C' in s['params']:
                self.assertEqual(np.round(s['scores'][1], 2), 0.63)
                self.assertEqual(len(s['scores']), 3)


class TestRF(TestCase):
    def setUp(self):
        iris = datasets.load_iris()
        y = iris.target
        y[y != 1] = 0
        training_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        training_df['is_versicolor'] = y
        self.training_data = training_df

        rf_clf = RandomForestClassifier()
        rf_model = Model(clf=rf_clf, training_data=training_df, target_column='is_versicolor')
        rf_model.evaluate()
        self.rf_model = rf_model

    def test_feature_importance(self):
        # check that there are 4 rows in the feature importance df (one for each feature)
        self.assertEqual(len(self.rf_model.eval._feature_importance), 4)

    def test_scores(self):
        assert('roc_auc' in self.rf_model.eval._scores)
