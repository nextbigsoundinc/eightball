import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import Imputer


MISSING_VALUE_FILLER = -1  # if imputation doesn't work fill with this


def add_subtract_features(df, feature_list, error_on_missing_features=False):
    for feature in feature_list:
        if feature not in df.columns:
            if error_on_missing_features:
                raise Exception(
                    'ERROR: %s feature dos not exist in this data set and is required for '
                    'fitting. This will negatively affect the quality of these predictions.'
                    % feature
                )
            else:
                print(
                    'WARNING: %s feature dos not exist in this data set and is required for '
                    'fitting. This will negatively affect the quality of these predictions.'
                    % feature
                )
            df[feature] = np.nan
    return df[feature_list]


class NoImpute(object):
    """No Impute option (still adds or removes columns)"""
    def __init__(self):
        self.features = None

    def fit(self, features_df):
        self.features = features_df.columns

    def impute(self, features_df):
        if self.features is None:
            raise Exception('must fit before imputing')
        features_df = add_subtract_features(features_df, self.features)
        return features_df


class ZeroFillImputer(object):
    """Simple Imputer that just imputes missing values as 0"""
    def __init__(self):
        self.features = None

    def fit(self, features_df):
        self.features = features_df.columns

    def impute(self, features_df):
        if self.features is None:
            raise Exception('must fit before imputing')
        features_df = add_subtract_features(features_df, self.features)
        return features_df.fillna(0)


class NBSImputer(object):
    """
        Custom imputer that transforms select columns, normalizes all columns, and
        then does a row-wise mean imputation on the tranformed normalized result.

        Any columns not included in the df used to fit the imputer are dropped from
        imputed (transformed) results.

        Any rows missing all feature values are dropped. This can cause problems if
        you also have a target variable. In that case, in the dataframe passed to
        the impute method include a columns for the target variable and specify the
        target_column name. A target column should not be included when fitting.
    """

    def __init__(
        self, transform_patterns=[], transform_type='none',
        include_isnan=True, target_column=None
    ):
        """
            transform_patterns is a list of patterns to match against the column
            names to determine whether to apply the transform (specified by the
            transform_type variable). For example, if
                transform_patter=['_a_', '_d_'] and transform_type='log'
            the columns '11_a_14', '28_a_14', '11_d_07', '28_d_07', '11_d_30', '28_d_30'
            would all be log-transformed before normalizing, fitting, and imputing.

            transform_types are 'log' and 'ihs' (inverse hyperbolic sine).
        """
        self.imp = None
        self.transform_patterns = transform_patterns
        self.transform_type = transform_type
        self.include_isnan = include_isnan
        self.target_column = target_column

    def fit(self, features_df):
        """
            Note: target column should not be included in features_df
        """
        self.features = {}
        if self.target_column is not None and self.target_column in features_df:
            X = features_df.drop(self.target_column, axis=1).copy()
        else:
            X = features_df.copy()

        for c in X.columns:
            for p in self.transform_patterns:
                no_transform = True
                if p in c:
                    no_transform = False
                    (avg, std) = self._get_mean_and_std(X[c], self.transform_type)
                    self.features[c] = {'transform': self.transform_type, 'mean': avg, 'std': std}
                    break
            if no_transform:
                (avg, std) = self._get_mean_and_std(X[c], 'none')
                self.features[c] = {'transform': 'none', 'mean': avg, 'std': std}

    def impute(self, df):
        # TODO check that self.imp and self.features are not None
        df_normalized = pd.DataFrame(index=df.index)
        df_isnan = pd.DataFrame(index=df.index)
        df_index = df.index
        if self.target_column in df.columns:
            y = df[[self.target_column]]
            X = df.drop(self.target_column, axis=1)
        else:
            y = None
            X = df
        for c, params in self.features.iteritems():
            # TODO check that c in X.columns and add it if it isn't
            df_normalized[c] = self._transform(X[c], params['transform'])
            if self.include_isnan:
                df_isnan[c + '_nan'] = np.isnan(X[c])

        # remove empty rows
        df_normalized.replace([-np.inf, np.inf], np.nan, inplace=True)
        empty_rows = df_normalized[np.isnan(df_normalized.sum(axis=1))].index
        df_normalized.drop(empty_rows, inplace=True)
        df_isnan.drop(empty_rows, inplace=True)

        feature_type_dict = self._group_by_feature_type(df_normalized)
        for k, v in feature_type_dict.iteritems():
            empty_rows = np.isnan(df_normalized[v].sum(axis=1, min_count=1))
            df_normalized.loc[empty_rows, v] = MISSING_VALUE_FILLER
            imp = Imputer(np.nan, 'mean', axis=1)
            df_normalized[v] = imp.fit_transform(df_normalized[v])
        df_normalized[v].head()

        df_normalized_imputed = pd.DataFrame(
            df_normalized,
            index=df_index,
            columns=df_normalized.columns
        )

        if self.include_isnan:
            df_normalized_imputed = pd.concat([df_normalized_imputed, df_isnan], axis=1)
        if y is not None:
            df_normalized_imputed = pd.concat([df_normalized_imputed, y.drop(empty_rows)], axis=1)
        return df_normalized_imputed

    def _group_by_feature_type(self, df):
        feature_type_dict = {}
        for c in df.columns:
            match = re.match(r'.*_(.*?)_.*', c, re.M | re.I)
            if match is not None:
                feature_type = match.group(1)
                if feature_type in feature_type_dict:
                    feature_type_dict[feature_type].append(c)
                else:
                    feature_type_dict[feature_type] = [c]
        return feature_type_dict

    def _get_mean_and_std(self, x, transform_type):
        x = self._transform(x, transform_type)
        return (np.mean(x), np.std(x))

    def _transform(self, x, transform_type):
        if transform_type == 'log':
            # make the array log tranforamable (this is a hack)
            x.loc[x <= 0, ] = 0.00001
            return np.log(x)
        elif transform_type == 'ihs':
            return np.arcsinh(x)
        elif transform_type == 'none':
            return x
        raise Exception('transform_type must be log, ihs, or none')

    def _normalize(self, x, params):
        x = self._transform(x, params['transform'])
        return (x - params['mean']) / params['std']
