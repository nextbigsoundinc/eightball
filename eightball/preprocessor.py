import pandas as pd


class Preprocessor(object):
    def make_others(self, df, column_name, min_count):
        """Regroup categories with small instance count into an "Other" category

        Args:
        df: dataframe containing column that needs to be regrouped
        column_name (str): name of column to regroup
        min_count (int): if a category instance from `column_name` has less than
            min_count instances, it will be regrouped into the "Other" category

        Returns:
        dataframe identical to df, but with instances from `column_name` column
            with counts less than min_count replaced by "Other".
        """
        df_copy = df.copy()
        cnts = df.groupby(column_name).size()
        others = []
        for label, cnt in cnts.iteritems():
            if cnt < min_count:
                others.append(label)
        df_copy.loc[df[column_name].isin(others), column_name] = 'Other'
        return df_copy

    def encode_categories(self, df, cat_cols):
        """Replaces a set of categorical columns with a set of binary (dummy) columns

        Args:
        df: dataframe containing column to be replaced
        cat_cols (list of str): list of column names to replace

        Returns:
        dataframe identical to df, but with each column in cat_cols replaced with
            a set of columns for each instance within all of the columns.
        """
        df = df.copy()
        return pd.get_dummies(df, columns=cat_cols, prefix=cat_cols)

    def fit(self, df):
        pass

    def transform(self, df):
        return df
