import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class LogTransformerDF(BaseEstimator, TransformerMixin):
    """
    Custom log transformer for specified columns in DataFrame. Output is DataFrame as well.
    """
    
    def __init__(self, features) -> None:
        """
        Initialization of transformer
        :param features: list of features which has to be log-transformed (np.log)
        """
        self.__transform_features = features
        self.__aggregates = {}
        self.__features = []

    def fit(self, X):
        """
        Fit function of transformer
        :param X: Dataframe to fit transformer
        """
        self.__features = X.columns.tolist()

        for feature in self.__transform_features:
            self.__aggregates[feature] = X[feature].min()
        return self

    def get_feature_names(self) -> list:
        """
        Function to get output feature names after transformation
        :return: list of feature names (string)
        """
        return self.__features

    def transform(self, X) -> pd.DataFrame:
        """
        Transform function of transformer
        :param X: Dataframe to apply transformation
        :return: pd.DataFrame with transformed features (suffix: '_log') and dropped original values
        """
        X_copy = X.copy()
        
        for feature in self.__transform_features:
            X_copy[feature + '_log'] = np.log(X_copy[feature] - self.__aggregates[feature] + 1)
            X_copy = X_copy.drop(feature, axis=1)

        self.__features = X_copy.columns.tolist()

        return X_copy
    
    
class OneHotEncoderDF(BaseEstimator, TransformerMixin):
    """
    Transformer to apply one-hot encoding. Wrapper of sklearn OneHotEncoder to return DataFrame and parse feature names
    """
    
    def __init__(self, feature_names) -> None:
        """
        Initialization of transformer
        param: feature_names: list of feature names that needs to be one-hot-encoded
        """
        self.__feature_names = feature_names
        self.__encoder = OneHotEncoder(handle_unknown='ignore')
        self.__col_names = []
    
    def fit(self, X):
        """
        Fit function of transformer
        :param X: Dataframe to fit transformer
        """
        # Copy input dataframe
        X_copy = X[self.__feature_names].copy()

        # Fit one-hot encoder
        self.__encoder.fit(X_copy)

        # Parse new feature names
        for feature, categories in zip(self.__feature_names, self.__encoder.categories_):
            for category in categories:
                self.__col_names.append(str(feature + '_' + str(category).replace('.', '_')).lower())
                
        return self
    
    def get_feature_names(self) -> list:
        """
        Function to get list of feature names after one-hot encoding
        :return: list of feature names after one-hot encoding
        """
        return self.__col_names
    
    def transform(self, X):
        """
        Transformation function of one-hot encoder.
        :param X: DataFrame to apply one-hot encoding
        :return: pd.DataFrame with one-hot-encoded features with column names that specifies original column name
            and class.
        """
        # Copy input DataFrame
        X_copy = X.copy()

        # Apply one-hot encoder
        encodings = self.__encoder.transform(X_copy[self.__feature_names]).toarray()

        # Concat results with new feature names
        output = pd.concat(
            [
                X_copy.drop(columns=self.__feature_names),
               pd.DataFrame(data=encodings, index=X_copy.index, columns=self.__col_names) 
            ],
            axis=1
        )
        
        return output
    
    
class OrdinalEncoderDF(BaseEstimator, TransformerMixin):
    """
    Ordinal transformer that performs ranking per each group of specified columns
    """

    def __init__(self, agg_feature_names, rank_feature_name) -> None:
        """
        Initialization of transformer
        :param agg_feature_names: list of column names to group by
        :param rank_feature_name: name of feature to rank
        """
        self.__agg_feature_names = agg_feature_names
        self.__rank_feature_name = rank_feature_name
        self.__map_config = {}
    
    def fit(self, X):
        """
        Fit function of transformer
        :param X: input DataFrame to fit transformer
        """
        # Copy input DataFrame
        X_copy = X.copy()

        # Group data and apply ranking
        X_copy['rank'] = X_copy.groupby(self.__agg_feature_names)[self.__rank_feature_name].rank('dense', ascending=False)

        # Save results for further use
        df_mapping = X_copy[self.__agg_feature_names + [self.__rank_feature_name] + ['rank']].drop_duplicates()
        df_mapping = df_mapping.set_index(self.__agg_feature_names + [self.__rank_feature_name])
 
        self.__map_config = df_mapping.to_dict()
            
        return self
    
    def transform(self, X) -> pd.DataFrame:
        """
        Transform function of transformer. Merges mappings with new data.
        :param X: input DataFrame to apply transformation
        :return: DataFrame with new feature - ranking of specified ordinal feature
        """
        # Copy input DataFrame
        X_copy = X.copy()

        # Merge data with saved results/mappings
        X_copy[self.__rank_feature_name + '_rank'] = X_copy\
            .set_index(self.__agg_feature_names + [self.__rank_feature_name])\
            .index.map(self.__map_config['rank'])

        # Fill unseen values with constant value
        X_copy[self.__rank_feature_name + '_rank'].fillna(value=-1, inplace=True)
        
        return X_copy.drop(columns=self.__rank_feature_name)
