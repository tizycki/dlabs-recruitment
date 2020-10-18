import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class LogTransformerDF(BaseEstimator, TransformerMixin):
    """
    Custom log transformer for specified columns in DataFrame
    Args:
        features: list of features which has to be log-transformed (np.log)
    Returns:
        pd.DataFrame with transformed features (suffix: '_log') and dropped original values
    """
    
    def __init__(self, features):
        self.__transform_features = features
        self.__aggregates = {}
        self.__features = []

    def fit(self, X, y=None):
        self.__features = X.columns.tolist()

        for feature in self.__transform_features:
            self.__aggregates[feature] = X[feature].min()
        return self

    def get_feature_names(self):
        return self.__features

    def transform(self, X, y=None):
        X_copy = X.copy()
        
        for feature in self.__transform_features:
            X_copy[feature + '_log'] = np.log(X_copy[feature] - self.__aggregates[feature] + 1)
            X_copy = X_copy.drop(feature, axis=1)

        self.__features = X_copy.columns.tolist()

        return X_copy
    
    
class OneHotEncoderDF(BaseEstimator, TransformerMixin):
    """
    
    Args:
        feature_names: list of feature names that were one-hot-encoded
        one_hot_encoder: one-hot-encoder that encoded original features (usually previous step in pipeline)
    Returns:
        pd.DataFrame with one-hot-encoded features with column names that specifies original column name and class
    """
    
    def __init__(self, feature_names):
        self.__feature_names = feature_names
        self.__encoder = OneHotEncoder(handle_unknown='ignore')
        self.__col_names = []
    
    def fit(self, X):
        X_copy = X[self.__feature_names].copy()

        self.__encoder.fit(X_copy)
        
        for feature, categories in zip(self.__feature_names, self.__encoder.categories_):
            for category in categories:
                self.__col_names.append(str(feature + '_' + str(category).replace('.', '_')).lower())
                
        return self
    
    def get_feature_names(self):
        return self.__col_names
    
    def transform(self, X):
        X_copy = X.copy()
        
        encodings = self.__encoder.transform(X_copy[self.__feature_names]).toarray()
        
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

    """
    
    def __init__(self, agg_feature_names, rank_feature_name):
        self.__agg_feature_names = agg_feature_names
        self.__rank_feature_name = rank_feature_name
        self.__map_config = {}
    
    def fit(self, X):
        X_copy = X.copy()
        
        X_copy['rank'] = X_copy.groupby(self.__agg_feature_names)[self.__rank_feature_name].rank('dense', ascending=False)
        
        df_mapping = X_copy[self.__agg_feature_names + [self.__rank_feature_name] + ['rank']].drop_duplicates()
        df_mapping = df_mapping.set_index(self.__agg_feature_names + [self.__rank_feature_name])
 
        self.__map_config = df_mapping.to_dict()
            
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.__rank_feature_name + '_rank'] = X_copy.set_index(self.__agg_feature_names + [self.__rank_feature_name]).index.map(self.__map_config['rank'])
        X_copy[self.__rank_feature_name + '_rank'].fillna(value=-1, inplace=True)
        
        return X_copy.drop(columns=self.__rank_feature_name)