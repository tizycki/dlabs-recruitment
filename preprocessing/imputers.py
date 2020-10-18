import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class SimpleImputerDF(BaseEstimator, TransformerMixin):
    """
    Simple imputer with constant strategy that Imputes all missing data in whole DataFrame.
    
    Args:
        strategy: imputation's strategy
            constant: fill column with fixed value
        fill_value: value that replaces missing values
    Returns:
        pd.DataFrame with imputed values
    """
    
    def __init__(self, strategy='constant', fill_value_str='unknown', fill_value_num=0):
        self.__strategy = strategy
        self.__fill_value_str = fill_value_str
        self.__fill_value_num = fill_value_num
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        if self.__strategy == 'constant':
            for col in X.columns:
                if X[col].dtype == object:
                    X[col] = X[col].fillna(self.__fill_value_str)
                else:
                    X[col] = X[col].fillna(self.__fill_value_num)
            return X
        else:
            raise NotImplementedError("That strategy is not implemented yet")