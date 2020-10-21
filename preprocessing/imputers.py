import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SimpleImputerDF(BaseEstimator, TransformerMixin):
    """
    Simple imputer with constant strategy that Imputes all missing data in a whole DataFrame.
    """
    def __init__(self, strategy='constant', fill_value_str='unknown', fill_value_num=0) -> None:
        """
        Initialization of Imputer
        :param strategy: imputation's strategy
        :param fill_value_str: fill value for object type columns
        :param fill_value_num: fill value for numerical type columns
        """
        self.__strategy = strategy
        self.__fill_value_str = fill_value_str
        self.__fill_value_num = fill_value_num
    
    def fit(self, X):
        """
        Fit function of Imputer
        :param X: Dataframe to fit transformer
        """
        return self
    
    def transform(self, X) -> pd.DataFrame:
        """
        Transform function of Imputer
        :param X: DataFrame to process (impute missing value)
        :return: DataFrame with imputed values
        """
        if self.__strategy == 'constant':
            for col in X.columns:
                if X[col].dtype == object:
                    X[col] = X[col].fillna(self.__fill_value_str)
                else:
                    X[col] = X[col].fillna(self.__fill_value_num)
            return X
        else:
            raise NotImplementedError("That strategy is not implemented yet")
