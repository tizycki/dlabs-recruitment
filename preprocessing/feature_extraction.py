import multiprocessing as mp
import numpy as np
import pandas as pd
import datetime
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer to extract features from browser's fingerprint and age from DOB.
    """
    def __init__(self, n_jobs=-1) -> None:
        """
        Transformer initialization
        :param n_jobs: number of threads to use (-1 = all available)
        """
        self.n_jobs = n_jobs

    def fit(self, X):
        """
        Fit function of transformer
        :param X: Dataframe for transformer's fit
        """
        return self

    def transform(self, X) -> pd.DataFrame:
        """
        Transform function of transformer
        :param X: Dataframe to transform
        :return: transformed DataFrame
        """
        X_copy = X.copy()
        X_copy = parallelize_dataframe(X_copy, extract_features, self.n_jobs)
        
        return X_copy
    
    
def extract_features(df) -> pd.DataFrame:
    """
    Function extract features from browser's fingerprint and age from DOB
    :param df: Dataframe to apply function
    :return: Dataframe with original and extracted features (source columns of extracted features are dropped)
    """
    # Extract features from UserBrowser
    df = pd.concat([df, df.UserBrowser.apply(browser_extractor)], axis=1)
    df.drop(columns=['UserBrowser'], inplace=True)
    
    # Calculate age
    df['age'] = datetime.datetime.now().year - df['D02'].astype(int)
    df.drop(columns=['D02'], inplace=True)

    return df


def parallelize_dataframe(df, func, n_jobs=-1) -> pd.DataFrame:
    """
    Function to parallelize mapping function to Dataframe
    :param df: Dataframe to apply function
    :param func: function to apply
    :param n_jobs: number of threads used for processing (-1 = all available)
    :return: transformed DataFrame
    """
    # Get number of available cores
    cores = mp.cpu_count()

    # Set number of data splits
    if n_jobs <= -1:
        partitions = cores
    elif n_jobs <= 0:
        partitions = 1
    else:
        partitions = min(n_jobs, cores)
    
    # Split dataframe into batches
    df_split = np.array_split(df, partitions)
    pool = mp.Pool(partitions)
    
    # Apply function in parallel and concatenate results
    df = pd.concat(pool.map(func, df_split))
    
    # Close processes
    pool.close()
    pool.join()
    
    return df


def browser_extractor(browser_fingerprint) -> pd.Series:
    """
    Function to extract browser name, version and device type from browser's fingerprint
    :param browser_fingerprint: fingerprint of browser
    :return: Series with extracted features
    """
    browser_info = browser_fingerprint.split(' ')[:3]
    
    if len(browser_info) == 3:
        browser_version = browser_info[1].split('.')
        browser_version = browser_version[0] + '.' + ''.join([x.ljust(3, '0')[:3] for x in browser_version[1:]])
        
        output = {
            'browser_name': browser_info[0],
            'browser_version': float('0' + browser_version),
            'device_type': browser_info[2].replace('(', '').replace(')', '')
        }
    else:
        output = {
            'browser_name': '',
            'browser_version': 0,
            'device_type': ''
        }
        
    return pd.Series(output)
