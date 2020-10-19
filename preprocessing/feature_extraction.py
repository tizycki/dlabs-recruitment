import multiprocessing as mp
import numpy as np
import pandas as pd
import datetime
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    TODO:
    """
    
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = parallelize_dataframe(X_copy, extract_features, self.n_jobs)
        
        return X_copy
    
    
def extract_features(df):
    # Extract features from UserBrowser
    df = pd.concat([df, df.UserBrowser.apply(browser_extractor)], axis=1)
    df.drop(columns=['UserBrowser'], inplace=True)
    
    # Calculate age
    df['age'] = datetime.datetime.now().year - df['D02'].astype(int)
    df.drop(columns=['D02'], inplace=True)

    return df
    
    
def parallelize_dataframe(df, func, n_jobs=-1):
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
    
def browser_extractor(browser_fingerprint):
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