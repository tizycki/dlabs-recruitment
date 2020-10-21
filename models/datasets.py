import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class UsersProfilesDataset(Dataset):
    """
    Dataset of users' demographic data and their psychological profiles
    """
    def __init__(self, annotations: pd.DataFrame, transformer=None) -> None:
        """
        Dataset initialization
        :param annotations: DataFrame with demographic data and psychological profiles
        :param transformer: instance of transformer to apply
        """
        super().__init__()
        self.annotations = annotations
        self.transformer = transformer

    def __len__(self) -> int:
        """
        Function to get number of rows in dataset
        :return: number of rows in dataset
        """
        return self.annotations.shape[0]

    def __getitem__(self, index: int) -> (np.array, int):
        """
        Function to get one element of dataset
        :param index: index of DataFrame
        """
        # Load users
        users = torch.tensor(self.load_users(index).values, dtype=torch.float)
        
        # Load target
        target = torch.tensor(self.load_profiles(index).values, dtype=torch.float)

        return users, target

    def load_users(self, index: int) -> pd.Series:
        """
        Function to load demographic data of users
        :param index: index of DataFrame
        :return: Vector of demographic data
        """
        return self.annotations.iloc[index][[x for x in self.annotations.columns if x not in ['A', 'B', 'C', 'D', 'E']]]
    
    def load_profiles(self, index: int) -> pd.Series:
        """
        Function to load psychological profile of users
        :param index: index of DataFrame
        :return: Vector of psychological features
        """
        return self.annotations.iloc[index][['A', 'B', 'C', 'D', 'E']]
