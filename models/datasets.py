import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class UsersProfilesDataset(Dataset):
    """
    """
    def __init__(self, annotations: pd.DataFrame, transformer=None):
        """

        """
        super().__init__()
        self.annotations = annotations
        self.transformer = transformer

    def __len__(self) -> int:
        """

        """
        return self.annotations.shape[0]

    def __getitem__(self, index: int) -> (np.array, int):
        """

        """

        # Load users
        users = torch.tensor(self.load_users(index).values, dtype=torch.float)
        
        # Load target
        target = torch.tensor(self.load_profiles(index).values, dtype=torch.float)

        return users, target

    def load_users(self, index: int) -> float:
        """

        """
        return self.annotations.iloc[index][[x for x in self.annotations.columns if x not in ['A', 'B', 'C', 'D', 'E']]]
    
    def load_profiles(self, index: int) -> float:
        """

        """
        return self.annotations.iloc[index][['A', 'B', 'C', 'D', 'E']]
