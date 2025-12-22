"""
Base HEP Dataset

Minimal abstract class showing the required interface for HEP point cloud datasets.
"""

from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class HEPDataset(Dataset, ABC):
    """
    Minimal interface for HEP point cloud datasets.
    
    Subclasses must implement:
        - __getitem__(idx) -> dict with at least 'coord' (N,3) and 'energy' (N,1)
        - __len__() -> int
    """

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        """
        Load a point cloud.

        Returns:
            dict with at least:
                - coord: (N, 3) float32 array of xyz coordinates
                - energy: (N, 1) float32 array of energy deposits
        """
        raise NotImplementedError