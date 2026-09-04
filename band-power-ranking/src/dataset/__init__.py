import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentations import EEGRankAugmenter
from .tuev_pretrain_dataset import TUEVPretrainDataset
from .tuev_downstream_dataset import TUEVDownstreamDataset, CLASS_NAMES, LABEL_MAP


class FastSubset(Dataset):
    """Lightweight Subset that stores indices as np.array (not Python list).
    
    Unlike torch.utils.data.Subset which stores a Python list of indices,
    this uses a numpy int64 array. On Windows, multiprocessing uses 'spawn'
    which pickles the entire Dataset object. A Python list of 180K ints
    generates a huge pickle payload that can deadlock the IPC pipe.
    A numpy array pickles efficiently via its buffer protocol.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.array(indices, dtype=np.int64)

    def __getitem__(self, idx):
        return self.dataset[int(self.indices[idx])]

    def __len__(self):
        return len(self.indices)


__all__ = [
    'EEGRankAugmenter',
    'TUEVPretrainDataset',
    'TUEVDownstreamDataset',
    'FastSubset',
    'CLASS_NAMES',
    'LABEL_MAP'
]
