from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import dataset, sampler, dataloader


def add_prefixes(x: List[str], y: List[str]) -> Tuple[List[str], List[str]]:
    input_prefix = 'Question: '
    label_prefix = ' Response:'
    label_suffix = ''

    x = [input_prefix + x_.replace('\n', ' ') + label_prefix for x_ in x]
    y = [' ' + y_.replace('\n', ' ') + label_suffix for y_ in y]

    return x, y


class CounselChatFtDataset(dataset.Dataset):
    """
    Counsel chat dataset for finetuning.
    Each element of the dataset is a (question, response) pair.
    """
    def __init__(self, n_train, n_val):
        pass

    def __getitem__(self, idx):
        pass



class CounselChatMetaDataset(dataset.Dataset):
    """
    Counsel chat dataset for meta learning.
    Each element of the dataset is a task.
    Each task consists of k+1 (question, response) pairs of a given topic.
    """
    def __init__(self, num_support, num_query=1):
        pass
    
    def __getitem__(self, class_idxs):
        pass

# TODO: do we need the sampler?
class CounselChatSampler(sampler.Sampler):
    """Samples task specification keys for an CounselChatMetaDataset."""

    def __init__(self, split_idxs, num_tasks):
        """Inits CounselChatSampler.

        Args:
            split_idxs (range): indices that comprise the
                training/validation/test split
            num_tasks (int): number of tasks to sample
        """
        super().__init__(None)
        self._split_idxs = split_idxs
        self._num_tasks = num_tasks

    def __iter__(self):
        return (
            np.random.default_rng().choice(
                self._split_idxs,
                size=1,
                replace=False
            ) for _ in range(self._num_tasks)
        )

    def __len__(self):
        return self._num_tasks


def get_counselchat_meta_learning_dataloader(
        split,
        batch_size,
        num_support,
        num_query,
        num_tasks_per_epoch
):
    pass
