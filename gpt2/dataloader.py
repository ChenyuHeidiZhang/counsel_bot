from typing import List, Tuple
import os
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
    def __init__(self, split='train', num_data=None):
        '''
        split can be either train or test
        num_data is the number of examples to get; None would get all the data
        '''
        super().__init__()

        self.num_data = num_data
        self.data_file = f'../data/finetune/counselchat_{split}.tsv'
        self.data = self.read_tsv_file()

    def read_tsv_file(self):
        questions = []
        responses = []
        with open(self.data_file, 'r') as f:
            for line in f.readlines():
                question, response = line.split('\t')
                questions.append(question)
                responses.append(response)

        qs, rs = add_prefixes(questions[:self.num_data], responses[:self.num_data])
        return {'x': qs, 'y': rs}
        # return [(x, y) for x, y in zip(qs, rs)]

    def __getitem__(self, idx):
        print(idx)
        return {'x': self.data['x'][idx], 'y': self.data['y'][idx]}
        # return {'x': self.data[idx][0], 'y': self.data[idx][1]}

    def __len__(self):
        return len(self.data['x'])


NUM_TRAIN_TOPICS = 26
NUM_VAL_TOPICS = 5
NUM_TEST_TOPICS = 0  # will get test topics later

class CounselChatMetaDataset(dataset.Dataset):
    """
    Counsel chat dataset for meta learning.
    Each element of the dataset is a task.
    Each task consists of k+1 (question, response) pairs of a given topic.
    """
    def __init__(self, num_support, num_query=1):
        super().__init__()

        data_path = '../data/meta_learn'
        self.topic_data_files = [os.path.join(data_path, f) for f in os.listdir(data_path)]
        # TODO: try shuffling the topic files
        self.num_topics = len(self.topic_data_files)  # currently 31 topics total; the smallest topic has only 3 examples, so k<=2
        self.num_support = num_support
        self.num_query = num_query

    def read_topic_file(self, file):
        questions = []
        responses = []
        with open(file, 'r') as f:
            for line in f.readlines():
                question, response = line.split('\t')
                questions.append(question)
                responses.append(response)

        qs, rs = add_prefixes(questions, responses)
        return {'x': qs, 'y': rs}

    def __getitem__(self, topic_idx):
        """Constructs a task.
        Returns:
            questions_support (num_support,)
            responses_support (num_support,)
            questions_query (num_query,)
            responses_query (num_query,)
        """
        print('topic:', self.topic_data_files[topic_idx])
        topic_data = self.read_topic_file(self.topic_data_files[topic_idx])
        num_examples = len(topic_data['x'])
        sampled_idxs =  np.random.randint(
            low=0, high=num_examples, size=self.num_support+self.num_query)
        questions_support = np.array(topic_data['x'])[sampled_idxs[:self.num_support]]
        responses_support = np.array(topic_data['y'])[sampled_idxs[:self.num_support]]
        questions_query = np.array(topic_data['x'])[sampled_idxs[self.num_support:]]
        responses_query = np.array(topic_data['y'])[sampled_idxs[self.num_support:]]
        return questions_support, responses_support, questions_query, responses_query

# TODO: do we need the sampler?
class CounselChatSampler(sampler.Sampler):
    """Samples task specification keys for an CounselChatMetaDataset."""

    def __init__(self, split_idxs, num_tasks):
        """Inits CounselChatSampler.

        Args:
            split_idxs (range): topic indices that comprise the
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
            ) for _ in range(self._num_tasks)
        )

    def __len__(self):
        return self._num_tasks

def identity(x):
    return x

def get_counselchat_meta_learning_dataloader(
        split,
        batch_size,
        num_support,
        num_query,
        num_tasks_per_epoch
):

    if split == 'train':
        split_idxs = range(NUM_TRAIN_TOPICS)
    elif split == 'val':
        split_idxs = range(
            NUM_TRAIN_TOPICS,
            NUM_TRAIN_TOPICS + NUM_VAL_TOPICS
        )
    elif split == 'test':
        split_idxs = range(
            NUM_TRAIN_TOPICS + NUM_VAL_TOPICS,
            NUM_TRAIN_TOPICS + NUM_VAL_TOPICS + NUM_TEST_TOPICS
        )
    else:
        raise ValueError

    return dataloader.DataLoader(
        dataset=CounselChatMetaDataset(num_support, num_query),
        batch_size=batch_size,
        sampler=CounselChatSampler(split_idxs, num_tasks_per_epoch),
        # num_workers=8,
        collate_fn=identity,
        pin_memory=torch.cuda.is_available()
    )
