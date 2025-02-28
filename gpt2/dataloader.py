from typing import List, Tuple
import os
import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import dataset, sampler, dataloader
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)

INPUT_PREFIX = 'Client: '
LABEL_PREFIX = ' You:'
LABEL_SUFFIX = ''

def add_prefixes(x: List[str], y: List[str]) -> Tuple[List[str], List[str]]:
    x = [INPUT_PREFIX + x_.replace('\n', ' ') + LABEL_PREFIX for x_ in x]
    y = [y_.replace('\n', ' ') + LABEL_SUFFIX for y_ in y]

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
        return {'x': self.data['x'][idx], 'y': self.data['y'][idx]}
        # return {'x': self.data[idx][0], 'y': self.data[idx][1]}

    def __len__(self):
        return len(self.data['x'])


NUM_TRAIN_TOPICS = 15
NUM_VAL_TOPICS = 2
NUM_TEST_TOPICS = 7

class CounselChatMetaDataset(dataset.Dataset):
    """
    Counsel chat dataset for meta learning.
    Each element of the dataset is a task.
    Each task consists of k+1 (question, response) pairs of a given topic.
    """
    def __init__(
        self, num_support, num_query=1, classify_topic=False, num_sents_to_shorten_to=None
    ):
        super().__init__()

        self.num_support = num_support
        self.num_query = num_query
        self.classify_topic = classify_topic

        # keep num_sents sentences maximum for each question and each response
        # num_sents = 2 for MAML, 4 for ICL with k=4, 5 for ICL with k=2, and None for k=1
        # TODO: find better way to shorten the training responses
        self.num_sents = num_sents_to_shorten_to

        data_path = '../data/meta_learn'
        self.topic_data_files = [os.path.join(data_path, f) for f in os.listdir(data_path)]
        np.random.shuffle(self.topic_data_files)

        self.all_topics = self.get_all_topics(self.topic_data_files)

        if classify_topic:
            # using multiple topics in support is only supported when there is only 1 query example
            assert num_query == 1
            self.embedding_model = SentenceTransformer('bert-base-nli-mean-tokens')
            with open("../topic-embeddings.json", 'r') as f:
                topic_embeddings = json.load(f)
            self.topic_embds = [topic_embeddings[
                file.split('/')[-1].split('.tsv')[0]] for file in self.all_topics]


    def get_all_topics(self, data_files):
        '''Get all topics.
        Ignore topics where the number of unique questions are not enough for meta-training.
        '''
        all_topics = []
        for file in data_files:
            df = pd.read_csv(file, delimiter='\t', encoding='utf-8')
            unique_questions = df.iloc[:,0].unique()
            if len(unique_questions) < self.num_support + self.num_query:
                continue

            # topic = file.split('/')[-1].split('.tsv')[0]
            all_topics.append(file)
        return all_topics

    def read_topic_file(self, file):
        '''Read data from file.
        Returns a map from question to list of corresponding responses
        '''
        df = pd.read_csv(file, delimiter='\t', encoding='utf-8')
        questions = list(df.iloc[:, 0])
        responses = list(df.iloc[:, 1])

        if self.num_sents:
            questions = [' '.join(sent_tokenize(q)[:self.num_sents]) for q in questions]
            responses = [' '.join(sent_tokenize(r)[:self.num_sents]) for r in responses]
        qs, rs = add_prefixes(questions, responses)

        formatted_data = {}
        for i, q in enumerate(qs):
            if q in formatted_data:
                formatted_data[q].append(rs[i])
            else:
                formatted_data[q] = [rs[i]]
        return formatted_data

    def find_topic_dist(self, questions):
        '''questions: a list of questions to find topics for;
        in our case it always has length 1.
        Returns a distribution over self.all_topics.
        '''
        question_embds = self.embedding_model.encode(questions)
        sim = cosine_similarity(question_embds, self.topic_embds)  # (num_sents, num_topics)
        # print(np.array(self.all_topics)[np.argmax(sim[0])])
        return sim[0] / np.sum(sim[0])
        # print(sim[0] / np.sum(sim[0]))
        # print(np.exp(sim[0]) / sum(np.exp(sim[0])))
        # return np.exp(sim[0]) / sum(np.exp(sim[0]))  # use the softmax function

    def __getitem__(self, topic_idx):
        """Constructs a task. Questions should be different for entries in each task.
        Returns:
            questions_support (num_support,)
            responses_support (num_support,)
            questions_query (num_query,)
            responses_query (num_query,)
        """
        topic_filepath = self.all_topics[topic_idx]
        # print('topic:', topic_filepath)
        formatted_data = self.read_topic_file(topic_filepath)  # question mapped to responses
        all_questions = np.array(list(formatted_data.keys()))
        if not self.classify_topic:
            # sample questions
            question_sampled_idxs =  np.random.choice(
                len(all_questions), size=self.num_support+self.num_query, replace=False)

            questions_support = all_questions[question_sampled_idxs[:self.num_support]]
            questions_query = all_questions[question_sampled_idxs[self.num_support:]]

            responses_support = []
            for q in questions_support:
                responses_support.append(np.random.choice(formatted_data[q]))  # choose one
            responses_query = []
            for q in questions_query:
                responses_query.append(np.random.choice(formatted_data[q]))
            return list(questions_support), responses_support, list(questions_query), responses_query
        else:
            question_query = np.random.choice(all_questions)
            topic_dist = self.find_topic_dist([question_query])
            # Sample support topics according to the classification distribution
            support_topics = np.random.choice(
                self.all_topics, size=self.num_support, replace=True, p=topic_dist)
            questions_support = []
            responses_support = []
            for topic in support_topics:
                topic_data = self.read_topic_file(topic)
                q_sup = np.random.choice(list(topic_data.keys()))
                while q_sup in questions_support or q_sup == question_query:
                    q_sup = np.random.choice(list(topic_data.keys()))
                r_sup = np.random.choice(topic_data[q_sup])
                questions_support.append(q_sup)
                responses_support.append(r_sup)

            response_query = np.random.choice(formatted_data[question_query])

            # print(questions_support, responses_support, [question_query], [response_query])
            return questions_support, responses_support, [question_query], [response_query]

    def __len__(self):
        return len(self.all_topics)  # currently 31 topics max


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
        num_tasks_per_epoch,
        classify_topic=False,
        num_sents_to_shorten_to=None
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
        dataset=CounselChatMetaDataset(num_support, num_query, classify_topic, num_sents_to_shorten_to),
        batch_size=batch_size,
        sampler=CounselChatSampler(split_idxs, num_tasks_per_epoch),
        # num_workers=8,
        collate_fn=identity,
        pin_memory=torch.cuda.is_available()
    )


# if __name__ == "__main__":
#     dataset = CounselChatMetaDataset(num_support=4)
#     print(dataset.all_topics)
#     print(len(dataset))  # there are 24 topics that have at least 5 distinct questions