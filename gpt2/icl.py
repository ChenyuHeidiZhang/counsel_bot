from typing import Dict, List, Optional, Tuple
import torch
import transformers
import numpy as np
import random

import argparse
from collections import defaultdict
import json
import os
import tqdm

import utils
from dataloader import CounselChatMetaDataset, NUM_TRAIN_TOPICS, NUM_VAL_TOPICS, NUM_TEST_TOPICS

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='med')
parser.add_argument('--k', default=1)
parser.add_argument('--num_val', default=128)
parser.add_argument('--postprocess', default=True)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--repeats', default=1, type=int)
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

CLASSIFY_TOPIC = False
RESULTS_PATH = f'results/icl/classify_topics={CLASSIFY_TOPIC}'
MAX_NUM_SENTS = 5  # max # sentences for each input question / response; None for k=0, 5 for k=1 and k=2, 3 for k=4
MAX_TOKENS = 1024  # max # output tokens; 512 is good enough for k=0
# Note: MAX_TOKENS has to be at least larger than the input prompt size
DEVICE = torch.device(args.device)


def get_icl_prompts(
    support_inputs: List[str],
    support_labels: List[str],
    test_input: str) -> str:
    """
    Take a list of contexts and combine them into k-shot prompts.

    Args:
      support_inputs: The k inputs used for in-context learning (k may be zero!)
      support_labels: The k labels used for in-context learning (k may be zero!)
      test_input: The input we are evaluating on

    Returns:
      A string containing the complete input to the model.
    """
    prompt = 'You are a counselor and your client approachs you with their concern.\n'

    input_labels = [(inp, lab) for inp, lab in zip(support_inputs, support_labels)]
    random.shuffle(input_labels)

    for (inp, lab) in input_labels:
        prompt +=  inp + ' ' + lab + "\n"
    prompt += test_input

    return prompt


def run_icl(model_name: str, k: int, n_val: int):
    results = {}
    print(f'Loading model {model_name}...')
    model, tokenizer = utils.get_model_and_tokenizer(model_name, transformers.AutoModelForCausalLM)
    model.to(DEVICE)

    if args.debug:
        n_val = 1
    dataset = CounselChatMetaDataset(
        num_support=k, num_query=1,
        classify_topic=CLASSIFY_TOPIC, num_sents_to_shorten_to=MAX_NUM_SENTS)

    # split_idxs = range(
    #     NUM_TRAIN_TOPICS,
    #     NUM_TRAIN_TOPICS + NUM_VAL_TOPICS
    # )
    split_idxs = range(
        NUM_TRAIN_TOPICS + NUM_VAL_TOPICS,
        NUM_TRAIN_TOPICS + NUM_VAL_TOPICS + NUM_TEST_TOPICS
    )

    print(f'Running in-context learning with {model_name} with k={k}')
    targets = []
    predictions = []
    pbar = tqdm.tqdm(list(range(n_val)))
    for row in pbar:
        topic_idx = np.random.choice(split_idxs)
        inp_support, out_support, inp_query, out_query = dataset[topic_idx]

        targets.append(out_query[0])

        prompt = get_icl_prompts(inp_support, out_support, inp_query[0])
        decoded_prediction = utils.model_generate(tokenizer, model, prompt, DEVICE, MAX_TOKENS)
        if args.postprocess:
            decoded_prediction = utils.batch_postprocess_generations(decoded_prediction)
        predictions.extend(decoded_prediction)

        # print('PROMPT:')
        # print(prompt)
        # print('PREDICTION:')
        # print(decoded_prediction)
        # print('TARGET:')
        # print(targets[-1])
        # print('================')

        metric = utils.get_bleu(predictions, targets)
        pbar.set_description(f'Eval: {metric:.04f}')
        results[prompt] = {'PREDICTION': predictions[-1], 'TARGET': targets[-1]}

    print('Evaluation results:', metric)

    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    experiment_name = '_'.join([model_name, str(k)])
    with open(f'{RESULTS_PATH}/metric_{experiment_name}.json', 'w') as f:
        json.dump({'metric': metric}, f)
    with open(f'{RESULTS_PATH}/outputs_{experiment_name}.json', 'w') as f:
        json.dump(results, f, indent=4)


def run():
    run_icl(args.model, args.k, args.num_val)


if __name__ == '__main__':
    run()
