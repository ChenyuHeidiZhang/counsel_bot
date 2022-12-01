from typing import Dict, List, Optional, Tuple
import numpy as np
import random

import argparse
import json
import os
import openai
import tqdm
import sys
sys.path.append("..")
import time
from gpt2.dataloader import CounselChatMetaDataset, NUM_TRAIN_TOPICS, NUM_VAL_TOPICS, NUM_TEST_TOPICS
from gpt2 import utils

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='text-curie-001')
parser.add_argument('--k', default=1)
parser.add_argument('--max_tokens', default=512)
parser.add_argument('--temperature', default=0.8)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

CLASSIFY_TOPIC = True
RESULTS_PATH = f'results/icl/classify_topics={CLASSIFY_TOPIC}'

def get_icl_prompts(
    support_inputs: List[str],
    support_labels: List[str],
    test_input: str) -> str:
    """
    Take a list of contexts and combine them into k-shot prompts.

    Args:
      support_inputs: The k questions used for in-context learning (k may be zero!)
      support_labels: The k responses used for in-context learning (k may be zero!)
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


def run_icl(k, n_val=128, postprocess=True):
    results = {}

    if args.debug:
        n_val = 1
    dataset = CounselChatMetaDataset(num_support=k, num_query=1, classify_topic=CLASSIFY_TOPIC)

    split_idxs = range(
        NUM_TRAIN_TOPICS + NUM_VAL_TOPICS,
        NUM_TRAIN_TOPICS + NUM_VAL_TOPICS + NUM_TEST_TOPICS
    )

    print(f'Running in-context learning with {args.model} with k={k}')
    targets = []
    predictions = []
    pbar = tqdm.tqdm(list(range(n_val)))
    for row in pbar:
        topic_idx = np.random.choice(split_idxs)
        inp_support, out_support, inp_query, out_query = dataset[topic_idx]

        targets.append(out_query[0])

        prompt = get_icl_prompts(inp_support, out_support, inp_query[0])
        time.sleep(4)

        generation_output = openai.Completion.create(engine=args.model,
                                                     prompt=prompt,
                                                     max_tokens=int(args.max_tokens),
                                                     temperature=args.temperature,
                                                     top_p=0.9,
                                                     frequency_penalty=0.0,
                                                     presence_penalty=0.1,
                                                     best_of=1,
                                                     stop=None,
                                                     logprobs=0,  # log probability of top tokens
                                                    )

        generation_output = generation_output.choices[0].text
        if postprocess:
            generation_output = utils._postprocess_generations(generation_output)

        predictions.append(generation_output)

        # print('PROMPT:')
        # print(prompt)
        # print('PREDICTION:')
        # print(generation_output)
        # print('TARGET:')
        # print(targets[-1])
        # print('================')

        metric = utils.get_bleu(predictions, targets)

        pbar.set_description(f'Eval: {metric:.04f}')
        results[prompt] = {'PREDICTION': generation_output, 'TARGET': targets[-1]}

    results['metric'] = metric
    print('Evaluation results:', metric)

    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    filename = '_'.join(['icl', args.model, str(k)])
    with open(f'{RESULTS_PATH}/{filename}.json', 'a') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    run_icl(int(args.k))
