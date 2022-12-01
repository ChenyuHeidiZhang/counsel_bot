# Check this guide for how to run GPT-3 finetuning: https://beta.openai.com/docs/guides/fine-tuning
# We will be using text-curie-001

import json
import openai
import time
import os
import sys
sys.path.append("..")
from gpt2 import utils

def make_prompt(question):
    prompt_text = 'You are a counselor and your client approachs you with their concern.\n'
    return prompt_text + 'Client: ' + question + '\nYou:'


def make_ft_data(split='train'):
    with open(f"../data/finetune/counselchat_{split}.tsv", 'r') as f, open(f'gpt3_ft_data_{split}.jsonl', 'w') as out:
        for line in f:
            question, response = line.split('\t')
            prompt = make_prompt(question)
            json.dump({"prompt": prompt, "completion": response.strip()}, out)
            out.write('\n')



# TODO: finetune the model with train data, and fill in the model name
FINE_TUNED_MODEL_NAME = 'curie:ft-personal-2022-11-28-23-41-16'

def model_predict(postprocess=True):
    predictions = []
    targets = []
    results = {}

    with open("gpt3_ft_data_test_prepared.jsonl", 'r') as f:
        for line in f:
            d = json.loads(line)
            time.sleep(2)

            generation_output = openai.Completion.create(
                model=FINE_TUNED_MODEL_NAME,
                prompt=d['prompt'],
                max_tokens=512,
                temperature=0.8,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.1,
                best_of=1,
                stop=None,
                logprobs=0)  # log probability of top tokens)
            # TODO: add other completion parameters
            generation_output = generation_output.choices[0].text
            if postprocess:
                generation_output = utils._postprocess_generations(generation_output)
            print(generation_output)
            predictions.append(generation_output)
            targets.append(d['completion'])
            score = utils.get_bleu(predictions, targets)
            print(score)
            print(len(predictions))
            results[d['prompt']] = {'PREDICTION': generation_output, 'TARGET': d['completion']}

    score = utils.get_bleu(predictions, targets)
    results['metric'] = score
    print('Evaluation results:', score)
    if not os.path.exists('results/finetune'):
        os.makedirs('results/finetune')

    filename = '_'.join(['finetune', 'gpt3'])
    with open(f'results/finetune/{filename}.json', 'a') as f:
        json.dump(results, f, indent=4)
    return predictions, score


if __name__ == '__main__':
    # make_ft_data('test')
    model_predict()
    # pass

