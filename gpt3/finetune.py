# Check this guide for how to run GPT-3 finetuning: https://beta.openai.com/docs/guides/fine-tuning
# We will be using text-curie-001

import json
import openai
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
FINE_TUNED_MODEL_NAME = ''

def model_predict(postprocess=True):
    predictions = []
    targets = []
    with open("gpt3_ft_data_test.jsonl", 'r') as f:
        for line in f:
            d = json.loads(line)

            openai.Completion.create(
                model=FINE_TUNED_MODEL_NAME,
                prompt=d['prompt'])
            # TODO: add other completion parameters
            generation_output = generation_output['choices'][0]['text']
            generation_output = generation_output.strip()
            if postprocess:
                generation_output = utils._postprocess_generations(generation_output)

            predictions.append(generation_output)
            targets.append(d['completion'])

    score = utils.get_bleu(predictions, targets)
    return predictions, score


if __name__ == '__main__':
    # make_ft_data('test')
    pass

