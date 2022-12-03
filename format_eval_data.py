import json
from nltk.tokenize import sent_tokenize

def format_gpt_outputs(num_sents=-1):
    with open(FILENAME, 'r') as f, open(OUTPUT_FILENAME, 'w') as out_f:
        results_d = json.load(f)
        for k, v in results_d.items():
            if k == "metric": continue
            question = k.split('Client: ')[-1].split('You:')[0].strip()
            response = v['PREDICTION']
            if num_sents != -1:
                response = ' '.join(sent_tokenize(response)[:num_sents])
            out_f.write(question)
            out_f.write('\n')
            out_f.write(response)
            out_f.write('\n=====\n')


def format_t5_outputs(num_sents=5):
    with open(FILENAME, 'r') as f, open(OUTPUT_FILENAME, 'w') as out_f:
        for line in f:
            results_d = json.loads(line)
            question = results_d['input']['inputs_pretokenized'].split('counselling: ')[-1]
            response = results_d['output']
            # Keep only the first 5 sentences of each response
            response = ' '.join(sent_tokenize(response)[:num_sents])
            out_f.write(question)
            out_f.write('\n')
            out_f.write(response)
            out_f.write('\n=====\n')

FILENAME = 'gpt2/results/ft/outputs_med_all_128.json'
OUTPUT_FILENAME = 'formatted_results/ft_128_gpt2_med.txt'

FILENAME = 'gpt2/results/icl/classify_topics=False/outputs_med_1.json'
OUTPUT_FILENAME = 'formatted_results/icl_1_gpt2_med.txt'
FILENAME = 'gpt2/results/icl/classify_topics=False/outputs_med_4.json'
OUTPUT_FILENAME = 'formatted_results/icl_4_gpt2_med.txt'

# FILENAME = 'gpt2/results/icl/classify_topics=True/outputs_med_4.json'
# OUTPUT_FILENAME = 'formatted_results/icl_4_classify_topics_gpt2_med.txt'

# FILENAME = 'gpt2/results/maml/support=1.model=med.mode=last/test_support=1.json'
# OUTPUT_FILENAME = 'formatted_results/maml_1_gpt2_med.txt'
# FILENAME = 'gpt2/results/maml/support=4.model=small.mode=last/test_support=4.json'
# OUTPUT_FILENAME = 'formatted_results/maml_4_gpt2_sm.txt'

# FILENAME = 'gpt3/results/finetune/finetune_gpt3.json'
# OUTPUT_FILENAME = 'formatted_results/ft_finetune_gpt3.txt'

# FILENAME = 'gpt3/results/icl/classify_topics=False/icl_text-curie-001_4.json'
# OUTPUT_FILENAME = 'formatted_results/icl_4_gpt3.txt'

# FILENAME = 'gpt3/results/icl/classify_topics=True/icl_text-curie-001_4.json'
# OUTPUT_FILENAME = 'formatted_results/icl_4_classify_topics_gpt3.txt'
format_gpt_outputs()

# FILENAME = 't5x/finetune-model/inference_eval/counsel_bot-1004000.jsonl'
# OUTPUT_FILENAME = 'formatted_results/t5x_counsel_bot-1004000.txt'
# format_t5_outputs()