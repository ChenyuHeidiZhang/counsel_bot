from typing import List, Tuple
import argparse
import torch
import transformers
import torch.nn as nn
import utils
import copy
import numpy as np
import os
import json
import tqdm
import random

from dataloader import CounselChatFtDataset

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='med')
parser.add_argument('--mode', default='all')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

DEVICE = torch.device(args.device)
MAX_TOKENS = 1024
EARLY_STOPPING_THRES = 0.1  # loss value for early stopping

class LoRAConv1DWrapper(nn.Module):
    def __init__(self, conv1dmodule: nn.Module, lora_rank: int):
        super().__init__()

        self.base_module = conv1dmodule

        shape = self.base_module.weight.shape  # [d1, d2]
        self.A = nn.Parameter(
            torch.zeros([shape[0], lora_rank], dtype=torch.float32, device=DEVICE))
        self.B = nn.Parameter(
            torch.empty([shape[1], lora_rank], dtype=torch.float32, device=DEVICE))
        nn.init.xavier_normal_(self.B)

    def forward(self, x):
        out = self.base_module(x) + torch.matmul(torch.matmul(x, self.A), self.B.transpose(0,1))
        return out


def parameters_to_fine_tune(model: nn.Module, mode: str) -> List:
    """
    Select the parameters in `model` that should be fine-tuned in mode `mode`.

    Args:
      model: the model we're fine-tuning
      mode: the fine-tuning mode we're using; may be 'all', 'last', 'first',
        'middle', or 'loraN' (where N is an integer)
    
    Returns:
      A list of nn.Parameters of `model` that should be fine-tuned in the given
        fine-tuning mode.
    """
    if mode == 'all':
        return model.parameters()
    elif mode == 'last':
        return model.transformer.h[-2:].parameters()  # last 2 transformer blocks of GPT-2
    elif mode == 'first':
        return model.transformer.h[:2].parameters()  # first 2 transformer blocks of GPT-2
    elif mode == 'middle':
        middle_block = (len(model.transformer.h)-1) // 2
        return model.transformer.h[middle_block:middle_block+2].parameters()
    elif mode.startswith('lora'):
        lst = []
        for mod in model.modules():
            if isinstance(mod, LoRAConv1DWrapper):
                lst.append(mod.A)
                lst.append(mod.B)
        return lst
    else:
        raise NotImplementedError()


def evaluate(model, tokenizer, val_data):
    targets = []
    predictions = []
    pbar = tqdm.tqdm(list(range(len(val_data['x']))))

    for row in pbar:
        test_input = val_data['x'][row]
        targets.append(val_data['y'][row])
        input_ids = tokenizer(test_input, return_tensors='pt').input_ids.to(DEVICE)
        sampled_tokens = model.generate(input_ids, max_length=MAX_TOKENS)
        decoded = tokenizer.decode(sampled_tokens).strip()
        predictions.append(decoded)
    return utils.get_bleu(predictions, targets)


def ft_gpt2(model, tok, x, y, val_data, mode, batch_size=8, grad_accum=8):
    model = copy.deepcopy(model)

    if mode.startswith('lora'):
        for m in model.transformer.h:
            m.mlp.c_fc = LoRAConv1DWrapper(m.mlp.c_fc, int(mode[4:]))
            m.mlp.c_proj = LoRAConv1DWrapper(m.mlp.c_proj, int(mode[4:]))
            m.attn.c_attn = LoRAConv1DWrapper(m.attn.c_attn, int(mode[4:]))

    model.to(DEVICE)

    optimizer = torch.optim.Adam(parameters_to_fine_tune(model, mode), lr=2e-5)
    max_n = len(x) * 5
    pbar = tqdm.tqdm(range(max_n))
    idxs = []
    for step in pbar:
        model.train()

        if len(idxs) < batch_size // grad_accum:
            idxs = list(range(len(x)))
            random.shuffle(idxs)
        batch_idxs = idxs[:batch_size // grad_accum]
        idxs = idxs[batch_size // grad_accum:]

        batch_x = [x[id] for id in batch_idxs]
        batch_y = [y[id] for id in batch_idxs]
        tokenized_seqs = utils.tokenize_gpt2_batch(tok, batch_x, batch_y, DEVICE)
        output = model(**tokenized_seqs, use_cache=False)
        loss = output.loss / grad_accum
        loss.backward()
        if step % grad_accum == 0 and step > 0:
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f'Loss: {loss:.04f}')

        if step % (grad_accum * 50) == 0:
            with torch.inference_mode():
                model.eval()
                bleu_score = evaluate(model, tok, val_data)
                print(f'Step: {step}, Fine-tuning score: {bleu_score:.04f}')

        if loss <= EARLY_STOPPING_THRES:
            print('Early stopping!')
            break
    return model


def run_ft(model_name: str, mode: str, n_train: int = 512, n_val: int = 128):
    results = {}
    if args.debug:
        n_val = 1
    train = CounselChatFtDataset(split='train', num_data=n_train)
    val = train = CounselChatFtDataset(split='test', num_data=n_val)
    model, tokenizer = utils.get_model_and_tokenizer(model_name, transformers.AutoModelForCausalLM)

    print(f'Fine-tuning {model_name} with k={n_train} and mode={mode}')
    fine_tuned = ft_gpt2(model, tokenizer, train['x'], train['y'], val, mode)

    fine_tuned.eval()
    metric = evaluate(fine_tuned, tokenizer, val)
    results['_'.join([model_name, mode])] = metric

    print(results)
    question = 'ft'
    if not os.path.exists(f'results/{question}'):
        os.makedirs(f'results/{question}')

    for k_, v in results.items():
        with open(f'results/{question}/{k_}.json', 'w') as f:
            json.dump({'metric': v}, f)



if __name__ == '__main__':
    run_ft(args.model, args.mode)

