from typing import List, Tuple
import argparse
import copy
import json
import os
import torch
from torch import nn

import higher

from functools import partial
from torch.utils.checkpoint import checkpoint
torch.utils.checkpoint.checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)

import transformers
import numpy as np
import tensorboard
# import tensorflow as tf

import utils
from dataloader import get_counselchat_meta_learning_dataloader as get_dataloader

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='small')
parser.add_argument('--mode', default='last')
parser.add_argument('--log_dir', type=str, default=None,
                    help='directory to save to or load from')
parser.add_argument('--num_support', type=int, default=4,
                    help='number of support (question, response) pairs in a task')
parser.add_argument('--num_query', type=int, default=1,
                    help='number of query (question, response) pairs in a task')
parser.add_argument('--num_inner_steps', type=int, default=1,
                    help='number of inner-loop updates')
parser.add_argument('--inner_lr', type=float, default=0.01,
                    help='inner-loop learning rate initialization')
parser.add_argument('--learn_inner_lrs', default=True, action='store_true',
                    help='whether to optimize inner-loop learning rates')
parser.add_argument('--outer_lr', type=float, default=0.001,
                    help='outer-loop learning rate')
parser.add_argument('--batch_size', type=int, default=1,
                    help='number of tasks per outer-loop update')
parser.add_argument('--num_train_iterations', type=int, default=2000,
                    help='number of outer-loop updates to train for')
parser.add_argument('--test', default=False, action='store_true',
                    help='train or test')
parser.add_argument('--checkpoint_step', type=int, default=-1,
                    help=('checkpoint iteration to load for resuming '
                            'training, or for evaluation (-1 is ignored)'))

parser.add_argument('--device', default='cuda')

args = parser.parse_args()

DEVICE = torch.device(args.device)
POSTPROCESS = True
GRADIENT_CHECKPOINTING = True
MAX_NUM_SENTS = 3  # maximum number of sentences for each input question / response
MAX_TOKENS = 1024  # maximum number of tokens to generate
NUM_TEST_TASKS = 128 // args.num_query  # TODO: update this

SAVE_INTERVAL = 100
LOG_INTERVAL = 10
VAL_INTERVAL = LOG_INTERVAL * 10


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


class Gpt2MAML:
    """Trains and assesses a MAML."""

    def __init__(
            self,
            model_name,
            mode,
            num_inner_steps,
            inner_lr,
            learn_inner_lrs,
            outer_lr,
            log_dir,
            log_file
    ):
        """
        Inits MAML.
        The network is initialized with the parameters from GPT2.
        """
        model, tokenizer = utils.get_model_and_tokenizer(model_name, transformers.AutoModelForCausalLM)
        self._model = model.to(DEVICE)
        if GRADIENT_CHECKPOINTING:
            self._model.transformer.gradient_checkpointing = True

        self._tokenizer = tokenizer
        self._mode = mode
        self._num_inner_steps = num_inner_steps
        self._inner_lr = torch.tensor(inner_lr, requires_grad=learn_inner_lrs)
        self._outer_lr = outer_lr

        # optimize the initial params to be adapted
        # self.meta_opt = torch.optim.Adam(list(parameters_to_fine_tune(self._model, self._mode)), self._outer_lr)
        self.meta_opt = torch.optim.Adam(
            params=list(parameters_to_fine_tune(self._model, self._mode)) + [self._inner_lr],
            lr=self._outer_lr
        )

        self._log_dir = log_dir
        self._log_file = log_file

        self._start_train_step = 0


    def train(self, dataloader_train, dataloader_val):
        """Train the MAML.

        Consumes dataloader_train to optimize MAML meta-parameters
        while periodically validating on dataloader_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_train (DataLoader): loader for train tasks
            dataloader_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        print(f'Starting training at iteration {self._start_train_step}.')

        for i_step, task_batch in enumerate(
                dataloader_train,
                start=self._start_train_step
        ):
            inner_opt = torch.optim.Adam(params=parameters_to_fine_tune(self._model, self._mode),lr=self._inner_lr)

            self.meta_opt.zero_grad()

            qry_losses = []

            for task in task_batch:
                inp_support, out_support, inp_query, out_query = task
                tokenized_seqs = utils.tokenize_gpt2_batch(self._tokenizer, inp_support, out_support, DEVICE)
                tokenized_seqs_query = utils.tokenize_gpt2_batch(self._tokenizer, inp_query, out_query, DEVICE)

                with higher.innerloop_ctx(self._model, inner_opt, copy_initial_weights=False) as (maml_model, diffopt):
                    for _ in range(self._num_inner_steps):

                        spt_loss = maml_model(**tokenized_seqs).loss
                        diffopt.step(spt_loss)

                    qry_loss = maml_model(**tokenized_seqs_query).loss
                    qry_losses.append(qry_loss.detach().item())
                    # Update the model's meta-parameters to optimize the query losses across all of the tasks sampled in this batch.
                    # This unrolls through the gradient steps.
                    qry_loss.backward()

            self.meta_opt.step()
            outer_loss = np.mean(qry_losses)

            if i_step % LOG_INTERVAL == 0:
                # log = f'Iteration {i_step}: loss: {outer_loss.item():.3f}'
                log = f'Iteration {i_step}: loss: {outer_loss:.3f}'
                print(log)
                with open(self._log_file, 'a') as f:
                    f.write(log + '\n')

            if i_step % VAL_INTERVAL == 0:
                print('inner lr:', self._inner_lr)
                self.eval(dataloader_val, test=False)

            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)


    def eval(self, dataloader_val, test=False):
        losses = []
        query_scores = []
        output_records = {}

        inner_opt = torch.optim.Adam(
            params=parameters_to_fine_tune(self._model, self._mode), lr=self._inner_lr)

        for val_task_batch in dataloader_val:  # batch_size = 1
            val_inp_support, val_out_support, val_inp_query, val_out_query = val_task_batch[0]
            val_tokenized_seqs = utils.tokenize_gpt2_batch(self._tokenizer, val_inp_support, val_out_support, DEVICE)
            val_tokenized_seqs_query = utils.tokenize_gpt2_batch(self._tokenizer, val_inp_query, val_out_query, DEVICE)

            with higher.innerloop_ctx(self._model, inner_opt, track_higher_grads=False) as (maml_model, diffopt):
                # decoded_out = utils.model_generate_v2(self._tokenizer, maml_model, val_inp_query, DEVICE, MAX_TOKENS)
                # print(decoded_out)
                for _ in range(self._num_inner_steps):
                    spt_loss = maml_model(**val_tokenized_seqs).loss
                    diffopt.step(spt_loss)

                qry_loss = maml_model(**val_tokenized_seqs_query).loss
                # diffopt.step(qry_loss)
                losses.append(qry_loss.detach().item())

                decoded_out = utils.model_generate_v2(self._tokenizer, maml_model, val_inp_query, DEVICE, MAX_TOKENS)

            if POSTPROCESS:
                decoded_out = utils.batch_postprocess_generations(decoded_out)

            print("Input:", val_inp_query)
            print("Output:", decoded_out)
            print("Target:", val_out_query)
            score = utils.get_bleu(decoded_out, val_out_query)
            query_scores.append(score)

            if test:
                for i in range(len(val_inp_query)):
                    output_records[val_inp_query[i]] = {'PREDICTION': decoded_out[i], 'TARGET': val_out_query[i]}

        score_query = np.mean(query_scores)
        outer_loss = np.mean(losses)

        log = f'Validation: loss: {outer_loss:.3f}, score: {score_query:.3f}'
        print(log)
        with open(self._log_file, 'a') as f:
            f.write(log + '\n')

        if test:
            with open(os.path.join(self._log_dir, f'test_support:{args.num_support}.json'), 'w') as f:
                json.dump(output_records, f, indent=4)

            std = np.std(query_scores)
            mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
            print(
                f'Score over {NUM_TEST_TASKS} test tasks: '
                f'mean {score_query:.3f}, '
                f'95% confidence interval {mean_95_confidence_interval:.3f}'
            )


    def test(self, dataloader_test):
        """Evaluate the MAML on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        self.eval(dataloader_test, test=True)


    def load(self, checkpoint_step):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._model.load_state_dict(state['model_state_dict'])
            self._inner_lr = state['inner_lr']
            self.meta_opt.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """Saves parameters and optimizer state_dict as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        optimizer_state_dict = self.meta_opt.state_dict()
        torch.save(
            dict(model_state_dict=self._model.state_dict(),
                 inner_lr=self._inner_lr,
                 optimizer_state_dict=optimizer_state_dict),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')

def main(args):
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./results/maml/support={args.num_support}.model={args.model}.mode={args.mode}'  # pylint: disable=line-too-long
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'inner_steps:{args.num_inner_steps}.inner_lr:{args.inner_lr}.outer_lr:{args.outer_lr}.txt')
    print(f'log_file: {log_file}')

    maml = Gpt2MAML(
        args.model,
        args.mode,
        args.num_inner_steps,
        args.inner_lr,
        args.learn_inner_lrs,
        args.outer_lr,
        log_dir, log_file
    )

    if args.checkpoint_step > -1:
        maml.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        print(
            f'Training on {num_training_tasks} tasks with composition: '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_train = get_dataloader(
            'train',
            args.batch_size,
            args.num_support,
            args.num_query,
            num_training_tasks,
            num_sents_to_shorten_to=MAX_NUM_SENTS
        )
        dataloader_val = get_dataloader(
            'val',
            1,
            args.num_support,
            args.num_query,
            args.batch_size * 2,
            num_sents_to_shorten_to=MAX_NUM_SENTS
        )

        maml.train(dataloader_train, dataloader_val)
    else:
        print(
            f'Testing on tasks with composition '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_test = get_dataloader(
            'test',
            1,
            args.num_support,
            args.num_query,
            NUM_TEST_TASKS,
            num_sents_to_shorten_to=MAX_NUM_SENTS
        )
        maml.test(dataloader_test)

if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(True):
    main(args)
