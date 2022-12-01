from typing import List, Tuple
import argparse
import copy
import json
import os
import torch
from torch import nn

from functools import partial
torch.utils.checkpoint.checkpoint = functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)

import transformers
import numpy as np
# import tensorboard
# import tensorflow as tf

import utils
from dataloader import get_counselchat_meta_learning_dataloader as get_dataloader

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='med')
parser.add_argument('--mode', default='all')
parser.add_argument('--log_dir', type=str, default=None,
                    help='directory to save to or load from')
parser.add_argument('--num_support', type=int, default=2,
                    help='number of support (question, response) pairs in a task')
parser.add_argument('--num_query', type=int, default=1,
                    help='number of query (question, response) pairs in a task')
parser.add_argument('--num_inner_steps', type=int, default=1,
                    help='number of inner-loop updates')
parser.add_argument('--inner_lr', type=float, default=0.4,
                    help='inner-loop learning rate initialization')
parser.add_argument('--learn_inner_lrs', default=False, action='store_true',
                    help='whether to optimize inner-loop learning rates')
parser.add_argument('--outer_lr', type=float, default=0.001,
                    help='outer-loop learning rate')
parser.add_argument('--batch_size', type=int, default=1,
                    help='number of tasks per outer-loop update')
parser.add_argument('--num_train_iterations', type=int, default=1000,
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
MAX_NUM_SENTS = 2  # maximum number of sentences for each input question / response
MAX_TOKENS = 256  # maximum number of tokens to generate
NUM_TEST_TASKS = 128 // args.num_query  # TODO: update this

SAVE_INTERVAL = 100
LOG_INTERVAL = 5
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
        self._tokenizer = tokenizer
        self._mode = mode
        self._num_inner_steps = num_inner_steps
        self._inner_lrs = torch.tensor(inner_lr, requires_grad=learn_inner_lrs)
        self._outer_lr = outer_lr

        self.gpt2_model = transformers.AutoModelForCausalLM.from_pretrained(‘gpt2’)
        self.gpt2_model.transformer.gradient_checkpointing = True

        # TODO: train only part of GPT-2's parameters
        # TODO: initialize different modules of GPT2 for different lr
        self._optimizer = torch.optim.Adam(
            params=list(parameters_to_fine_tune(self._model, self._mode)) + [self._inner_lrs],
            lr=self._outer_lr
        )

        # optimize the initial params to be adapted
        self.meta_opt = torch.optim.Adam(list(parameters_to_fine_tune(self._model, self._mode)), self._outer_lr)


        self._log_dir = log_dir
        self._log_file = log_file

        self._start_train_step = 0


    def _inner_loop(self, tokenized_seqs, train):
        """Computes the adapted network parameters via the MAML inner loop.

        Args:
            tokenized_seqs (dict): output of utils.tokenize_gpt2_batch
            train (bool): whether we are training or evaluating

        Returns:
            parameters (dict[str, Tensor]): adapted network parameters
            scores (list[float]): support set score (BLEU or ROUGE) over the course of
                the inner loop, length num_inner_steps + 1
        """
        model_copy = copy.deepcopy(self._model)
        inner_optimizer = torch.optim.Adam(
            # params=parameters_to_fine_tune(self._model, self._mode),
            params=parameters_to_fine_tune(model_copy, self._mode),
            lr=self._inner_lrs
        )
        for _ in range(self._num_inner_steps):
            # loss = self._model(**tokenized_seqs).loss
            loss = model_copy(**tokenized_seqs).loss
            loss.backward(create_graph=train)
            inner_optimizer.step()
        inner_optimizer.zero_grad()  # issue: torch optimizer is not differentiable
        return model_copy


    def _outer_step(self, task_batch, train):
        """Computes the MAML loss and metrics on a batch of tasks.

        Args:
            task_batch (tuple): batch of tasks from a Counsel chat DataLoader
            train (bool): whether we are training or evaluating

        Returns:
            outer_loss (Tensor): mean MAML loss over the batch, scalar
            scores_support (ndarray): support set score over the
                course of the inner loop, averaged over the task batch
                shape (num_inner_steps + 1,)
            score_query (float): query set score of the adapted
                parameters, averaged over the task batch
        """
        output_record = {}
        outer_loss_batch = []
        score_query_batch = []
        for task in task_batch:
            inp_support, out_support, inp_query, out_query = task

            tokenized_seqs = utils.tokenize_gpt2_batch(self._tokenizer, inp_support, out_support, DEVICE)
            # model_copy = copy.deepcopy(self._model.state_dict())
            model_copy = self._inner_loop(tokenized_seqs, train)
            # self._inner_loop(tokenized_seqs, train)

            tokenized_query_seqs = utils.tokenize_gpt2_batch(self._tokenizer, inp_query, out_query, DEVICE)
            # loss = self._model(**tokenized_query_seqs).loss
            loss = model_copy(**tokenized_query_seqs).loss
            outer_loss_batch.append(loss)

            if not train:  # do the evaluation only when not training
                # decoded_out = utils.model_generate(self._tokenizer, self._model, inp_query, DEVICE, MAX_TOKENS)
                decoded_out = utils.model_generate(self._tokenizer, model_copy, inp_query, DEVICE, MAX_TOKENS)
                if POSTPROCESS:
                    decoded_out = utils.batch_postprocess_generations(decoded_out)
                print("Input:", inp_query)
                print("Output:", decoded_out)
                score = utils.get_bleu(decoded_out, out_query)
                score_query_batch.append(score)
                for i in range(len(inp_query)):
                    output_record[inp_query[i]] = {'PREDICTION': decoded_out[i], 'TARGET': out_query[i]}
            
            # self._model.load_state_dict(model_copy)

        outer_loss = torch.mean(torch.stack(outer_loss_batch))
        score_query = 0 if train else np.mean(score_query_batch)
        return outer_loss, score_query, output_record

    # def train(self, dataloader_train, dataloader_val):
    #     """Train the MAML.

    #     Consumes dataloader_train to optimize MAML meta-parameters
    #     while periodically validating on dataloader_val, logging metrics, and
    #     saving checkpoints.

    #     Args:
    #         dataloader_train (DataLoader): loader for train tasks
    #         dataloader_val (DataLoader): loader for validation tasks
    #         writer (SummaryWriter): TensorBoard logger
    #     """
    #     print(f'Starting training at iteration {self._start_train_step}.')
    #     for i_step, task_batch in enumerate(
    #             dataloader_train,
    #             start=self._start_train_step
    #     ):
    #         self._optimizer.zero_grad()
    #         outer_loss, _, _ = self._outer_step(task_batch, train=True)
    #         outer_loss.backward()
    #         self._optimizer.step()

    #         if i_step % LOG_INTERVAL == 0:
    #             log = f'Iteration {i_step}: loss: {outer_loss.item():.3f}'
    #             print(log)
    #             with open(self._log_file, 'a') as f:
    #                 f.write(log + '\n')

    #         if i_step % VAL_INTERVAL == 0:
    #             losses = []
    #             query_scores = []
    #             for val_task_batch in dataloader_val:
    #                 outer_loss, query_score, _ = self._outer_step(val_task_batch, train=False)
    #                 losses.append(outer_loss.item())
    #                 query_scores.append(query_score)
    #             loss = np.mean(losses)
    #             score = np.mean(query_scores)
    #             log = f'Validation: loss: {loss:.3f}, score: {score:.3f}'
    #             print(log)
    #             with open(self._log_file, 'a') as f:
    #                 f.write(log + '\n')

    #         if i_step % SAVE_INTERVAL == 0:
    #             self._save(i_step)

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

            meta_opt = self.meta_opt
            inner_opt = torch.optim.Adam(params=parameters_to_fine_tune(self._model, self._mode),lr=self._inner_lrs)
            n_inner_iter = self._num_inner_steps

            meta_opt.zero_grad()

            qry_losses = []
            qry_accs = []

            for task in task_batch:
                inp_support, out_support, inp_query, out_query = task
                tokenized_seqs = utils.tokenize_gpt2_batch(self._tokenizer, inp_support, out_support, DEVICE)
                tokenized_seqs_query = utils.tokenize_gpt2_batch(self._tokenizer, inp_query, out_query, DEVICE)

                with higher.innerloop_ctx(self._model, inner_opt, copy_initial_weights=False) as (maml_model, diffopt):
                    for _ in range(n_inner_iter):

                        spt_loss = maml_model(**tokenized_seqs).loss
                        diffopt.step(spt_loss)

                    qry_loss = maml_model(**tokenized_seqs_query).loss
                    qry_losses.append(qry_loss.detach())
                    # Update the model's meta-parameters to optimize the query losses across all of the tasks sampled in this batch.
                    # This unrolls through the gradient steps.
                    qry_loss.backward()
            
            meta_opt.step()
            outer_loss = np.avg(qry_losses)
            score_query = 0



            if i_step % LOG_INTERVAL == 0:
                # log = f'Iteration {i_step}: loss: {outer_loss.item():.3f}'
                log = f'Iteration {i_step}: loss: {outer_loss:.3f}'
                print(log)
                with open(self._log_file, 'a') as f:
                    f.write(log + '\n')

            if i_step % VAL_INTERVAL == 0:
                losses = []
                query_scores = []
                for val_task_batch in dataloader_val:
                    val_inp_support, val_out_support, val_inp_query, val_out_query = val_task_batch
                    val_tokenized_seqs = utils.tokenize_gpt2_batch(self._tokenizer, val_inp_support, val_out_support, DEVICE)
                    val_tokenized_seqs_query = utils.tokenize_gpt2_batch(self._tokenizer, val_inp_query, val_out_query, DEVICE)

                    with higher.innerloop_ctx(self._model, inner_opt, copy_initial_weights=False) as (maml_model, diffopt):
                        decoded_out = utils.model_generate(self._tokenizer, maml_model, val_inp_query, DEVICE, MAX_TOKENS)
                        losses.append(maml_model(**val_tokenized_seqs_query).loss)

                    if POSTPROCESS:
                        decoded_out = utils.batch_postprocess_generations(decoded_out)

                    print("Input:", val_inp_query)
                    print("Output:", decoded_out)
                    score = utils.get_bleu(decoded_out, val_out_query)
                    query_scores.append(score)


                    for i in range(len(val_inp_query)):
                        output_record[val_inp_query[i]] = {'PREDICTION': decoded_out[i], 'TARGET': val_out_query[i]}

                score_query = np.mean(query_scores)
                outer_loss = np.mean(losses)
                    
                loss = outer_loss
                score = score_query
                log = f'Validation: loss: {loss:.3f}, score: {score:.3f}'
                print(log)
                with open(self._log_file, 'a') as f:
                    f.write(log + '\n')

            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)


    def test(self, dataloader_test):
        """Evaluate the MAML on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        scores = []
        output_records = {}
        for task_batch in dataloader_test:

            tst_inp_support, tst_out_support, tst_inp_query, tst_out_query = dataloader_test
            tst_tokenized_seqs = utils.tokenize_gpt2_batch(self._tokenizer, tst_inp_support, tst_out_support, DEVICE)
            tst_tokenized_seqs_query = utils.tokenize_gpt2_batch(self._tokenizer, tst_inp_query, tst_out_query, DEVICE)

            with higher.innerloop_ctx(self._model, inner_opt, copy_initial_weights=False) as (maml_model, diffopt):
                decoded_out = utils.model_generate(self._tokenizer, maml_model, tst_inp_query, DEVICE, MAX_TOKENS)
                losses.append(maml_model(**tst_tokenized_seqs_query).loss)

            if POSTPROCESS:
                decoded_out = utils.batch_postprocess_generations(decoded_out)

            print("Input:", val_inp_query)
            print("Output:", decoded_out)
            score = utils.get_bleu(decoded_out, val_out_query)
            scores.append(score)


            for i in range(len(val_inp_query)):
                output_record[val_inp_query[i]] = {'PREDICTION': decoded_out[i], 'TARGET': val_out_query[i]}

    
            for k, v in output_record.items():
                output_records[k] = v
        with open(os.path.join(self._log_dir, f'support:{args.num_support}.json'), 'w') as f:
            json.dump(output_records, f, indent=4)

        mean = np.mean(scores)
        std = np.std(scores)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Score over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )


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
            self._inner_lrs = state['inner_lrs']
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
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
        optimizer_state_dict = self._optimizer.state_dict()
        torch.save(
            dict(model_state_dict=self._model.state_dict(),
                 inner_lrs=self._inner_lrs,
                 optimizer_state_dict=optimizer_state_dict),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')

def main(args):
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./results/maml/support={args.num_support}'  # pylint: disable=line-too-long
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'support:{args.num_support}.query:{args.num_query}.inner_steps:{args.num_inner_steps}.inner_lr:{args.inner_lr}.learn_inner_lrs:{args.learn_inner_lrs}.outer_lr:{args.outer_lr}.txt')
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
