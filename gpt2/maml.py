import argparse
import copy
import os
import torch
import transformers
import numpy as np
import tensorboard
import tensorflow as tf

import utils
from dataloader import get_counselchat_meta_learning_dataloader as get_dataloader

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='med')
parser.add_argument('--log_dir', type=str, default=None,
                    help='directory to save to or load from')
parser.add_argument('--num_support', type=int, default=1,
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
MAX_TOKENS = 512  # maximum number of tokens to generate
NUM_TEST_TASKS = 0  # TODO: update this

SAVE_INTERVAL = 100
LOG_INTERVAL = 2
VAL_INTERVAL = LOG_INTERVAL * 10

class Gpt2MAML:
    """Trains and assesses a MAML."""

    def __init__(
            self,
            model_name,
            num_inner_steps,
            inner_lr,
            learn_inner_lrs,
            outer_lr,
            log_dir
    ):
        """
        Inits MAML.
        The network is initialized with the parameters from GPT2.
        """
        model, tokenizer = utils.get_model_and_tokenizer(model_name, transformers.AutoModelForCausalLM)
        self._model = model.to(DEVICE)
        self._tokenizer = tokenizer
        self._num_inner_steps = num_inner_steps
        self._inner_lrs = inner_lr
        # TODO: train only part of GPT-2's parameters
        # TODO: initialize different modules of GPT2 for different lr
        self._learn_inner_lrs = learn_inner_lrs
        self._outer_lr = outer_lr

        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self._outer_lr
        )

        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

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
        model = copy.deepcopy(self._model)  # does gradient flow through deepcopy?
        for _ in range(self._num_inner_steps):
            loss = model(**tokenized_seqs).loss
            d_loss = torch.autograd.grad(outputs=loss, inputs=model.parameters(), create_graph=train)

            for i, param in enumerate(model.parameters()):
                param = param - self._inner_lrs * d_loss[i]
        return model
    
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
        outer_loss_batch = []
        score_query_batch = []
        for task in task_batch:
            inp_support, out_support, inp_query, out_query = task
            inp_support = list(inp_support)
            out_support = list(out_support)
            inp_query = list(inp_query)
            out_query = list(out_query)

            tokenized_seqs = utils.tokenize_gpt2_batch(self._tokenizer, inp_support, out_support, DEVICE)
            model = self._inner_loop(tokenized_seqs, train)
            tokenized_query_seqs = utils.tokenize_gpt2_batch(self._tokenizer, inp_query, out_query, DEVICE)
            loss = model(**tokenized_query_seqs).loss
            outer_loss_batch.append(loss)

            if not train:  # do the evaluation only when not training
                encodings = self._tokenizer(inp_query, return_tensors='pt')
                input_ids = encodings['input_ids'].to(DEVICE)
                attn_mask = encodings['attention_mask'].to(DEVICE)
                sampled_tokens = model.generate(input_ids, attention_mask=attn_mask, max_length=MAX_TOKENS)
                decoded_out = []
                for i in range(sampled_tokens.size(0)):
                    decoded = self._tokenizer.decode(sampled_tokens[i]).split('Response: ')[-1].strip()
                    decoded_out.append(decoded)
                score = utils.get_bleu(decoded_out, out_query)
                score_query_batch.append(score)

        outer_loss = torch.mean(torch.stack(outer_loss_batch))
        score_query = 0 if train else np.mean(score_query_batch)
        return outer_loss, score_query

    def train(self, dataloader_train, dataloader_val, writer):
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
            self._optimizer.zero_grad()
            outer_loss, _ = self._outer_step(task_batch, train=True)
            outer_loss.backward()
            self._optimizer.step()

            if i_step % LOG_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {outer_loss.item():.3f}, '
                )
                with writer.as_default():
                    tf.summary.scalar('loss/train', outer_loss.item(), i_step)
                    writer.flush()

            if i_step % VAL_INTERVAL == 0:
                losses = []
                query_scores = []
                for val_task_batch in dataloader_val:
                    outer_loss, query_score = self._outer_step(val_task_batch, train=False)
                    losses.append(outer_loss.item())
                    query_scores.append(query_score)
                loss = np.mean(losses)
                score = np.mean(query_scores)
                print(
                    f'Validation: '
                    f'loss: {loss:.3f}, '
                    f'score: {score:.3f}'
                )
                with writer.as_default():
                    tf.summary.scalar('loss/val', loss, i_step)
                    tf.summary.scalar('score/val', score, i_step)
                    writer.flush()
            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)

    def test(self, dataloader_test):
        """Evaluate the MAML on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        scores = []
        for task_batch in dataloader_test:
            _, score_query = self._outer_step(task_batch, train=False)
            scores.append(score_query)
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
        log_dir = f'./logs/maml/counselchat.support:{args.num_support}.query:{args.num_query}.inner_steps:{args.num_inner_steps}.inner_lr:{args.inner_lr}.learn_inner_lrs:{args.learn_inner_lrs}.outer_lr:{args.outer_lr}.batch_size:{args.batch_size}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    # writer = tensorboard.SummaryWriter(log_dir=log_dir)
    writer = tf.summary.create_file_writer(log_dir)

    maml = Gpt2MAML(
        args.model,
        args.num_inner_steps,
        args.inner_lr,
        args.learn_inner_lrs,
        args.outer_lr,
        log_dir
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
            num_training_tasks
        )
        dataloader_val = get_dataloader(
            'val',
            args.batch_size,
            args.num_support,
            args.num_query,
            args.batch_size * 4
        )
        maml.train(dataloader_train, dataloader_val, writer)
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
            NUM_TEST_TASKS
        )
        maml.test(dataloader_test)

if __name__ == '__main__':
    main(args)
