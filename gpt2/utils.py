import torch
import torch.nn as nn
import sacrebleu
from rouge_score import rouge_scorer
import transformers

def model2hfname(model: str) -> str:
    return {
        'small': 'gpt2',
        'med': 'gpt2-medium',
        'large': 'gpt2-large',
        'full': 'gpt2-xl',
        'gpt2-sm': 'gpt2',
        'gpt2-med': 'gpt2-medium',
        'gpt2-lg': 'gpt2-large',
        'gpt2': 'gpt2-xl',
        'neo': 'EleutherAI/gpt-neo-2.7B',
    }[model]

def get_model_and_tokenizer(model: str, Cls, **model_kwargs):
    hf_model_name = model2hfname(model)

    m = Cls.from_pretrained(hf_model_name, **model_kwargs)
    if isinstance(m, transformers.GPT2LMHeadModel):
        m.transformer.gradient_checkpointing_enable()

    tok = transformers.AutoTokenizer.from_pretrained(hf_model_name)

    if tok.pad_token_id is None:
        if Cls == transformers.AutoModelForCausalLM:
            tok.pad_token = tok.eos_token
        else:
            print("Adding pad token to tokenizer")
            tok.add_special_tokens({'pad_token': '[PAD]'})
            tok.pad_token = '[PAD]'
    return m, tok


def tokenize_gpt2_batch(tokenizer, x, y, DEVICE):
    """
    Implement the tokenization step for a batch of examples for GPT-2.

    Args:
        tokenizer: a GPT2Tokenizer that you can call and receive a dictionary of:
          - input_ids: a list (or tensor) of token ids
          - attention_mask: a list (or tensor) of 1s and 0s indicating which tokens
              are padding (if you requested padding and tensors from the tokenizer)
        x: a list of strings, each of which is the input for a single example
        y: a list of strings, each of which is a *target* for a single example
    
    Returns:
        A dictionary with the following keys:
            - input_ids: a tensor of shape [batch_size, sequence_length] 
                containing the token ids
            - attention_mask: a tensor of shape [batch_size, sequence_length] 
                containing 1s and 0s indicating which tokens are padding
            - labels: a tensor of shape [batch_size, sequence_length] containing
                the target token ids, with -100 for non-target tokens (i.e., the
                tokens in the input part of each example or padding tokens)
        where sequence_length is determined by the (x, y) pair whose tokenized
        length is the longest in the batch. The other sequences should be padded to
        this length (you can get the tokenizer to handle this padding!).
    """
    tokenized_sequences = tokenizer([x_ + y_ for x_, y_ in zip(x, y)], return_tensors='pt', padding=True)
    labels = tokenized_sequences['input_ids'].clone()
    x_input_ids = tokenizer(x)['input_ids']
    for i, sent_ids in enumerate(x_input_ids):
        labels[i][:len(sent_ids)] = -100  # mask the tokens in x
    labels[~tokenized_sequences['attention_mask'].to(torch.bool)] = -100  # mask the paddings
    tokenized_sequences['labels'] = labels
    return tokenized_sequences.to(DEVICE)


def get_loss(logits: torch.tensor, targets: torch.tensor) -> torch.tensor:
    """
    Computes the cross-entropy loss for text generation.

    For generation, you'll need to deal with the fact that different sequences witihn
      the batch are different lengths, and the targets tensor includes some mask
      values (-100). The average loss is the *average loss over all non-masked timesteps*.
      You'll also need to handle the fact that the prediction for what token t will be is
      made after seeing only t - 1 tokens; that is, there is an off-by-one shift needed
      between the logits and targets.

    Args:
      logits: a [batch_size, sequence_length, vocab_size] tensor of *UNNORMALIZED* logits
      targets: a 2D [batch_size, sequence_length] tensor of target indices. May contain
        -100 in some positions, meaning that the loss for this timestep should be ignored.
    
    Returns:
      A zero-dim tensor representing the average cross-entropy loss over all batch 
        elements (and sequence timesteps, if applicable)
    """
    batch_size = logits.size(0)
    seq_len = logits.size(1)
    combined_sz = batch_size * (seq_len-1)
    logits = logits[:,:-1,:].view(combined_sz, -1)
    targets = targets[:,1:].view(combined_sz)  # target tokens are one to the right of the logits
    non_mask_idxs = (targets != -100)
    return nn.functional.cross_entropy(logits[non_mask_idxs], targets[non_mask_idxs])


def get_bleu(targets, predictions):
  """Computes BLEU score.

  Args:
    targets: list of strings or list of list of strings if multiple references
      are present.
    predictions: list of strings

  Returns:
    bleu_score across all targets and predictions
  """
  if isinstance(targets[0], list):
    targets = [[x for x in target] for target in targets]
  else:
    # Need to wrap targets in another list for corpus_bleu.
    targets = [targets]

  bleu_score = sacrebleu.corpus_bleu(predictions, targets,
                                     smooth_method="exp",
                                     smooth_value=0.0,
                                     force=False,
                                     lowercase=False,
                                     tokenize="intl",
                                     use_effective_order=False)
  return bleu_score.score


def get_rouge(targets, predictions):
    """Computes Rouge score."""
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = []
    for p, t in zip(predictions, targets):
        score = scorer.score(p, t)['rouge1'].fmeasure
        scores.append(score)
    return sum(scores) / len(scores)
