import seqio
import sacrebleu
import tensorflow as tf
import functools
# import tensorflow_datasets as tfds


def data_preprocessor(ds):
    def normalize_text(text):
        """Lowercase and remove quotes from a TensorFlow string."""
        text = tf.strings.lower(text)
        text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
        return text

    def to_inputs_and_targets(ex):
        """Map "... \t ..."->{"inputs": ..., "targets": ...}."""
        ex = tf.io.decode_csv(ex, record_defaults=["", ""],
                          field_delim="\t", use_quote_delim=False)
        x = {
            "inputs":
                tf.strings.join(["counselling: ", normalize_text(ex[0])]),
            "targets": normalize_text(ex[1])
        }
        return x

    return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def bleu(targets, predictions):
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
  return {"bleu": bleu_score.score}


def register_counsel_bot_task():
    TSV_PATH = {
    'train': '../finetune_t5/counselchat_train.tsv',
    'validation': '../finetune_t5/counselchat_test.tsv'}

    vocabulary = seqio.SentencePieceVocabulary(
        'gs://t5-data/vocabs/cc_all.32000/sentencepiece.model', extra_ids=100)
    output_features = {
        'inputs': seqio.Feature(vocabulary=vocabulary),
        'targets': seqio.Feature(vocabulary=vocabulary)
    }

    seqio.TaskRegistry.add(
        "counsel_bot",
        seqio.TextLineDataSource(TSV_PATH),
        preprocessors=[data_preprocessor, seqio.preprocessors.tokenize],
        output_features=output_features,
        metric_fns=[bleu]
    )


# register_counsel_bot_task()

# dataset = seqio.get_mixture_or_task("counsel_bot").get_dataset(
#     sequence_length={"inputs": 256, "targets": 256},
#     split="train",
#     shuffle=True,
#     num_epochs=1,
#     shard_info=seqio.ShardInfo(index=0, num_shards=10),
#     use_cached=False,
#     seed=42
# )

# # Print the first 5 examples.
# for _, ex in zip(range(5), dataset.as_numpy_iterator()):
#   print(ex)