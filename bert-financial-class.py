import datetime
import functools
import os
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow.compat.v1 as tf 
tpu = tf.distribute.cluster_resolver.TPUClusterResolver("au-tpu-0001")
TPU_ADDRESS = tpu.get_master()
tf.disable_v2_behavior()

from contextlib import contextmanager
import logging as py_logging

tf.get_logger().propagate = False
py_logging.root.setLevel('INFO')

def tf_verbosity_level(level):
	og_level = tf.logging.get_verbosity()
	tf.logging.set_verbosity(level)
	yield
	tf.logging.set_verbosity(og_level)

import sys

import modeling
import optimization
import run_classifier
import run_classifier_with_tfhub
import tokenization

import financial_proc as fproc

import tensorflow_hub as hub

import pandas as pd

dft = pd.read_csv('gs://rrg3rd-t5-financial-0/data/splits/financial-train.csv',sep=';')
dfv = pd.read_csv('gs://rrg3rd-t5-financial-0/data/splits/financial-val.csv',sep=';')
dfT = pd.read_csv('gs://rrg3rd-t5-financial-0/data/splits/financial-test.csv',sep=';')

dft['message'] = dft['Message'].str.lower()
dfv['message'] = dfv['Message'].str.lower()
dfT['message'] = dfT['Message'].str.lower()

dft["message"] = dft["message"].str.replace('\n', ' ').str.replace('\t', ' ')
dfv["message"] = dfv["message"].str.replace('\n', ' ').str.replace('\t', ' ')
dfT["message"] = dfT["message"].str.replace('\n', ' ').str.replace('\t', ' ')

dfv['nclass'] = dfv['NominalReturnSign']#/158.8235
dft['nclass'] = dft['NominalReturnSign']#/158.8235
dfT['nclass'] = dfT['NominalReturnSign']#/158.8235

dfv.nclass[dfv.nclass == 'Positive'] = 1
dfv.nclass[dfv.nclass == 'Negative'] = 0
dft.nclass[dft.nclass == 'Positive'] = 1
dft.nclass[dft.nclass == 'Negative'] = 0
dfT.nclass[dfT.nclass == 'Positive'] = 1
dfT.nclass[dfT.nclass == 'Negative'] = 0

dfv["input_text"] = dfv["message"]
dfv["target_text"] = dfv["nclass"]
dft["input_text"] = dft["message"]
dft["target_text"] = dft["nclass"]
dfT['input_text'] = dfT['message']
dfT['target_text'] = dfT['nclass']

train_df = dft[["input_text", "target_text"]]
val_df = dfv[["input_text", "target_text"]]
test_df = dfT[["input_text", "target_text"]]
train_df = train_df.astype(str)
val_df = val_df.astype(str)
test_df = test_df.astype(str)
train_df.to_csv("gs://au-tpu-0000/data/class1/train.tsv", index=False, header=False, sep="\t")
val_df.to_csv("gs://au-tpu-0000/data/class1/dev.tsv", index=False, header=False, sep = "\t")
test_df.to_csv("gs://au-tpu-0000/data/class1/test.tsv", index=False, header=False, sep= "\t")

TASK_DATA_DIR = "gs://au-tpu-0000/data/class1/"
TASK = "f_class_1"

BUCKET = 'au-tpu-0000' #@param {type:"string"}
assert BUCKET, 'Must specify an existing GCS bucket name'
OUTPUT_DIR = 'gs://{}/bert-tfhub/models/{}'.format(BUCKET, TASK)
tf.gfile.MakeDirs(OUTPUT_DIR)
print('***** Model output directory: {} *****'.format(OUTPUT_DIR))

# Available pretrained model checkpoints:
#   uncased_L-12_H-768_A-12: uncased BERT base model
#   uncased_L-24_H-1024_A-16: uncased BERT large model
#   cased_L-12_H-768_A-12: cased BERT large model
BERT_MODEL = 'uncased_L-12_H-768_A-12' #@param {type:"string"}
BERT_MODEL_HUB = 'https://tfhub.dev/google/bert_' + BERT_MODEL + '/1'

tokenizer = run_classifier_with_tfhub.create_tokenizer_from_hub_module(BERT_MODEL_HUB)

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
PREDICT_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 10.0
MAX_SEQ_LENGTH = 1024
# Warmup is a period of time where hte learning rate 
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 500

fprocessor = fproc.FinancialProc()
label_list = fprocessor.get_labels()

# Compute number of train and warmup steps from batch size
train_examples = fprocessor.get_train_examples(TASK_DATA_DIR)
num_train_steps = int(len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

# Setup TPU related config
tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)
NUM_TPU_CORES = 8
ITERATIONS_PER_LOOP = 1000

def get_run_config(output_dir):
  return tf.estimator.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=output_dir,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    tpu_config=tf.estimator.tpu.TPUConfig(
        iterations_per_loop=ITERATIONS_PER_LOOP,
        num_shards=NUM_TPU_CORES,
        per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2))

# Force TF Hub writes to the GS bucket we provide.
os.environ['TFHUB_CACHE_DIR'] = OUTPUT_DIR

model_fn = run_classifier_with_tfhub.model_fn_builder(
  num_labels=len(label_list),
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps,
  use_tpu=True,
  bert_hub_module_handle=BERT_MODEL_HUB
)

estimator_from_tfhub = tf.estimator.tpu.TPUEstimator(
  use_tpu=True,
  model_fn=model_fn,
  config=get_run_config(OUTPUT_DIR),
  train_batch_size=TRAIN_BATCH_SIZE,
  eval_batch_size=EVAL_BATCH_SIZE,
  predict_batch_size=PREDICT_BATCH_SIZE,
)

# Train the model
def model_train(estimator):
  print('MRPC/CoLA on BERT base model normally takes about 2-3 minutes. Please wait...')
  # We'll set sequences to be at most 128 tokens long.
  train_features = run_classifier.convert_examples_to_features(
      train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
  print('***** Started training at {} *****'.format(datetime.datetime.now()))
  print('  Num examples = {}'.format(len(train_examples)))
  print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
  tf.logging.info("  Num steps = %d", num_train_steps)
  train_input_fn = run_classifier.input_fn_builder(
      features=train_features,
      seq_length=MAX_SEQ_LENGTH,
      is_training=True,
      drop_remainder=True)
  estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  print('***** Finished training at {} *****'.format(datetime.datetime.now()))

model_train(estimator_from_tfhub)

def model_eval(estimator):
  # Eval the model.
  #eval_examples = fprocessor.get_dev_examples(TASK_DATA_DIR)
  eval_examples = fprocessor.get_test_examples(TASK_DATA_DIR)
  eval_features = run_classifier.convert_examples_to_features(
      eval_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
  print('***** Started evaluation at {} *****'.format(datetime.datetime.now()))
  print('  Num examples = {}'.format(len(eval_examples)))
  print('  Batch size = {}'.format(EVAL_BATCH_SIZE))

  # Eval will be slightly WRONG on the TPU because it will truncate
  # the last batch.
  eval_steps = int(len(eval_examples) / EVAL_BATCH_SIZE)
  eval_input_fn = run_classifier.input_fn_builder(
      features=eval_features,
      seq_length=MAX_SEQ_LENGTH,
      is_training=False,
      drop_remainder=True)
  result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
  print('***** Finished evaluation at {} *****'.format(datetime.datetime.now()))
  output_eval_file = os.path.join(OUTPUT_DIR, "eval_results.txt")
  with tf.gfile.GFile(output_eval_file, "w") as writer:
    print("***** Eval results *****")
    for key in sorted(result.keys()):
      print('  {} = {}'.format(key, str(result[key])))
      writer.write("%s = %s\n" % (key, str(result[key])))

model_eval(estimator_from_tfhub)

