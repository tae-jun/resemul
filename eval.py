import argparse
import keras

from keras.layers import Input, Lambda
from keras.models import load_model
from sklearn.metrics import roc_auc_score
from data.batch import batch_inputs
from utils import *

keras.layers.TFRecordModel = TFRecordModel


def main():
  parser = argparse.ArgumentParser(description='Evaluate a model on test set.')
  parser.add_argument('path', type=str, metavar='PATH', help='Path to a saved model.')
  parser.add_argument('--data-dir', type=str, default='./dataset', metavar='PATH')

  args = parser.parse_args()

  evaluate(args.path, args.data_dir, 4332, 100, verbose=1)
  print('\n=> Done.\n')


def evaluate(model_or_path, data_dir, num_examples, num_audios_per_shard, verbose=0):
  if type(model_or_path) == str:
    model = load_model(model_or_path)
  else:
    model = model_or_path

  # Prepare inputs.
  segments, labels = batch_inputs(
    file_pattern=make_path(data_dir, 'tfrecord', 'test-????-of-????.seq.tfrecord'),
    batch_size=1, is_training=False, is_sequence=True, examples_per_shard=num_audios_per_shard,
    num_read_threads=1, shard_queue_name='filename_queue', example_queue_name='input_queue')

  segments = Input(tensor=tf.squeeze(segments))
  labels = Input(tensor=tf.squeeze(labels))

  pred = model(segments)
  avg_pred = Lambda(lambda x: tf.reduce_mean(x, axis=0))(pred)
  avg_model = TFRecordModel(inputs=[segments, labels], outputs=[avg_pred, labels])

  print('=> Start evaluation.')
  preds, trues = [], []
  for i in range(num_examples):
    pred, true = avg_model.predict_tfrecord(segments)
    preds.append(pred)
    trues.append(true)
    if verbose > 0 and i % (num_examples // 100) == 0 and i:
      print('Evaluated [{:04d}/{:04d}].'.format(i + 1, num_examples))

  y_true, y_pred = np.stack(trues), np.stack(preds)
  roc_auc = roc_auc_score(y_true, y_pred, average='macro')
  print('=> @ ROC AUC score: {}'.format(roc_auc))


if __name__ == '__main__':
  main()
