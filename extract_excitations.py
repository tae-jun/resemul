import argparse
import keras
import pandas as pd

from keras.layers import Input
from keras.models import load_model
from data.batch import batch_inputs
from data import load_annotations
from utils import *

keras.layers.TFRecordModel = TFRecordModel


def main():
  parser = argparse.ArgumentParser(description='Extract excitations of SE blocks.')
  parser.add_argument('path', type=str, metavar='PATH', help='Path to a saved model.')
  parser.add_argument('--data-dir', type=str, default='./dataset', metavar='PATH')
  parser.add_argument('--out', type=str, default='./excitations.pkl', metavar='PATH')

  args = parser.parse_args()

  extract_excitations(args.path, args.data_dir, 4332, 100, args.out)

  print('\n=> The excitations are saved to ' + args.out)
  print('=> Done.\n')


def extract_excitations(model_or_path, data_dir, num_examples, num_audios_per_shard, out_path):
  if type(model_or_path) == str:
    model = load_model(model_or_path)
  else:
    model = model_or_path

  # Prepare inputs.
  segments, label = batch_inputs(
    file_pattern=make_path(data_dir, 'tfrecord', 'test-????-of-????.seq.tfrecord'),
    batch_size=1, is_training=False, is_sequence=True, examples_per_shard=num_audios_per_shard,
    num_read_threads=1, shard_queue_name='filename_queue', example_queue_name='input_queue')

  segments = Input(tensor=tf.squeeze(segments))
  label = Input(tensor=tf.squeeze(label))

  # Create a model to extract excitations.
  excitations = [model.get_layer('dense_' + str(i)).output for i in range(2, 20, 2)]
  model_ex = TFRecordModel(inputs=model.inputs, outputs=excitations)
  model_ex = TFRecordModel(inputs=[segments, label], outputs=model_ex(segments) + [label])

  # Extract excitations.
  outputs = [model_ex.predict_tfrecord(segments) for _ in range(num_examples)]
  exs, labels = [output[:-1] for output in outputs], [output[-1] for output in outputs]

  # Average excitations for each song.
  exs = [[ex_depth.squeeze().mean(axis=0) for ex_depth in ex] for ex in exs]
  labels = np.stack(labels)

  # Collect data to create a DataFrame of excitations.
  rows = []
  for ex, label in zip(exs, labels):
    for depth, ex_depth in enumerate(ex):
      row = [ex_depth, depth] + label.tolist()
      rows.append(row)

  # Create the DataFrame and save them as a pickle file.
  tag_names = load_annotations(data_dir + '/annotations_final.csv', num_audios_per_shard).columns.tolist()[:50]
  df = pd.DataFrame(data=rows, columns=['ex', 'depth'] + tag_names)
  df.to_pickle(out_path)


if __name__ == '__main__':
  main()
