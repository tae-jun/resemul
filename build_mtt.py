import os
import argparse
import tensorflow as tf
from multiprocessing import Queue, Process
from madmom.audio.signal import LoadAudioFileError
from data import load_annotations
from data import audio_to_examples, audio_to_sequence_example

parser = argparse.ArgumentParser(description='Build MagnaTagATune dataset.')
parser.add_argument('--data-dir', type=str, default='./dataset', metavar='PATH')
parser.add_argument('--sample-rate', type=int, default=22050, metavar='N', help='Sampling rate for audios.')
parser.add_argument('--num-samples', type=int, default=59049, metavar='N', help='Number of samples per segment.')
parser.add_argument('--num-audios-per-shard', type=int, default=100, metavar='N', help='Number of audios per shard.')

args = parser.parse_args()


def process_audio_files(queue):
  """Processes and saves audios as TFRecord files in one sub-process.

  Args:
    queue: A queue contains arguments which consist of:
      assigned_anno: A DataFrame which contains information about the audios
        that should be process in this sub-process.
      sample_rate: Sampling rate of the audios. If the sampling rate is different
        with an audio's original sampling rate, then it re-samples the audio.
      num_samples: Number of samples one segment contains.
      split: Dataset split which is one of 'train', 'val', or 'test'.
      shard: Shard index.
      num_total_shards: Number of the entire shards.
  """
  while not queue.empty():
    (assigned_anno, sample_rate, num_samples, split, shard, num_total_shards) = queue.get()

    is_test = (split == 'test')
    output_filename_format = ('{}-{:04d}-of-{:04d}.seq.tfrecord' if is_test else '{}-{:04d}-of-{:04d}.tfrecord')
    output_filename = output_filename_format.format(split, shard, num_total_shards)
    output_file_path = args.data_dir + '/tfrecord/' + output_filename

    writer = tf.python_io.TFRecordWriter(output_file_path)
    for _, row in assigned_anno.iterrows():
      audio_path = args.data_dir + '/mp3/' + row['mp3_path']
      labels = row[:50].tolist()

      try:
        if is_test:
          examples = [audio_to_sequence_example(audio_path, labels, sample_rate, num_samples)]
        else:
          examples = audio_to_examples(audio_path, labels, sample_rate, num_samples)
      except LoadAudioFileError:
        # There are some broken mp3 files. Skip it.
        print('Cannot load audio "{}". Skip it.'.format(audio_path))
        continue

      for example in examples:
        writer.write(example.SerializeToString())
    writer.close()

    print('{} audios are written into "{}".'.format(len(assigned_anno), output_filename))


def process_dataset(anno, sample_rate, num_samples):
  """Processes, and saves MagnaTagATune dataset using multi-processes.

  Args:
    anno: Annotation DataFrame contains tags, mp3_path, split, and shard.
    sample_rate: Sampling rate of the audios. If the sampling rate is different
      with an audio's original sampling rate, then it re-samples the audio.
    num_samples: Number of samples one segment contains.
    n_threads: Number of threads to process the dataset.
  """
  queue = Queue()
  split_and_shard_sets = anno[['split', 'shard']].drop_duplicates().values
  for split, shard in split_and_shard_sets:
    assigned_anno = anno[(anno['split'] == split) & (anno['shard'] == shard)]
    num_total_shards = anno[anno['split'] == split]['shard'].nunique()
    queue.put((assigned_anno, sample_rate, num_samples, split, shard, num_total_shards))

  procs = []
  for _ in range(os.cpu_count()):
    proc = Process(target=process_audio_files, args=(queue,))
    proc.start()
    procs.append(proc)

  for proc in procs:
    proc.join()


def main():
  # Create the output directory.
  os.makedirs(args.data_dir + '/tfrecord', exist_ok=True)
  df = load_annotations(args.data_dir + '/annotations_final.csv', args.num_audios_per_shard)

  print('Start building the dataset.')
  process_dataset(df, args.sample_rate, args.num_samples)

  print('Done.\n')


if __name__ == '__main__':
  main()
