import tensorflow as tf


def batch_inputs(file_pattern, batch_size, is_training, examples_per_shard,
                 is_sequence=False, input_queue_capacity_factor=16, num_read_threads=4,
                 shard_queue_name='filename_queue', example_queue_name='input_queue'):
  data_files = []
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))

  assert len(data_files) > 0, 'Found no input files matching {}'.format(file_pattern)

  tf.logging.info('Prefetching values from %d files matching %s', len(data_files), file_pattern)

  if is_sequence:
    read_example_fn = _read_sequence_example
  else:
    read_example_fn = _read_example

  if is_training:
    filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=16, name=shard_queue_name)

    # examples_per_shard
    #   = examples_per_song * n_training_songs / n_training_shards
    #   = 10 * 15250 / 152
    #   = 1003
    #
    # example_size = 59049 * 4bytes = 232KB
    #
    # queue_size
    #   = examples_per_shard * input_queue_capacity_factor * example_size
    #   = 1003 * 16 * 232KB = 3.7GB
    min_queue_examples = examples_per_shard * input_queue_capacity_factor
    capacity = min_queue_examples + 3 * batch_size

    example_list = [read_example_fn(filename_queue) for _ in range(num_read_threads)]
    segment, label = tf.train.shuffle_batch_join(example_list, batch_size, capacity, min_queue_examples,
                                                 name='shuffle_' + example_queue_name, )

    return segment, label

  else:
    filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=1, name=shard_queue_name)
    segment, label = read_example_fn(filename_queue)

    capacity = examples_per_shard + 2 * batch_size
    segment_batch, label_batch = tf.train.batch([segment, label], batch_size, 1, capacity,
                                                name='fifo_' + example_queue_name)

    return segment_batch, label_batch


def _read_example(filename_queue, n_labels=50, n_samples=59049):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(serialized_example, features={
    'raw_labels': tf.FixedLenFeature([], tf.string),
    'raw_segment': tf.FixedLenFeature([], tf.string)
  })

  segment = tf.decode_raw(features['raw_segment'], tf.float32)
  segment.set_shape([n_samples])

  labels = tf.decode_raw(features['raw_labels'], tf.uint8)
  labels.set_shape([n_labels])
  labels = tf.cast(labels, tf.float32)

  return segment, labels


def _read_sequence_example(filename_queue,
                           n_labels=50, n_samples=59049, n_segments=10):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  context, sequence = tf.parse_single_sequence_example(
    serialized_example,
    context_features={
      'raw_labels': tf.FixedLenFeature([], dtype=tf.string)
    },
    sequence_features={
      'raw_segments': tf.FixedLenSequenceFeature([], dtype=tf.string)
    })

  segments = tf.decode_raw(sequence['raw_segments'], tf.float32)
  segments.set_shape([n_segments, n_samples])

  labels = tf.decode_raw(context['raw_labels'], tf.uint8)
  labels.set_shape([n_labels])
  labels = tf.cast(labels, tf.float32)

  return segments, labels
