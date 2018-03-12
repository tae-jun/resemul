import tensorflow as tf
import numpy as np

from madmom.audio.signal import Signal


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _segments_to_sequence_example(segments, labels):
  """Converts a list of segments to a SequenceExample proto.
  
  Args: 
    segments: A list of segments.
    labels: A list of labels of the segments.
  
  Returns:
    A SequenceExample proto.
  """
  raw_segments = [segment.tostring() for segment in segments]
  raw_labels = np.array(labels, dtype=np.uint8).tostring()

  context = tf.train.Features(feature={
    'raw_labels': _bytes_feature(raw_labels)  # uint8 Tensor (50,)
  })

  feature_lists = tf.train.FeatureLists(feature_list={
    'raw_segments': _bytes_feature_list(raw_segments)  # list of float32 Tensor (59049,)
  })

  sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)

  return sequence_example


def _segment_to_example(segment, labels):
  """Converts a list of segments to a list of Example protos.
  
  Args: 
    segments: A list of segments.
    labels: A list of labels of the segments.
  
  Returns:
    A list of Example protos.
  """
  # dtype of segment is float32
  raw_segment = segment.tostring()
  raw_labels = np.array(labels, dtype=np.uint8).tostring()

  example = tf.train.Example(features=tf.train.Features(feature={
    'raw_labels': _bytes_feature(raw_labels),  # uint8 Tensor (50,)
    'raw_segment': _bytes_feature(raw_segment)  # float32 Tensor (59049,)
  }))

  return example


def _audio_to_segments(filename, sample_rate, num_samples):
  """Loads, and splits an audio into N segments.
  
  Args:
    filename: A path to the audio.
    sample_rate: Sampling rate of the audios. If the sampling rate is different 
      with an audio's original sampling rate, then it re-samples the audio.
    num_samples: Number of samples one segment contains.
    
  Returns:
    A list of numpy arrays; segments.
  """
  # Load an audio file as a numpy array
  sig = Signal(filename, sample_rate=sample_rate, dtype=np.float32, stop=29, num_channels=1)

  # Split the signal into segments
  total_samples = sig.shape[0]
  n_segment = total_samples // num_samples
  segments = [sig[i * num_samples:(i + 1) * num_samples] for i in range(n_segment)]

  return segments


def audio_to_sequence_example(filename, labels, sample_rate, num_samples):
  """Converts an audio to a SequenceExample proto.
  
  Args:
    filename: A path to the audio.
    labels: A list of labels of the audio.
    sample_rate: Sampling rate of the audios. If the sampling rate is different 
      with an audio's original sampling rate, then it re-samples the audio.
    num_samples: Number of samples one segment contains.
  
  Returns:
    A SequenceExample proto.
  """
  segments = _audio_to_segments(filename, sample_rate=sample_rate, num_samples=num_samples)
  sequence_example = _segments_to_sequence_example(segments, labels)
  return sequence_example


def audio_to_examples(filename, labels, sample_rate, num_samples):
  """Converts an audio to a list of Example protos.
  
  Args:
    filename: A path to the audio.
    labels: A list of labels of the audio.
    sample_rate: Sampling rate of the audios. If the sampling rate is different 
      with an audio's original sampling rate, then it re-samples the audio.
    num_samples: Number of samples one segment contains.
  
  Returns:
    A list of Example protos.
  """
  segments = _audio_to_segments(filename, sample_rate=sample_rate, num_samples=num_samples)
  examples = [_segment_to_example(segment, labels) for segment in segments]
  return examples
