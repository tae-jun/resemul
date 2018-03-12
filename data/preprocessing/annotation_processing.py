import pandas as pd
import numpy as np


def load_annotations(filename, num_audios_per_shard=100, num_top=50):
  """Reads annotation file, takes top N tags, and splits data samples.

  Results 54 (top50_tags + [clip_id, mp3_path, split, shard]) columns:

    ['choral', 'female voice', 'metal', 'country', 'weird', 'no voice',
     'cello', 'harp', 'beats', 'female vocal', 'male voice', 'dance',
     'new age', 'voice', 'choir', 'classic', 'man', 'solo', 'sitar', 'soft',
     'pop', 'no vocal', 'male vocal', 'woman', 'flute', 'quiet', 'loud',
     'harpsichord', 'no vocals', 'vocals', 'singing', 'male', 'opera',
     'indian', 'female', 'synth', 'vocal', 'violin', 'beat', 'ambient',
     'piano', 'fast', 'rock', 'electronic', 'drums', 'strings', 'techno',
     'slow', 'classical', 'guitar', 'clip_id', 'mp3_path', 'split', 'shard']
     
  NOTE: This will exclude audios which have only zero-tags. Therefore, number of
    each split will be 15250 / 1529 / 4332 (training / validation / test).

  Args:
    filename: A path to annotation CSV file.
    num_top: Number of the most popular tags to take.
    num_audios_per_shard: Number of audios per shard.

  Returns:
    A DataFrame contains information of audios.

    Schema:
      <tags>: 0 or 1
      clip_id: clip_id of the original dataset
      mp3_path: A path to a mp3 audio file.
      split: A split of dataset (training / validation / test).
             The split is determined by its directory (0, 1, ... , f).
             First 12 directories (0 ~ b) are used for training,
             1 (c) for validation, and 3 (d ~ f) for test.
      shard: A shard index of the audio.
  """
  np.random.seed(0)

  df = pd.read_csv(filename, delimiter='\t')

  # Calculate TOP 50 tags.
  top50 = (df.drop(['clip_id', 'mp3_path'], axis=1)
    .sum()
    .sort_values()
    .tail(num_top)
    .index
    .tolist())

  # Select TOP 50 columns.
  df = df[top50 + ['clip_id', 'mp3_path']]

  # Select rows which has at least one label.
  df = df.loc[df.iloc[:, :num_top].any(axis=1)]

  def split_by_directory(mp3_path):
    directory = mp3_path.split('/')[0]
    part = int(directory, 16)

    if part in range(12):
      return 'train'
    elif part is 12:
      return 'val'
    elif part in range(13, 16):
      return 'test'

  # Split by directories.
  df['split'] = df['mp3_path'].apply(
    lambda mp3_path: split_by_directory(mp3_path))

  for split in ['train', 'val', 'test']:
    num_audios = sum(df['split'] == split)
    num_shards = num_audios // num_audios_per_shard
    num_remainders = num_audios % num_audios_per_shard

    shards = np.tile(np.arange(num_shards), num_audios_per_shard)
    shards = np.concatenate([shards, np.arange(num_remainders) % num_shards])
    shards = np.random.permutation(shards)

    df.loc[df['split'] == split, 'shard'] = shards

  df['shard'] = df['shard'].astype(int)

  return df
