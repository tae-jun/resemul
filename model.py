from keras.layers import (Conv1D, MaxPool1D, BatchNormalization, GlobalAvgPool1D, Multiply, GlobalMaxPool1D,
                          Dense, Dropout, Activation, Reshape, Input, Concatenate, Add)
from keras.regularizers import l2
from keras.models import Model


def se_fn(x, amplifying_ratio):
  num_features = x.shape[-1].value
  x = GlobalAvgPool1D()(x)
  x = Reshape((1, num_features))(x)
  x = Dense(num_features * amplifying_ratio, activation='relu', kernel_initializer='glorot_uniform')(x)
  x = Dense(num_features, activation='sigmoid', kernel_initializer='glorot_uniform')(x)
  return x


def basic_block(x, num_features, weight_decay, _):
  x = Conv1D(num_features, kernel_size=3, padding='same', use_bias=True,
             kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPool1D(pool_size=3)(x)
  return x


def se_block(x, num_features, weight_decay, amplifying_ratio):
  x = basic_block(x, num_features, weight_decay, amplifying_ratio)
  x = Multiply()([x, se_fn(x, amplifying_ratio)])
  return x


def rese_block(x, num_features, weight_decay, amplifying_ratio):
  if num_features != x.shape[-1].value:
    shortcut = Conv1D(num_features, kernel_size=1, padding='same', use_bias=True,
                      kernel_regularizer=l2(weight_decay), kernel_initializer='glorot_uniform')(x)
    shortcut = BatchNormalization()(shortcut)
  else:
    shortcut = x
  x = Conv1D(num_features, kernel_size=3, padding='same', use_bias=True,
             kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(0.2)(x)
  x = Conv1D(num_features, kernel_size=3, padding='same', use_bias=True,
             kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
  x = BatchNormalization()(x)
  if amplifying_ratio > 0:
    x = Multiply()([x, se_fn(x, amplifying_ratio)])
  x = Add()([shortcut, x])
  x = Activation('relu')(x)
  x = MaxPool1D(pool_size=3)(x)
  return x


def resemul(x, block_type='se', multi=True, init_features=128, amplifying_ratio=16,
            drop_rate=0.5, weight_decay=0., num_classes=50):
  """Build a SampleCNN model.

  Args:
    batch_shape: A tensor shape including batch size (e.g. [23, 59049])
    block_type: A type of convolutional block among {se|rese|res|basic}
    multi: Whether to use multi-level feature aggregation.
    init_features: Number of feature maps of the first convolution.
    amplifying_ratio: Amplifying ratio of SE (not used for res and basic).
    weight_decay: L2 weight decay rate.
    drop_rate: Dropout rate.
    num_classes: Number of classes to predict.

  Returns:
    Keras Model.
  """
  if block_type == 'se':
    block = se_block
  elif block_type == 'rese':
    block = rese_block
  elif block_type == 'res':
    block = rese_block
    amplifying_ratio = -1
  elif block_type == 'basic':
    block = basic_block
  else:
    raise Exception('Unknown block type: ' + block_type)

  x = Input(tensor=x)
  x = Reshape([-1, 1])(x)

  x = Conv1D(init_features, kernel_size=3, strides=3, padding='valid', use_bias=True,
             kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  num_features = init_features
  layer_outputs = []
  for i in range(9):
    num_features *= 2 if (i == 2 or i == 8) else 1
    x = block(x, num_features, weight_decay, amplifying_ratio)
    layer_outputs.append(x)

  if multi:
    x = Concatenate()([GlobalMaxPool1D()(output) for output in layer_outputs[-3:]])
  else:
    x = GlobalMaxPool1D()(x)

  x = Dense(x.shape[-1].value, kernel_initializer='glorot_uniform')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  if drop_rate > 0.:
    x = Dropout(drop_rate)(x)
  x = Dense(num_classes, activation='sigmoid', kernel_initializer='glorot_uniform')(x)
  return x
