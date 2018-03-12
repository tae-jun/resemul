import os
import re
import math
import warnings
import copy
import six
import tensorflow as tf
import numpy as np

from glob import glob
from keras.models import Model
from keras import optimizers
from keras import losses
from keras import callbacks as cbks
from keras import backend as K
from keras import metrics as metrics_module
from keras.engine.training import _collect_metrics


def make_path(*paths):
  path = os.path.join(*[str(path) for path in paths])
  path = os.path.realpath(path)
  return path


def find_best_checkpoint(*dirs, prefix='ckpt'):
  best_checkpoint_path = None
  best_epoch = -1
  best_val_loss = 1e+10
  for dir in dirs:
    checkpoint_paths = glob('{}/{}*'.format(dir, prefix))
    for checkpoint_path in checkpoint_paths:
      epoch = int(re.findall('e\d+', checkpoint_path)[0][1:])
      val_loss = float(re.findall('l\d\.\d+', checkpoint_path)[0][1:])

      if val_loss < best_val_loss:
        best_checkpoint_path = checkpoint_path
        best_epoch = epoch
        best_val_loss = val_loss

  return best_checkpoint_path, best_epoch, best_val_loss


def calculate_steps(n_examples, n_segments, batch_size):
  steps = 1. * n_examples * n_segments
  steps = math.ceil(steps / batch_size)
  return steps


class TFRecordModel(Model):
  def __init__(self, inputs, outputs, val_inputs=None, name=None):
    super(TFRecordModel, self).__init__(inputs, outputs, name=name)

    # Prepare val_inputs.
    if val_inputs is None:
      self.val_inputs = []
    elif isinstance(val_inputs, (list, tuple)):
      self.val_inputs = list(val_inputs)  # Tensor or list of tensors.
    else:
      self.val_inputs = [val_inputs]

    # Prepare val_outputs.
    if val_inputs is None:
      self.val_outputs = []
    else:
      val_outputs = self(val_inputs)
      if isinstance(val_outputs, (list, tuple)):
        self.val_outputs = list(val_outputs)  # Tensor or list of tensors.
      else:
        self.val_outputs = [val_outputs]

  def compile_tfrecord(self, optimizer, loss, y, metrics=None,
                       y_val=None):
    """Configures the model for training.

    # Arguments
        optimizer: str (name of optimizer) or optimizer object.
          See [optimizers](/optimizers).
        loss: str (name of objective function) or objective function.
          See [losses](/losses).
          If the model has multiple outputs, you can use a different loss
          on each output by passing a dictionary or a list of losses.
          The loss value that will be minimized by the model
          will then be the sum of all individual losses.
        metrics: list of metrics to be evaluated by the model
          during training and testing.
          Typically you will use `metrics=['accuracy']`.
          To specify different metrics for different outputs of a
          multi-output model, you could also pass a dictionary,
          such as `metrics={'output_a': 'accuracy'}`.

    # Raises
        ValueError: In case of invalid arguments for
            `optimizer`, `loss`, `metrics` or `sample_weight_mode`.
    """
    loss = loss or {}
    self.optimizer = optimizers.get(optimizer)
    self.loss = loss
    self.sample_weight_mode = None
    self.loss_weights = None
    self.y_val = y_val

    do_validation = bool(len(self.val_inputs) > 0)
    if do_validation and y_val is None:
      raise ValueError('When you use validation inputs, '
                       'you should provide y_val.')

    # Prepare loss functions.
    if isinstance(loss, dict):
      for name in loss:
        if name not in self.output_names:
          raise ValueError('Unknown entry in loss '
                           'dictionary: "' + name + '". '
                                                    'Only expected the following keys: ' +
                           str(self.output_names))
      loss_functions = []
      for name in self.output_names:
        if name not in loss:
          warnings.warn('Output "' + name +
                        '" missing from loss dictionary. '
                        'We assume this was done on purpose, '
                        'and we will not be expecting '
                        'any data to be passed to "' + name +
                        '" during training.', stacklevel=2)
        loss_functions.append(losses.get(loss.get(name)))
    elif isinstance(loss, list):
      if len(loss) != len(self.outputs):
        raise ValueError('When passing a list as loss, '
                         'it should have one entry per model outputs. '
                         'The model has ' + str(len(self.outputs)) +
                         ' outputs, but you passed loss=' +
                         str(loss))
      loss_functions = [losses.get(l) for l in loss]
    else:
      loss_function = losses.get(loss)
      loss_functions = [loss_function for _ in range(len(self.outputs))]
    self.loss_functions = loss_functions

    # Prepare training targets of model.
    if isinstance(y, (list, tuple)):
      y = list(y)  # Tensor or list of tensors.
    else:
      y = [y]
    self.targets = []
    for i in range(len(self.outputs)):
      target = y[i]
      self.targets.append(target)

    # Prepare validation targets of model.
    if isinstance(y_val, (list, tuple)):
      y_val = list(y_val)  # Tensor or list of tensors.
    else:
      y_val = [y_val]
    self.y_val = y_val
    self.val_targets = []
    for i in range(len(self.val_outputs)):
      val_target = y_val[i]
      self.val_targets.append(val_target)

    # Prepare metrics.
    self.metrics = metrics
    self.metrics_names = ['loss']
    self.metrics_tensors = []
    self.val_metrics_names = ['loss']
    self.val_metrics_tensors = []

    # Compute total training loss.
    total_loss = None
    for i in range(len(self.outputs)):
      y_true = self.targets[i]
      y_pred = self.outputs[i]
      loss_function = loss_functions[i]
      val_output_loss = K.mean(loss_function(y_true, y_pred))
      if len(self.outputs) > 1:
        self.metrics_tensors.append(val_output_loss)
        self.metrics_names.append(self.output_names[i] + '_loss')
      if total_loss is None:
        total_loss = val_output_loss
      else:
        total_loss += val_output_loss
    if total_loss is None:
      if not self.losses:
        raise RuntimeError('The model cannot be compiled '
                           'because it has no loss to optimize.')
      else:
        total_loss = 0.

    # Compute total validation loss.
    val_total_loss = None
    for i in range(len(self.val_outputs)):
      y_true = self.val_targets[i]
      y_pred = self.val_outputs[i]
      loss_function = loss_functions[i]
      val_output_loss = K.mean(loss_function(y_true, y_pred))
      if len(self.outputs) > 1:
        self.val_metrics_tensors.append(val_output_loss)
        self.val_metrics_names.append(self.output_names[i] + '_val_loss')
      if val_total_loss is None:
        val_total_loss = val_output_loss
      else:
        val_total_loss += val_output_loss
    if val_total_loss is None:
      if not self.losses and do_validation:
        raise RuntimeError('The model cannot be compiled '
                           'because it has no loss to optimize.')
      else:
        val_total_loss = 0.

    # Add regularization penalties
    # and other layer-specific losses.
    for loss_tensor in self.losses:
      total_loss += loss_tensor
      val_total_loss += loss_tensor

    # List of same size as output_names.
    # contains tuples (metrics for output, names of metrics).
    nested_metrics = _collect_metrics(metrics, self.output_names)

    def append_metric(layer_num, metric_name, metric_tensor):
      """Helper function used in loop below."""
      if len(self.output_names) > 1:
        metric_name = self.output_layers[layer_num].name + '_' + metric_name
      self.metrics_names.append(metric_name)
      self.metrics_tensors.append(metric_tensor)

    for i in range(len(self.outputs)):
      y_true = self.targets[i]
      y_pred = self.outputs[i]
      output_metrics = nested_metrics[i]
      for metric in output_metrics:
        if metric == 'accuracy' or metric == 'acc':
          # custom handling of accuracy
          # (because of class mode duality)
          output_shape = self.internal_output_shapes[i]
          acc_fn = None
          if (output_shape[-1] == 1 or
                  self.loss_functions[i] == losses.binary_crossentropy):
            # case: binary accuracy
            acc_fn = metrics_module.binary_accuracy
          elif self.loss_functions[i] == losses.sparse_categorical_crossentropy:
            # case: categorical accuracy with sparse targets
            acc_fn = metrics_module.sparse_categorical_accuracy
          else:
            acc_fn = metrics_module.categorical_accuracy

          append_metric(i, 'acc', K.mean(acc_fn(y_true, y_pred)))
        else:
          metric_fn = metrics_module.get(metric)
          metric_result = metric_fn(y_true, y_pred)
          metric_result = {
            metric_fn.__name__: metric_result
          }
          for name, tensor in six.iteritems(metric_result):
            append_metric(i, name, tensor)

    def append_val_metric(layer_num, metric_name, metric_tensor):
      """Helper function used in loop below."""
      if len(self.output_names) > 1:
        metric_name = self.output_layers[layer_num].name + '_val_' + metric_name
      self.val_metrics_names.append(metric_name)
      self.val_metrics_tensors.append(metric_tensor)

    for i in range(len(self.val_outputs)):
      y_true = self.val_targets[i]
      y_pred = self.val_outputs[i]
      output_metrics = nested_metrics[i]
      for metric in output_metrics:
        if metric == 'accuracy' or metric == 'acc':
          # custom handling of accuracy
          # (because of class mode duality)
          output_shape = self.internal_output_shapes[i]
          acc_fn = None
          if (output_shape[-1] == 1 or
                  self.loss_functions[i] == losses.binary_crossentropy):
            # case: binary accuracy
            acc_fn = metrics_module.binary_accuracy
          elif self.loss_functions[i] == losses.sparse_categorical_crossentropy:
            # case: categorical accuracy with sparse targets
            acc_fn = metrics_module.sparse_categorical_accuracy
          else:
            acc_fn = metrics_module.categorical_accuracy

          append_val_metric(i, 'acc', K.mean(acc_fn(y_true, y_pred)))
        else:
          metric_fn = metrics_module.get(metric)
          metric_result = metric_fn(y_true, y_pred)
          metric_result = {
            metric_fn.__name__: metric_result
          }
          for name, tensor in six.iteritems(metric_result):
            append_val_metric(i, name, tensor)

    # Prepare gradient updates and state updates.
    self.total_loss = total_loss
    self.val_total_loss = val_total_loss

    # Functions for train, test and predict will
    # be compiled lazily when required.
    # This saves time when the user is not using all functions.
    self.train_function = None
    self.val_function = None
    self.test_function = None
    self.predict_function = None

    # Collected trainable weights and sort them deterministically.
    trainable_weights = self.trainable_weights
    # Sort weights by name.
    if trainable_weights:
      trainable_weights.sort(key=lambda x: x.name)
    self._collected_trainable_weights = trainable_weights

  def _make_tfrecord_train_function(self):
    if not hasattr(self, 'train_function'):
      raise RuntimeError('You must compile your model before using it.')
    if self.train_function is None:
      inputs = []
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        inputs += [K.learning_phase()]

      training_updates = self.optimizer.get_updates(
        self._collected_trainable_weights,
        self.constraints,
        self.total_loss)
      updates = self.updates + training_updates
      # Gets loss and metrics. Updates weights at each call.
      self.train_function = K.function(inputs,
                                       [self.total_loss] + self.metrics_tensors,
                                       updates=updates)

  def _make_tfrecord_test_function(self):
    if not hasattr(self, 'test_function'):
      raise RuntimeError('You must compile your model before using it.')
    if self.test_function is None:
      inputs = []
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        inputs += [K.learning_phase()]
      # Return loss and metrics, no gradient updates.
      # Does update the network states.
      self.test_function = K.function(inputs,
                                      [self.total_loss] + self.metrics_tensors,
                                      updates=self.state_updates)

  def _make_tfrecord_val_function(self):
    if not hasattr(self, 'val_function'):
      raise RuntimeError('You must compile your model before using it.')
    if self.val_function is None:
      inputs = []
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        inputs += [K.learning_phase()]
      # Return loss and metrics, no gradient updates.
      # Does update the network states.
      self.val_function = K.function(
        inputs,
        [self.val_total_loss] + self.val_metrics_tensors,
        updates=self.state_updates)

  def fit_tfrecord(self, steps_per_epoch,
                   epochs=1,
                   verbose=1,
                   callbacks=None,
                   validation_steps=None,
                   initial_epoch=0):
    epoch = initial_epoch

    self._make_tfrecord_train_function()

    do_validation = bool(len(self.val_inputs) > 0)
    if do_validation and not validation_steps:
      raise ValueError('When using a validation batch, '
                       'you must specify a value for '
                       '`validation_steps`.')

    # Prepare display labels.
    out_labels = self._get_deduped_metrics_names()

    if do_validation:
      callback_metrics = copy.copy(out_labels) + ['val_' + n
                                                  for n in out_labels]
    else:
      callback_metrics = copy.copy(out_labels)

    # prepare callbacks
    self.history = cbks.History()
    callbacks = [cbks.BaseLogger()] + (callbacks or []) + [self.history]
    if verbose:
      callbacks += [cbks.ProgbarLogger(count_mode='steps')]
    callbacks = cbks.CallbackList(callbacks)

    # it's possible to callback a different model than self:
    if hasattr(self, 'callback_model') and self.callback_model:
      callback_model = self.callback_model
    else:
      callback_model = self
    callbacks.set_model(callback_model)
    callbacks.set_params({
      'epochs': epochs,
      'steps': steps_per_epoch,
      'verbose': verbose,
      'do_validation': do_validation,
      'metrics': callback_metrics,
    })
    callbacks.on_train_begin()

    if do_validation:
      val_sample_weight = None
      for cbk in callbacks:
        cbk.validation_data = [self.val_inputs, self.y_val, val_sample_weight]

    try:
      sess = K.get_session()
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      callback_model.stop_training = False
      while epoch < epochs:
        callbacks.on_epoch_begin(epoch)
        steps_done = 0
        batch_index = 0
        while steps_done < steps_per_epoch:
          # build batch logs
          batch_logs = {
            'batch': batch_index,
            'size': self.inputs[0].shape[0].value
          }
          callbacks.on_batch_begin(batch_index, batch_logs)

          if self.uses_learning_phase and not isinstance(K.learning_phase(),
                                                         int):
            ins = [1.]
          else:
            ins = []
          outs = self.train_function(ins)

          if not isinstance(outs, list):
            outs = [outs]
          for l, o in zip(out_labels, outs):
            batch_logs[l] = o

          callbacks.on_batch_end(batch_index, batch_logs)

          # Construct epoch logs.
          epoch_logs = {}
          batch_index += 1
          steps_done += 1

          # Epoch finished.
          if steps_done >= steps_per_epoch and do_validation:
            val_outs = self._validate_tfrecord(steps=validation_steps)
            if not isinstance(val_outs, list):
              val_outs = [val_outs]
            # Same labels assumed.
            for l, o in zip(out_labels, val_outs):
              epoch_logs['val_' + l] = o

        callbacks.on_epoch_end(epoch, epoch_logs)
        epoch += 1
        if callback_model.stop_training:
          break

    finally:
      # TODO: If you close the queue, you can't open it again..
      # coord.request_stop()
      # coord.join(threads)
      pass

    callbacks.on_train_end()
    return self.history

  def _validate_tfrecord(self, steps):
    self._make_tfrecord_val_function()

    steps_done = 0
    all_outs = []
    batch_sizes = []

    while steps_done < steps:
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        ins = [0.]
      else:
        ins = []
      outs = self.val_function(ins)
      if len(outs) == 1:
        outs = outs[0]

      batch_size = self.val_inputs[0].shape[0].value
      all_outs.append(outs)

      steps_done += 1
      batch_sizes.append(batch_size)

    if not isinstance(outs, list):
      return np.average(np.asarray(all_outs),
                        weights=batch_sizes)
    else:
      averages = []
      for i in range(len(outs)):
        averages.append(np.average([out[i] for out in all_outs],
                                   weights=batch_sizes))
      return averages

  def evaluate_tfrecord(self, steps):
    """Evaluates the model on a data generator.

    The generator should return the same kind of data
    as accepted by `test_on_batch`.

    # Arguments
        x_batch:
        y_batch:
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
        stop_queue_runners: If True, stop queue runners after evaluation.

    # Returns
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    # Raises
        ValueError: In case the generator yields
          data in an invalid format.
    """
    self._make_tfrecord_test_function()

    steps_done = 0
    all_outs = []
    batch_sizes = []

    try:
      sess = K.get_session()
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      while steps_done < steps:

        if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
          ins = [0.]
        else:
          ins = []
        outs = self.test_function(ins)
        if len(outs) == 1:
          outs = outs[0]

        batch_size = self.inputs[0].shape[0].value
        all_outs.append(outs)

        steps_done += 1
        batch_sizes.append(batch_size)

    finally:
      # TODO: If you close the queue, you can't open it again..
      # if stop_queue_runners:
      #   coord.request_stop()
      #   coord.join(threads)
      pass

    if not isinstance(outs, list):
      return np.average(np.asarray(all_outs),
                        weights=batch_sizes)
    else:
      averages = []
      for i in range(len(outs)):
        averages.append(np.average([out[i] for out in all_outs],
                                   weights=batch_sizes))
      return averages

  def _make_tfrecord_predict_function(self):
    if not hasattr(self, 'predict_function'):
      self.predict_function = None
    if self.predict_function is None:
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        inputs = [K.learning_phase()]
      else:
        inputs = []
      # Gets network outputs. Does not update weights.
      # Does update the network states.
      self.predict_function = K.function(inputs,
                                         self.outputs,
                                         updates=self.state_updates)

  def predict_tfrecord(self, x_batch):
    if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
      ins = [0.]
    else:
      ins = []
    self._make_tfrecord_predict_function()

    try:
      sess = K.get_session()
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      outputs = self.predict_function(ins)

    finally:
      # TODO: If you close the queue, you can't open it again..
      # if stop_queue_runners:
      #   coord.request_stop()
      #   coord.join(threads)
      pass

    if len(outputs) == 1:
      return outputs[0]
    return outputs
