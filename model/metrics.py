import tensorflow as tf
import tensorflow_probability as tfp

class Mean(tf.keras.metrics.Mean):

  def __init__(self, name="mean", dtype=None):
    super().__init__(name=name, dtype=dtype)

def distance_error(target: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
  return tf.sqrt(
      tf.reduce_sum(tf.square(pred - target), axis=-1, keepdims=True))

class ADE(tf.keras.metrics.Metric):
  """Average Displacement Error over a n dimensional track.

  Calculates the mean L2 distance over all predicted timesteps.
  """
  # TODO: timestep?
  def __init__(self, num_history_steps=6, timestep=0.4, cutoff_seconds=None, at_cutoff=False, name='ADE'):
    """Initializes the ADE metric.

    Args:
      params: ModelParams
      cutoff_seconds: Cutoff up to which time the metric should be calculated
        in seconds.
      at_cutoff: If True metric will be calculated at cutoff timestep.
        Otherwise metric is calculated as average up to cutoff_seconds.
      name: Metric name.
    """
    super().__init__(name=name)
    self.cutoff_seconds = cutoff_seconds
    self.at_cutoff = at_cutoff
    if cutoff_seconds is None:
      self.cutoff_idx = None
    else:
      # +1 due to current time step.
      self.cutoff_idx = int(
          num_history_steps +
          cutoff_seconds / timestep) + 1

    self.num_predictions = self.add_weight(
        name='num_predictions', initializer='zeros')
    self.total_deviation = self.add_weight(
        name='total_deviation', initializer='zeros')

  def _reduce(self, ade_with_modes, input_batch, predictions):
    """Reduces mode dimension. The base class squeezes a single mode."""
    return tf.squeeze(ade_with_modes, axis=-1)

  def update_state(self, input_dict, predictions):
    should_predict = tf.cast(input_dict['should_predict'], tf.float32)
    #print("should_predict: ", should_predict.shape)

    target = input_dict['targets']
    target = target[..., :predictions['position'].shape[-1]]
    #print("target: ", target.shape)
    # [b, a, t, n, 3] -> [b, a, t, n, 1]
    per_position_ade = distance_error(
        target[..., tf.newaxis, :],
        predictions['position'])
    #print("per_position_ade: ", per_position_ade.shape)

    # Non-observed or past should not contribute to ade.
    deviation = tf.math.multiply_no_nan(per_position_ade,
                                        should_predict[..., tf.newaxis, :])
    #print("deviation: ", deviation.shape)
    # Chop off the un-wanted time part.
    # [b, a, cutoff_idx, 1]
    if self.at_cutoff and self.cutoff_seconds is not None:
      deviation = deviation[:, self.cutoff_idx-1:self.cutoff_idx, :]
      num_predictions = tf.reduce_sum(
          should_predict[:, self.cutoff_idx-1:self.cutoff_idx, :])
    else:
      deviation = deviation[:, :self.cutoff_idx, :]
      num_predictions = tf.reduce_sum(should_predict[:, :self.cutoff_idx, :])
    #print("deviation: ", deviation.shape)
    #print("num_predictions: ", num_predictions.shape)
    
    # Reduce along time
    deviation = tf.reduce_sum(deviation, axis=1)
    #print("deviation: ", deviation.shape)
    # Reduce along modes
    deviation = self._reduce(deviation, input_dict, predictions)
    #print("deviation: ", deviation.shape)
    # [1]
    deviation = tf.reduce_sum(deviation)
    #print("deviation: ", deviation.shape)

    self.num_predictions.assign_add(num_predictions)
    self.total_deviation.assign_add(deviation)

  def result(self):
    return self.total_deviation / self.num_predictions

  def reset_states(self):
    self.num_predictions.assign(0)
    self.total_deviation.assign(0.0)

class MinADE(ADE):
  """Takes the minimum over all modes."""

  def _reduce(self, ade_with_modes, input_batch, predictions):
    return tf.reduce_min(ade_with_modes, axis=-2)

class MLADE(ADE):
  """Takes the maximum likelihood mode."""

  def _reduce(self, ade_with_modes, input_batch, predictions):
    # Get index of mixture component with highest probability
    # [b, a=1, t=1, n]
    ml_indices = tf.math.argmax(predictions['mixture_logits'], axis=-1)
    a = ade_with_modes.shape[1]
    ml_indices = tf.tile(
        tf.squeeze(ml_indices, axis=1), [1, a])[..., tf.newaxis]
    #ml_indices = tf.squeeze(ml_indices, axis=1)

    #return tf.gather(
       # ade_with_modes, indices=ml_indices, batch_dims=1, axis=-2)[..., 0, :]
    return tf.gather(
            ade_with_modes, indices=ml_indices, batch_dims=1, axis=-2)[..., 0,:]

def force_positive(x, eps=1e-6):
  return tf.keras.activations.elu(x) + 1. + eps

def to_positive_definite_scale_tril(logit_sigma):
  tril = tfp.math.fill_triangular(logit_sigma)
  scale_tril = tf.linalg.set_diag(
      tril,
      force_positive(tf.linalg.diag_part(tril)))
  return scale_tril

def get_position_distribution(model_output):
  """Multivariate Normal distribution over position."""
  p_pos = tfp.distributions.MultivariateNormalTriL(
      loc=model_output['position'],
      scale_tril=to_positive_definite_scale_tril(
          model_output['position_raw_scale']))

  return p_pos

def get_multimodal_position_distribution(model_output):
  """Multivariate Normal Mixture distribution over position."""
  p_pos = get_position_distribution(model_output)

  p_pos_mm = tfp.distributions.MixtureSameFamily(
      mixture_distribution=tfp.distributions.Categorical(
          logits=model_output['mixture_logits']),
      components_distribution=p_pos)

  return p_pos_mm

class PositionNegativeLogLikelihood(tf.keras.metrics.Metric):
  """Position Negative Log Likelihood."""

  def __init__(self, num_history_steps=6, timestep=0.4,cutoff_seconds=None, at_cutoff=False,
               name='PosNLL'):
    """Initializes the PositionNegativeLogLikelihood metric.

    Args:
      params: ModelParams
      cutoff_seconds: Cutoff up to which time the metric should be calculated
        in seconds.
      at_cutoff: If True metric will be calculated at cutoff timestep.
        Otherwise metric is calculated as average up to cutoff_seconds.
      name: Metric name.
    """
    super().__init__(name=name)
    self.cutoff_seconds = cutoff_seconds
    self.at_cutoff = at_cutoff
    if cutoff_seconds is None:
      self.cutoff_idx = None
    else:
      # +1 due to current time step.
      self.cutoff_idx = int(
          num_history_steps +
          cutoff_seconds / timestep) + 1

    self.num_predictions = self.add_weight(
        name='num_predictions', initializer='zeros')
    self.total_deviation = self.add_weight(
        name='total_deviation', initializer='zeros')

  def update_state(self, input_dict, predictions):
    should_predict = tf.cast(input_dict['should_predict'], tf.float32)
    #print("should_predict: ", should_predict.shape)

    p_pos = get_multimodal_position_distribution(predictions)
    #print("p_pos: ", p_pos.shape)

    target = input_dict['targets']
    target = target[..., :p_pos.event_shape_tensor()[0]]

    # [b, a, t, n, 1]
    per_position_nll = -p_pos.log_prob(target)[..., tf.newaxis]
    #print("per_position_nll: ", per_position_nll)

    # Non-observed or past should not contribute to metric.
    nll = tf.math.multiply_no_nan(per_position_nll, should_predict)
    # Chop off the un-wanted time part.
    # [b, a, cutoff_idx, 1]
    if self.at_cutoff and self.cutoff_seconds is not None:
      nll = nll[:, self.cutoff_idx-1:self.cutoff_idx, :]
      num_predictions = tf.reduce_sum(
          should_predict[:, self.cutoff_idx-1::self.cutoff_idx, :])
    else:
      nll = nll[:, :self.cutoff_idx, :]
      num_predictions = tf.reduce_sum(should_predict[:, :self.cutoff_idx, :])
    #print("nll: ", nll.shape)
    #print("num_predictions: ", num_predictions)

    # [1]
    nll = tf.reduce_sum(nll)
    #print("nll: ", nll.shape)

    self.num_predictions.assign_add(num_predictions)
    self.total_deviation.assign_add(nll)

  def result(self):
    return self.total_deviation / self.num_predictions

  def reset_states(self):
    self.num_predictions.assign(0)
    self.total_deviation.assign(0.0)
