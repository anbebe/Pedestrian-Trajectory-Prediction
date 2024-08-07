import tensorflow as tf
import tensorflow_probability as tfp

class Loss(object):
  """Base class for Human Scene Transformer Losses.
  """

  def __init__(self, name='Loss', clip_loss_max=tf.float32.max):
    self.clip_loss_max = clip_loss_max
    self.name = name

  @tf.function
  def __call__(self, input_batch, predictions):
    return self.call(input_batch, predictions)

  def call(self, input_dict, predictions):
    """Calculates loss for fields which should be predicted."""

    # [b, a, t]
    should_predict = input_dict['should_predict'][..., 0]
    # [b, a, t]
    loss_per_batch = self.get_per_batch_loss(input_dict, predictions)

    loss_per_batch = tfp.math.clip_by_value_preserve_gradient(
        loss_per_batch,
        tf.float32.min,
        self.clip_loss_max)

    # Compute loss only on positions w/ should_predict == True.
    #print("should predict Loss:", should_predict.shape)
    should_predict_ind = tf.where(should_predict)
    loss_should_predict_mat = tf.gather_nd(
        params=loss_per_batch, indices=should_predict_ind)

    loss_should_predict = tf.reduce_mean(loss_should_predict_mat)
    # If there are no agents to be predicted xyz_loss_should_predict can be NaN
    loss_should_predict = tf.math.multiply_no_nan(
        loss_should_predict,
        tf.cast(tf.math.reduce_any(should_predict), tf.float32))

    loss_dict = {
        'loss': loss_should_predict,
        f'{self.name}_loss': loss_should_predict
    }
    return loss_dict

  def get_per_batch_loss(self, input_batch, predictions):
    raise NotImplementedError

class MinNLLPositionMixtureCategoricalCrossentropyLoss(Loss):
  """MinNLLPositionNLLLoss and MixtureCategoricalCrossentropyLoss."""

  def __init__(self, **kwargs):
    super().__init__(name='MinNLLMixture', **kwargs)
    self.position_loss_obj = MinNLLPositionLoss()
    self.mixture_loss_obj = MinNLLMixtureCategoricalCrossentropyLoss()

  def call(self, input_batch, predictions):
    position_loss = self.position_loss_obj(input_batch, predictions)
    mixture_loss = self.mixture_loss_obj(input_batch, predictions)

    loss = position_loss['loss'] + mixture_loss['loss']

    #print("loss: ", loss)

    loss_dict = {**position_loss, **mixture_loss}

    loss_dict['loss'] = loss

    return loss_dict
  
class PositionNLLLoss(Loss):
  """Position NLL Loss for human trajectory predictions."""

  def __init__(self, **kwargs):
    super().__init__(name='position', **kwargs)

  # from model.output_distributions
  def force_positive(self, x, eps=1e-6):
    return tf.keras.activations.elu(x) + 1. + eps

  # from model.output_distributions
  def to_positive_definite_scale_tril(self, logit_sigma):
    tril = tfp.math.fill_triangular(logit_sigma)
    scale_tril = tf.linalg.set_diag(
        tril,
        self.force_positive(tf.linalg.diag_part(tril)))
    return scale_tril
  
  # from model.output_distributions
  def get_position_distribution(self, model_output):
    """Multivariate Normal distribution over position."""
    p_pos = tfp.distributions.MultivariateNormalTriL(
        loc=model_output['position'],
        scale_tril=self.to_positive_definite_scale_tril(
            model_output['position_raw_scale']))
    return p_pos

  def get_per_batch_loss(self, input_dict, predictions):
    """Negative log probability of ground truth with respect to predictions."""
    # [b, a, t, 3]
    p_position = self.get_position_distribution(predictions)

    # [b, a, t, 1]
    position_nll = -p_position.log_prob(
        input_dict['targets'])
    return position_nll
  
class MinNLLPositionLoss(PositionNLLLoss):
  """MinNLLPositionNLL loss."""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.mixture_loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

  def get_per_batch_loss(self, input_dict, predictions):
    """Negative log probability of mode with smallest ADE."""
    input_dict = input_dict.copy()

    input_dict['targets'] = input_dict['targets'][..., tf.newaxis, :]
    # [b, a, t, n]
    position_nll = super().get_per_batch_loss(input_dict, predictions)

    # [b, a, t, 1]
    should_predict = tf.cast(input_dict['should_predict'], tf.float32)

    # Extract the mixture logits and compute the mode probabilities
    #mixture_logits = predictions['mixture_logits']  # [b, 1, M]
    #mode_probabilities = tf.nn.softmax(mixture_logits, axis=-1)  # [b, 1, M]

    # [b, a, t, n, 1]
    per_position_nll = (
        position_nll[..., tf.newaxis] * should_predict[..., tf.newaxis, :]
    )
    #print("per_position_nll ", per_position_nll.shape)

    # Get mode with minimum NLL
    # [b, a, n, 1]
    per_mode_nll_sum = tf.reduce_sum(per_position_nll, axis=1)
    #print("per_mode_nll_sum ", per_mode_nll_sum.shape)

    t = tf.shape(position_nll)[1]

    # [b, a, 1]
    min_nll_indices = tf.math.argmin(per_mode_nll_sum, axis=-2)
    #print("min_nll_indices ", min_nll_indices.shape)

    # [b, a, t, 1]
    min_nll_indices_tiled = tf.tile(
        min_nll_indices[..., tf.newaxis], [1, t, 1])
    #print("min_nll_indices_tiled ", min_nll_indices_tiled.shape)

    # [b, a, t]
    position_nll_min_ade = tf.gather(
        position_nll, indices=min_nll_indices_tiled, batch_dims=2, axis=-1
        )[..., 0]
    #print("position_nll_min_ade ", position_nll_min_ade.shape)

    #final_loss = position_nll_min_ade - tf.reduce_sum(tf.math.log(mode_probabilities), axis=-1)

    return position_nll_min_ade

# used for JRDB and Pedestrians
class MinNLLMixtureCategoricalCrossentropyLoss(PositionNLLLoss):
  """Categorical Corssentropy Loss for Mixture Weight."""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.mixture_loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

  def get_per_batch_loss(self, input_dict, predictions):
    """Negative log probability of ground truth with respect to predictions."""
    input_dict= input_dict.copy()
    input_dict['targets'] = input_dict[
        'targets'][..., tf.newaxis, :]

    # Calculate ADE
    # [b, a, t, n, 1]
    position_nll = super().get_per_batch_loss(input_dict, predictions)
    position_nll = tf.expand_dims(position_nll, axis=-1)

    # [b, a, t, 1]
    ##should_predict = tf.cast(input_batch['should_predict'], tf.float32)
    should_predict = tf.cast(input_dict['should_predict'], tf.float32)

    # [b, a, t, n, 1]
    #per_position_nll = (
        #position_nll * should_predict[..., tf.newaxis, :]
    #)
    per_position_nll = (
        position_nll * should_predict[..., tf.newaxis, :]
    )

    # Get mode with minimum NLL
    # [b, a, n, 1]
    ##per_mode_nll_sum = tf.reduce_sum(per_position_nll, axis=2)
    per_mode_nll_sum = tf.reduce_sum(per_position_nll, axis=1)

    n = tf.shape(position_nll)[2]

    # [b, a, 1]
    min_nll_indices = tf.math.argmin(per_mode_nll_sum, axis=-2)

    # [b, a, n]
    min_nll_indices_one_hot = tf.one_hot(min_nll_indices[..., 0], n)

    # [b, a]
    # TODO
    #mixture_loss = self.mixture_loss(
        #min_nll_indices_one_hot,
        #tf.tile(predictions['mixture_logits'][..., 0, :], [1, a, 1]))

    
    mixture_loss = self.mixture_loss(
        min_nll_indices_one_hot,
        predictions['mixture_logits'][..., 0, :])

    return mixture_loss

  def call(self, input_dict, predictions):
    """Calculates loss."""

    # [b, a]
    should_predict = tf.reduce_any(
        input_dict['should_predict'][..., 0], axis=-1)
    #should_predict = tf.reduce_any(
       # tf.math.logical_not(predictions['mask'][..., 0]), axis=-1)
    #should_predict = tf.expand_dims(should_predict, axis=-1)
    # [b, a]
    loss_per_batch = self.get_per_batch_loss(input_dict, predictions)

    # Compute loss only on positions w/ should_predict == True.
    should_predict_ind = tf.where(should_predict)
    loss_should_predict_mat = tf.gather_nd(
        params=loss_per_batch, indices=should_predict_ind)

    loss_should_predict = tf.reduce_mean(loss_should_predict_mat)
    # If there are no agents to be predicted xyz_loss_should_predict can be NaN
    loss_should_predict = tf.math.multiply_no_nan(
        loss_should_predict,
        tf.cast(tf.math.reduce_any(should_predict), tf.float32))

    # Mixture weights are per scene. So we do not have to mask anything
    loss_dict = {
        'loss': loss_should_predict,
        f'{self.name}_loss': loss_should_predict
    }
    return loss_dict
  

# used for JRDB Challenge
class MultimodalPositionNLLLoss(Loss):
  """Position loss for human trajectory predictions w/ scene transformer."""

  def __init__(self, **kwargs):
    super().__init__(name='position', **kwargs)

  # from model.output_distributions
  def force_positive(self, x, eps=1e-6):
    return tf.keras.activations.elu(x) + 1. + eps

  # from model.output_distributions
  def to_positive_definite_scale_tril(self, logit_sigma):
    tril = tfp.math.fill_triangular(logit_sigma)
    scale_tril = tf.linalg.set_diag(
        tril,
        self.force_positive(tf.linalg.diag_part(tril)))
    return scale_tril

  # from model.output_distributions
  def get_position_distribution(self, model_output):
    """Multivariate Normal distribution over position."""
    p_pos = tfp.distributions.MultivariateNormalTriL(
        loc=model_output['position'],
        scale_tril=self.to_positive_definite_scale_tril(
            model_output['position_raw_scale']))
    return p_pos

  # from model.output_distributions
  def get_multimodal_position_distribution(self, model_output):
    """Multivariate Normal Mixture distribution over position."""
    p_pos = self.get_position_distribution(model_output)

    p_pos_mm = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=model_output['mixture_logits']),
        components_distribution=p_pos)

    return p_pos_mm

  def get_per_batch_loss(self, input_dict, predictions):
    """Negative log probability of ground truth with respect to predictions."""
    # [b, a, t, n, 3]
    p_position_mm = self.get_multimodal_position_distribution(
        predictions)

    target = input_dict['targets']
    target = target[..., :p_position_mm.event_shape_tensor()[0]]

    # [b, a, t]
    position_nll = -p_position_mm.log_prob(target)

    return position_nll