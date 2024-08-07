import tensorflow as tf
import numpy as np
from typing import Tuple, Optional
import keras_nlp

class PreprocessLayer(tf.keras.layers.Layer):
    """ Applies the masking to the sequence
    """

    def __init__(self, num_steps=15, num_history_steps=5):
        super().__init__()
        self._num_steps = num_steps
        self._num_history_steps = num_history_steps

    # from BPIsHiddenGenerator
    def calc_hidden_mask(self, batch_size=32, sequence_length=15):
        """Returns the is_hidden tensor for behavior prediction.

        Always returns 0 (not hidden) for history/current steps and 1 (hidden)
        for future steps.

        Args:
        num_agents: Number of agents in the scene.
        train_progress: A float between 0 to 1 representing the overall progress
            of training. This float can be current_step / total_training_steps. This
            float can be used for training w/ an annealing schedule.

        Returns:
        is_hidden: The is_hidden tensor for behavior prediction.
        """
        # [1, a, t, 1].
        is_hidden = np.ones([1, self._num_steps, 1],
                            dtype=bool)
        is_hidden[:, :self._num_history_steps + 1, :] = False
        return is_hidden
    
    def _set_elems_to_value(self, target, should_set, new_val):
        target = tf.where(should_set, tf.cast(new_val, target.dtype), target)
        return target

    def call(self,
           raw_input_batch: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        input_batch = raw_input_batch
  
        is_hidden = self.calc_hidden_mask() #tf.convert_to_tensor

        input_batch_new = []

        for feature_batch in input_batch:
            input_batch_new.append(tf.where(
            tf.math.is_nan(feature_batch),
            tf.zeros_like(feature_batch),
            feature_batch))
        
        t = input_batch_new[0]
        if t.dtype.is_floating:
            has_data =  tf.math.logical_not(
                tf.reduce_any(tf.math.is_nan(t), axis=-1, keepdims=True))
        else:
            has_data =  tf.math.logical_not(
                tf.reduce_any(t == t.dtype.min, axis=-1, keepdims=True))
        
        has_historic_data = tf.reduce_any( has_data[..., :self._num_history_steps + 1, :], axis=-2, keepdims=True)
        should_predict = tf.logical_and(is_hidden,tf.logical_and(has_data, has_historic_data))

        feature_is_padded = tf.logical_not(has_data)
        masked_input_pos = self._set_elems_to_value( input_batch_new[0], tf.logical_or(feature_is_padded,is_hidden), 0.)
        masked_input_pose = self._set_elems_to_value( input_batch_new[1], tf.logical_or(feature_is_padded,is_hidden), 0.)

        targets = input_batch_new[0]

        output_dict={
            "masked_inputs": (masked_input_pos, masked_input_pose),
            "has_data": has_data, 
            "is_hidden": is_hidden, 
            "targets": targets,
            "has_historic_data": has_historic_data,
            "should_predict": should_predict
        }

        return output_dict


""" Adapted Agent Position Encoding Layer from source: https://github.com/google-research/human-scene-transformer/blob/main/human_scene_transformer/model/agent_feature_encoder.py    """
class AgentPositionEncoder(tf.keras.layers.Layer):
  """Encodes agents spatial positions."""

  def __init__(self, output_shape, embedding_size):
    
    super().__init__()

    #self.embedding_layer = SinusoidalEmbeddingLayer(
       # hidden_size=embedding_size) # output_shape (batch_sie, sequence_length, feature size, hidden_size)
    self.embedding_layer = keras_nlp.layers.SinePositionEncoding(max_wavelength=10000)
    #self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)  
    self.mlp = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=output_shape,
        bias_axes='h',
        
        activation=None)

  """def call(self, input_batch):
    normalized_input = input_batch[0] #self.layer_norm(input_batch[0])
    embedded_input = self.embedding_layer(normalized_input)
    return self.mlp(embedded_input)"""
  def call(self, input_dict):
    is_hidden = input_dict["is_hidden"]
    has_data = input_dict["has_data"]
    input_batch = input_dict["masked_inputs"]
    not_is_hidden = tf.logical_not(is_hidden)
    mask = tf.logical_and(has_data, not_is_hidden)
    mask = tf.repeat(mask, tf.shape(input_batch[0])[-1], axis=-1)
    return self.mlp(self.embedding_layer(input_batch[0]))[..., tf.newaxis, :], mask
class AgentTemporalEncoder(tf.keras.layers.Layer):
  """Encodes agents temporal positions."""

  def __init__(self,output_shape, embedding_size, num_steps):
    super().__init__()
    #self.embedding_layer = SinusoidalEmbeddingLayer(
        #max_freq=num_steps,
        #hidden_size=embedding_size)
    self.embedding_layer = keras_nlp.layers.SinePositionEncoding(max_wavelength=10000)

    self.mlp = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=output_shape,
        bias_axes='h',
        activation=None)

  def _get_temporal_embedding(self, input_batch):
    # This weird thing is for exporting and loading keras model...
    b = tf.shape(input_batch[0])[0]
    num_steps = tf.shape(input_batch[0])[1]

    t = tf.range(0, num_steps, dtype=tf.float32)
    t = t[tf.newaxis, :]
    t = tf.tile(t, [b, 1])
    return self.embedding_layer(t[..., tf.newaxis])

  def call(self, input_dict):
    has_data = input_dict["has_data"]
    input_batch = input_dict["masked_inputs"]
    return (self.mlp(self._get_temporal_embedding(input_batch))[..., tf.newaxis, :],
            tf.ones_like(has_data))
  

  """ Adapted Agent Keypoint Encoding Layer from source: https://github.com/google-research/human-scene-transformer/blob/main/human_scene_transformer/model/agent_feature_encoder.py    """
class AgentKeypointsEncoder(tf.keras.layers.Layer):
  """Encodes the agent's keypoints."""

  def __init__(self, output_shape, embedding_size):
    super().__init__()

    self.mlp1 = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=output_shape,
        bias_axes='h',
        activation=tf.nn.relu)

  """def call(self, input_batch, training=None):

    keypoints = input_batch[1]

    out = self.mlp1(keypoints)

    return out"""
  def call(self, input_dict, training=None):
    is_hidden = input_dict["is_hidden"]
    has_data = input_dict["has_data"]
    input_batch = input_dict["masked_inputs"]
    not_is_hidden = tf.logical_not(is_hidden)
    mask = tf.logical_and(has_data, not_is_hidden)

    keypoints = input_batch[1]

    out = self.mlp1(keypoints)[..., tf.newaxis, :]

    return out, mask

class FeatureConcatAgentEncoderLayer(tf.keras.layers.Layer):
  

  """Independently encodes features and attends to them.

  Agent features are cross-attended with a learned query or hidden_vecs instead
  of MLP.
  """

  def __init__(self, input_length, batch_size=32, hidden_size=128, num_heads=4, ln_eps=1e-6, transformer_ff_dim=128, drop_prob=0.1):
    super().__init__()

    # Cross Attention and learned query.
    self.ff_layer2 = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=hidden_size,
        bias_axes='h',
        activation=None,
    )
    self.ff_dropout = tf.keras.layers.Dropout(drop_prob)

    self.agent_feature_embedding_layers = []
    # Position Feature [batch, sequence_len, feature_size, hidden_size]
    self.agent_feature_embedding_layers.append(
        AgentPositionEncoder(output_shape=hidden_size-8, embedding_size=hidden_size))
    # Feature Embedding - keypoints [batch, sequence_len, hidden_size]
    self.agent_feature_embedding_layers.append(
        AgentKeypointsEncoder(output_shape=hidden_size-8, embedding_size=hidden_size))

    # Temporal Embedding [batch, sequence_len, 1, hidden_size]
    self.agent_feature_embedding_layers.append(
        AgentTemporalEncoder(output_shape=hidden_size-8, embedding_size=hidden_size, num_steps=input_length))


  def call(self, input_dict, training = None):
    is_hidden = input_dict["is_hidden"]
    has_data = input_dict["has_data"]
    input_batch = input_dict["masked_inputs"]
    layer_embeddings = []
    for layer in self.agent_feature_embedding_layers:
      layer_embedding, _ = layer(input_dict, training=training)
      layer_embedding = layer_embedding[...,0,:]
      #print(layer_embedding)
      layer_embeddings.append(layer_embedding)
    embedding = tf.concat(layer_embeddings, axis=-1)
    #print("embedding", embedding)

    out = self.ff_layer2(embedding)

    return out
  
class AgentSelfAlignmentLayer(tf.keras.layers.Layer):
  """Enables agent to become aware of its temporal identity.

  Agent features are cross-attended with a learned query in temporal dimension.
  """

  def __init__(self,
               num_heads=8,
               hidden_size=128,
               ln_eps=1e-6,
               ff_dim=128):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_heads=num_heads
    if hidden_size % num_heads != 0:
      raise ValueError(f'hidden_size ({hidden_size}) must be an integer '
                       f'times bigger than num_heads ({num_heads}).')
    self.attn_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=ff_dim // num_heads,
        #value_dim = ff_dim // num_heads,
        attention_axes=1)
    self.attn_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)
    self.ff_layer1 = tf.keras.layers.EinsumDense(
        '...h,hf->...f', output_shape=ff_dim, bias_axes='f', activation='relu')
    self.ff_layer2 = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=hidden_size,
        bias_axes='h',
        activation=None)
    self.ff_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)

    # [1, 1, h]
    self.learned_query_vec = tf.Variable(
        tf.random_uniform_initializer(
            minval=-1., maxval=1.)(shape=[1, 1, hidden_size]),
        trainable=True,
        dtype=tf.float32)

  def build_learned_query(self, input_dict):
    """Converts self.learned_query_vec into a learned query vector."""
    # This weird thing is for exporting and loading keras model...
    b = tf.shape(input_dict['hidden_vecs'])[0]
    t = tf.shape(input_dict['hidden_vecs'])[1]

    # [b, t, 1, h]
    return tf.tile(self.learned_query_vec, [b, t, 1])

  def call(self, input_dict):
    # [b, t, h]
    hidden_vecs = input_dict['hidden_vecs']

    # Expand the attention mask with new dims so that Keras can broadcast to
    # the same shape as the attn_score: [b, num_heads, a, t, a, t].
    # attn_mask shape: [b, 1, 1, 1 t,]
    # True means the position participate in the attention while all
    # False positions are ignored.
    
    #print("input batch shape: ", hidden_vecs.shape)
    
    has_historic_data = tf.logical_and(
          input_dict['has_historic_data'][..., 0],
          tf.logical_not(input_dict['is_hidden'][..., 0]))
    attn_mask = has_historic_data[:, tf.newaxis, tf.newaxis, :]
    # [b, t, 1, h]
    learned_query = self.build_learned_query(input_dict)
    attn_out, attn_score = self.attn_layer(
        query=learned_query,
        key=hidden_vecs,
        value=hidden_vecs,
        attention_mask=attn_mask,
        return_attention_scores=True)

    attn_out = self.attn_ln(attn_out + hidden_vecs)

    # Feed-forward layers.
    out = self.ff_layer1(attn_out)
    out = self.ff_layer2(out)
    out = self.ff_ln(out + attn_out)


    return out, attn_score
class SelfAttnTransformerLayer(tf.keras.layers.Layer):
  """Performs full self-attention across the agent and time dimensions."""

  def __init__(
      self,
      num_heads=8,
      hidden_size=128,
      drop_prob=0.1,
      ln_eps=1e-6,
      ff_dim=128,
      mask=False,
      flatten=False,
      multimodality_induced=False,
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.mask = mask
    self.flatten = flatten
    self.multimodality_induced = multimodality_induced
    if hidden_size % num_heads != 0:
      raise ValueError(
          f'hidden_size ({hidden_size}) must be an integer '
          f'times bigger than num_heads ({num_heads}).'
      )
    self.attn_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=hidden_size // num_heads,
        attention_axes=1,
    )  # Full Attention time
    self.attn_dropout = tf.keras.layers.Dropout(drop_prob)
    self.attn_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)
    self.ff_layer1 = tf.keras.layers.EinsumDense(
        '...h,hf->...f', output_shape=ff_dim, bias_axes='f', activation='relu'
    )
    self.ff_layer2 = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=hidden_size,
        bias_axes='h',
        activation=None,
    )
    self.ff_dropout = tf.keras.layers.Dropout(drop_prob)
    self.ff_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)

  def call(self, input_dict, training=None):
    # [b, t, h] or [b, t, n, h]
    hidden_vecs = input_dict["hidden_vecs"]
    #print("hidden_vecs", hidden_vecs.shape)

    if self.flatten:
      h_shape = tf.shape(hidden_vecs)
      b = h_shape[0]
      t = h_shape[1]
      h = h_shape[-1]

      if self.multimodality_induced:
        n = h_shape[2]
        hidden_vecs = tf.reshape(hidden_vecs, (b, -1, n, h))
      else:
        hidden_vecs = tf.reshape(hidden_vecs, (b, -1, h))

    # Expand the attention mask with new dims so that Keras can broadcast to
    # the same shape as the attn_score: [b, num_heads, a, t, a, t].
    # attn_mask shape: [b, 1, 1, 1, a, t,]
    # True means the position participate in the attention while all
    # False positions are ignored.
    if not self.mask:
      attn_mask = None
    else:
      #print("has_historic_data", input_dict['has_historic_data'][..., 0].shape)
      has_historic_data = input_dict['has_historic_data'][..., 0]
      attn_mask = has_historic_data[:, tf.newaxis, tf.newaxis, :]
      #print("attn_mask", attn_mask.shape)

    if attn_mask is not None and self.flatten:
      t = h_shape[1]

      if self.multimodality_induced:  # We have modes
        n = h_shape[2]
        attn_mask_with_modes = attn_mask[..., tf.newaxis, :, :]
        tiled_mask = tf.tile(attn_mask_with_modes, [1, 1, 1, 1, t])
        attn_mask = tf.reshape(
            tiled_mask,
            [b, 1, 1, tf.cast(t*n, tf.int32), tf.cast(t*n, tf.int32)]
        )
      else:
        tiled_mask = tf.tile(attn_mask, [1, 1, t, 1])
        attn_mask = tf.reshape(
            tiled_mask,
            [b, 1, tf.cast(t, tf.int32), tf.cast(t, tf.int32)]
        )


    attn_out, attn_score = self.attn_layer(
        query=hidden_vecs,
        key=hidden_vecs,
        value=hidden_vecs,
        attention_mask=attn_mask,
        return_attention_scores=True)
    out = self.attn_dropout(attn_out, training=training)
    attn_out = self.attn_ln(out + hidden_vecs)

    # Feed-forward layers.
    out = self.ff_layer1(attn_out)
    out = self.ff_layer2(out)
    out = self.ff_dropout(out, training=training)
    out = self.ff_ln(out + attn_out)

    if self.flatten:
      out = tf.reshape(out, h_shape)

    return out, attn_score

class SelfAttnModeTransformerLayer(tf.keras.layers.Layer):
  """Performs full self-attention across the future modes dimensions."""

  def __init__(self,
               num_heads=8,
               hidden_size=128,
               drop_prob=0.1,
               ln_eps=1e-6,
               ff_dim=128):
    super().__init__()
    self.hidden_size = hidden_size
    if hidden_size % num_heads != 0:
      raise ValueError(f'hidden_size ({hidden_size}) must be an integer '
                       f'times bigger than num_heads ({num_heads}).')
    self.attn_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=hidden_size // num_heads,
        attention_axes=2)  # Attention over modes
    self.attn_dropout = tf.keras.layers.Dropout(drop_prob)
    self.attn_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)
    self.ff_layer1 = tf.keras.layers.EinsumDense(
        '...h,hf->...f', output_shape=ff_dim, bias_axes='f', activation='relu')
    self.ff_layer2 = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=hidden_size,
        bias_axes='h',
        activation=None)
    self.ff_dropout = tf.keras.layers.Dropout(drop_prob)
    self.ff_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)

  def call(self, input_dict, training=None):

    # [b, t, n, h]
    hidden_vecs = input_dict["hidden_vecs"]
    #print("hidden_vecs 3", hidden_vecs.shape)

    attn_out, attn_score = self.attn_layer(
        query=hidden_vecs,
        key=hidden_vecs,
        value=hidden_vecs,
        attention_mask=None,
        return_attention_scores=True)
    out = self.attn_dropout(attn_out, training=training)
    attn_out = self.attn_ln(out + hidden_vecs)

    # Feed-forward layers.
    out = self.ff_layer1(attn_out)
    out = self.ff_layer2(out)
    out = self.ff_dropout(out, training=training)
    out = self.ff_ln(out + attn_out)

    return out, attn_score

class MultimodalityInduction(tf.keras.layers.Layer):
  """Enables the model to forward and predict multi-mode predictions.

  1) Features are broadcasted to number of modes and summed with learned mode
      tensors.
  2) Mixture Weights are generated by cross-attention over all dimensions
      between learned mode tensors and hidden tensors.
  """

  def __init__(self,
               num_modes=5,
               num_heads=8,
               hidden_size=128,
               drop_prob=0.1,
               ln_eps=1e-6,
               ff_dim=128):
    super().__init__()
    self.num_modes = num_modes
    self.hidden_size = hidden_size
    if hidden_size % num_heads != 0:
      raise ValueError(f'hidden_size ({hidden_size}) must be an integer '
                       f'times bigger than num_heads ({num_heads}).')
    self.mm_attn_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=hidden_size // num_heads,
        attention_axes=2)
    self.mm_attn_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)
    self.mm_ff_layer1 = tf.keras.layers.EinsumDense(
        '...h,hf->...f', output_shape=ff_dim, bias_axes='f', activation='relu')
    self.mm_ff_layer2 = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=hidden_size,
        bias_axes='h',
        activation=None)
    self.mm_ff_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)

    self.mw_attn_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=hidden_size // num_heads,
        attention_axes=None)
    self.mw_attn_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)
    self.mw_ff_layer1 = tf.keras.layers.EinsumDense(
        '...h,hf->...f', output_shape=ff_dim, bias_axes='f', activation='relu')
    self.mw_ff_layer2 = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=1,  # Single logit per mode
        bias_axes='h',
        activation=None)
    self.mw_ff_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)

    self.attn_dropout = tf.keras.layers.Dropout(drop_prob)
    self.ff_dropout = tf.keras.layers.Dropout(drop_prob)

    # [1, 1, ,m,h]
    self.learned_add_mm = tf.Variable(
        tf.random_uniform_initializer(
            minval=-1.,
            maxval=1.)(shape=[1, 1, self.num_modes, hidden_size]),
        trainable=True,
        dtype=tf.float32)

  def call(self, input_dict, training=None):
    hidden_vecs = input_dict["hidden_vecs"]
    # [b, t, 1, h]
    hidden_vecs = hidden_vecs[..., tf.newaxis, :]

    # Multi Modes
    mm_add = self.mm_attn_ln(self.learned_add_mm + hidden_vecs)

    # Feed-forward layers.
    out = self.mm_ff_layer1(mm_add)
    out = self.mm_ff_layer2(out)
    out = self.ff_dropout(out)
    out = self.mm_ff_ln(out + hidden_vecs)

    # Mixture Weights
    # [b, 1, n, h]
    b = tf.shape(out)[0]
    attn_out_mw = self.mw_attn_layer(
        query=tf.tile(self.learned_add_mm, [b, 1, 1, 1]),
        key=mm_add,
        value=mm_add,
        return_attention_scores=False)
    attn_out_mw = self.attn_dropout(attn_out_mw, training=training)

    # [b, 1, n, h]
    attn_out_mw = self.mw_attn_ln(attn_out_mw)

    # Feed-forward layers.
    out_mw = self.mw_ff_layer1(attn_out_mw)
    out_mw = self.mw_ff_layer2(out_mw)
    out_mw = self.ff_dropout(out_mw, training=training)

    # [b, 1, n]
    mixture_logits = out_mw[..., 0]
    return out, mixture_logits

# Adapted from 2D Predictions to 3D predictions
class Prediction2DPositionHeadLayer(tf.keras.layers.Layer):
  """Converts transformer hidden vectors to model predictions."""

  def __init__(self, hidden_units=None, num_stages=5):
    super().__init__()

    self.dense_layers = []
    # Add hidden layers.
    if hidden_units is not None:
      for units in hidden_units:
        self.dense_layers.append(
            tf.keras.layers.Dense(units, activation='relu'))
    # Add the final prediction head.
    self.dense_layers.append(
        tf.keras.layers.EinsumDense(
            '...h,hf->...f',
            output_shape=9,
            bias_axes='f',
            activation=None))

  def call(self, input_dict):
    # [b, t, n,  h]
    hidden_vecs = input_dict["hidden_vecs"]

    x = hidden_vecs
    # [b, t, n, 5]
    for layer in self.dense_layers:
      x = layer(x)
    pred = x
    predictions = {
        'position': pred[..., 0:3],
        'position_raw_scale': pred[..., 3:],
        'mixture_logits': input_dict['mixture_logits']
    }
    return predictions
