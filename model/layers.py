import tensorflow as tf
import numpy as np
from typing import Tuple, Optional


class PreprocessLayer(tf.keras.layers.Layer):
    """ Applies the masking to the sequence
    """

    def calc_hidden_mask(self, batch_size=32, sequence_length=15):
        # create mask array, False = needs to be predicted
        mask_arrays = []
        print("batch_size:", batch_size)
        print("sequence_length:", sequence_length)
        for i in range(batch_size):
          mask_arr = [True] * 6 + [False] * (sequence_length-6)
          # hide 0-2 in between steps (for lazyness whole datapoint)
          hidden_nr = np.random.randint(3)
          hidden_idx = np.random.choice(range(6),hidden_nr, replace=False)
          for v in hidden_idx:
              mask_arr[v] = False
          mask_arrays.append(mask_arr)
        print("mask aaray:", np.asarray(mask_arrays).shape)
        return np.asarray(mask_arrays)

    def call(self,
           raw_input_batch: Tuple[tf.Tensor, tf.Tensor],
           is_hidden: Optional[tf.Tensor] = None) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        input_batch = raw_input_batch
  

        batch_size = tf.shape(input_batch[0])[0]
        sequence_length = tf.shape(input_batch[0])[1]
        feature_size1 = tf.shape(input_batch[0])[2]
        feature_size2 = tf.shape(input_batch[1])[2]

        mask = self.calc_hidden_mask() #tf.convert_to_tensor

        mask_tensor = tf.constant(mask, dtype=tf.bool)

        # Expand dimensions of mask to match the input tensor
        #expanded_mask = tf.expand_dims(mask_tensor, axis=0)  # Add batch dimension
        expanded_mask = tf.expand_dims(mask_tensor, axis=-1)  # Add feature dimension

        # Broadcast mask to match input tensor shape
        broadcasted_mask_pos = tf.broadcast_to(expanded_mask, (batch_size, sequence_length, feature_size1))
        broadcasted_mask_pose = tf.broadcast_to(expanded_mask, (batch_size, sequence_length, feature_size2))

        #batch_mask = tf.broadcast_to(expanded_mask, (batch_size, sequence_length))

        # Apply mask
        masked_input_pos = tf.where(broadcasted_mask_pos, input_batch[0], tf.zeros_like(input_batch[0]))
        masked_input_pose = tf.where(broadcasted_mask_pose, input_batch[1], tf.zeros_like(input_batch[1]))
        targets = tf.where(tf.math.logical_not(broadcasted_mask_pos), input_batch[0], tf.zeros_like(input_batch[0]))

        return (masked_input_pos, masked_input_pose), mask, targets

""" Adapted Sinusoidal Embedding Layer from source: https://github.com/google-research/human-scene-transformer/blob/main/human_scene_transformer/model/embedding.py    """
class SinusoidalEmbeddingLayer(tf.keras.layers.Layer):
  """Sinusoidal Postional Embedding for xyz and time."""

  def __init__(self, min_freq=4, max_freq=256, hidden_size=256):
    super().__init__()
    self.min_freq = float(min_freq)
    self.max_freq = float(max_freq)
    self.hidden_size = hidden_size
    if hidden_size % 2 != 0:
      raise ValueError('hidden_size ({hidden_size}) must be divisible by 2.')
    self.num_freqs_int32 = hidden_size // 2
    self.num_freqs = tf.cast(self.num_freqs_int32, dtype=tf.float32)

  def build(self, input_shape):
    log_freq_increment = (
        tf.math.log(float(self.max_freq) / float(self.min_freq)) /
        tf.maximum(1.0, self.num_freqs - 1))
    # [num_freqs]
    self.inv_freqs = self.min_freq * tf.exp(
        tf.range(self.num_freqs, dtype=tf.float32) * -log_freq_increment)

  def call(self, input_tensor):
    
    # [..., num_freqs]
    input_tensor = tf.expand_dims(input_tensor, -1)
    input_tensor = tf.tile(input_tensor, [1, 1, 1, self.num_freqs_int32])
    
    # Compute the sinusoidal embeddings
    sin_embeds = tf.sin(input_tensor * self.inv_freqs)
    cos_embeds = tf.cos(input_tensor * self.inv_freqs)
    
    # Concatenate along the last axis
    embedded = tf.concat([sin_embeds, cos_embeds], axis=-1)
    
    # Reshape to the desired output shape (batch_size, sequence_length, feature_size * hidden_size)
    batch_size = tf.shape(embedded)[0]
    sequence_length = tf.shape(embedded)[1]
    feature_size = tf.shape(embedded)[2]
    
    embedded = tf.reshape(embedded, (batch_size, sequence_length, feature_size * self.hidden_size))
    """
    # [..., num_freqs]
    input_tensor = tf.repeat(
        input_tensor[..., tf.newaxis], self.num_freqs_int32, axis=-1)
    # [..., h]
    embedded = tf.concat([
        tf.sin(input_tensor * self.inv_freqs),
        tf.cos(input_tensor * self.inv_freqs)
    ],
                         axis=-1)"""
    return embedded


""" Adapted Agent Position Encoding Layer from source: https://github.com/google-research/human-scene-transformer/blob/main/human_scene_transformer/model/agent_feature_encoder.py    """
class AgentPositionEncoder(tf.keras.layers.Layer):
  """Encodes agents spatial positions."""

  def __init__(self, output_shape, embedding_size):
    
    super().__init__()

    self.embedding_layer = SinusoidalEmbeddingLayer(
        hidden_size=embedding_size) # output_shape (batch_sie, sequence_length, feature size, hidden_size)

    self.mlp = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=output_shape,
        bias_axes='h',
        
        activation=None)

  def call(self, input_batch):
    return self.mlp(self.embedding_layer(input_batch[0])) 

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

  def call(self, input_batch, training=None):

    keypoints = input_batch[1]

    out = self.mlp1(keypoints)

    return out

class AgentTemporalEncoder(tf.keras.layers.Layer):
  """Encodes agents temporal positions."""

  def __init__(self,output_shape, embedding_size, num_steps):
    super().__init__()
    self.embedding_layer = SinusoidalEmbeddingLayer(
        max_freq=num_steps,
        hidden_size=embedding_size)

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

  def call(self, input_batch):
    return self.mlp(self._get_temporal_embedding(input_batch))

""" Adapted Feature Encoding Layer from source: https://github.com/google-research/human-scene-transformer/blob/main/human_scene_transformer/model/agent_encoder.py    """
class FeatureAttnAgentEncoderLearnedLayer(tf.keras.layers.Layer):
  """Independently encodes features and attends to them.

  Agent features are cross-attended with a learned query or hidden_vecs instead
  of MLP.
  """

  def __init__(self, input_length, batch_size=32, hidden_size=128, num_heads=4, ln_eps=1e-6, transformer_ff_dim=128, drop_prob=0.1):
    super(FeatureAttnAgentEncoderLearnedLayer, self).__init__()

    self.batch_size=batch_size
    self.input_length = input_length
    self.num_heads=num_heads

    # Cross Attention and learned query.
    self.attn_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=360//num_heads,  # "large" to prevent a bottleneck
        value_dim=360//num_heads)
    self.attn_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)
    self.ff_layer1 = tf.keras.layers.EinsumDense(
        '...h,hf->...f',
        output_shape=transformer_ff_dim,
        bias_axes='f',
        activation='relu')
    self.ff_layer2 = tf.keras.layers.EinsumDense(
        '...f,fh->...h',
        output_shape=hidden_size,
        bias_axes='h',
        activation=None)
    self.ff_dropout = tf.keras.layers.Dropout(drop_prob)
    self.ff_ln = tf.keras.layers.LayerNormalization(epsilon=ln_eps)

    self.agent_feature_embedding_layers = []
    # Position Feature
    self.agent_feature_embedding_layers.append(
        AgentPositionEncoder(output_shape=hidden_size-8, embedding_size=hidden_size))
    # Feature Embedding - keypoints
    self.agent_feature_embedding_layers.append(
        AgentKeypointsEncoder(output_shape=hidden_size-8, embedding_size=hidden_size))

    # Temporal Embedding
    self.agent_feature_embedding_layers.append(
        AgentTemporalEncoder(output_shape=hidden_size-8, embedding_size=hidden_size, num_steps=input_length))

    # [1, 1, h]
    self.learned_query_vec = tf.Variable(
        tf.random_uniform_initializer(
            minval=-1., maxval=1.)(shape=[1, 1, hidden_size]),
        trainable=True,
        dtype=tf.float32)

  def _build_learned_query(self, input_batch):
    """Converts self.learned_query_vec into a learned query vector."""
    # This weird thing is for exporting and loading keras model...
    b = tf.shape(input_batch[0])[0]
    num_steps = tf.shape(input_batch[0])[1]

    # [b, num_steps, h]
    return tf.tile(self.learned_query_vec, [b, num_steps, 1])

  def call(self, input_batch: Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor],
           training: Optional[bool] = None):
    mask = input_batch[1]
    input_batch = input_batch[0]
    layer_embeddings = []
    for layer in self.agent_feature_embedding_layers:
      layer_embedding = layer(input_batch, training=training)
      layer_embeddings.append(layer_embedding)
    embedding = tf.concat(layer_embeddings, axis=-1)

    b = tf.shape(embedding)[0]
    t = tf.shape(embedding)[1]
    n = tf.shape(embedding)[2]

    #print("embedding shape: ", embedding.shape)

    attention_mask = tf.where(mask, 1, 0)
    attention_mask = tf.cast(attention_mask, dtype=tf.float32)
    attention_mask = tf.expand_dims(attention_mask, axis=-1)
    batch_size = tf.shape(input_batch[0])[0]
    attention_mask = tf.broadcast_to(attention_mask, (batch_size, self.input_length, self.input_length))
    attention_mask = tf.expand_dims(attention_mask, axis=1)  # Shape: (batch_size, 1, seq_len, seq_len)
    attention_mask = tf.repeat(attention_mask, self.num_heads, axis=1)  # Shape: (batch_size, num_heads, seq_len, seq_len)
    
    learned_query = self._build_learned_query(input_batch)
    #print("learned_query shape: ", learned_query.shape)

    # Attention along axis 3
    attn_out, attn_score = self.attn_layer(
        query=learned_query,
        key=embedding,
        value=embedding,
        attention_mask=attention_mask,
        return_attention_scores=True)
    #print("attn_out shape: ", attn_out.shape)
    # [b, t, h]
    #attn_out = attn_out[..., 0, :]
    out = self.ff_layer1(attn_out)
    out = self.ff_layer2(out)
    out = self.ff_dropout(out, training=training)
    out = self.ff_ln(out + attn_out)

    return out, attn_score

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
        value_dim = ff_dim // num_heads,
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

  def build_learned_query(self, input_batch):
    """Converts self.learned_query_vec into a learned query vector."""
    # This weird thing is for exporting and loading keras model...
    b = tf.shape(input_batch)[0]
    t = tf.shape(input_batch)[1]

    # [b, t, 1, h]
    return tf.tile(self.learned_query_vec, [b, t, 1])

  def call(self, input_batch):
    # [b, t, h]
    hidden_vecs = input_batch[0]
    mask = input_batch[1]

    # Expand the attention mask with new dims so that Keras can broadcast to
    # the same shape as the attn_score: [b, num_heads, a, t, a, t].
    # attn_mask shape: [b, 1, 1, 1 t,]
    # True means the position participate in the attention while all
    # False positions are ignored.
    
    #print("input batch shape: ", hidden_vecs.shape)
    
    attention_mask = tf.where(mask, 1, 0)
    attention_mask = tf.cast(attention_mask, dtype=tf.float32)
    attention_mask = tf.expand_dims(attention_mask, axis=-1)
    batch_size = tf.shape(hidden_vecs)[0]
    input_length = tf.shape(hidden_vecs)[1]
    attention_mask = tf.broadcast_to(attention_mask, (batch_size, input_length, input_length))
    attention_mask = tf.expand_dims(attention_mask, axis=1)  # Shape: (batch_size, 1, seq_len, seq_len)
    attention_mask = tf.repeat(attention_mask, self.num_heads, axis=1)  # Shape: (batch_size, num_heads, seq_len, seq_len)

    # [b, t, 1, h]
    learned_query = self.build_learned_query(hidden_vecs)
    attn_out, attn_score = self.attn_layer(
        query=learned_query,
        key=hidden_vecs,
        value=hidden_vecs,
        attention_mask=attention_mask,
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

  def call(self, input_batch, training=None):
    # [b, t, h] or [b, t, n, h]
    hidden_vecs = input_batch[0]
    mask = input_batch[1]

    if self.flatten:
      h_shape = tf.shape(hidden_vecs)
      b = h_shape[0]
      t = h_shape[1]
      h = h_shape[-1]

      if self.multimodality_induced:
        n = h_shape[3]
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
      attention_mask = tf.where(mask, 1, 0)
      attention_mask = tf.cast(attention_mask, dtype=tf.float32)
      attention_mask = tf.expand_dims(attention_mask, axis=-1)
      batch_size = tf.shape(hidden_vecs)[0]
      input_length = tf.shape(hidden_vecs)[1]
      attention_mask = tf.broadcast_to(attention_mask, (batch_size, input_length, input_length))
      attn_mask = tf.expand_dims(attention_mask, axis=1)  # Shape: (batch_size, 1, seq_len, seq_len)

    if self.multimodality_induced:  # We have modes
      attn_mask = tf.expand_dims(attn_mask, axis=1)  # Shape: (batch_size, 1, 1, seq_len, seq_len)


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

  def call(self, input_batch, training=None):

    # [b, t, n, h]
    hidden_vecs = input_batch

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
        attention_axes=3)
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

    # [1, 1, m, h]
    self.learned_add_mm = tf.Variable(
        tf.random_uniform_initializer(
            minval=-1.,
            maxval=1.)(shape=[1, 1, self.num_modes, hidden_size]),
        trainable=True,
        dtype=tf.float32)

  def call(self, input_batch, training=None):
    mask = input_batch[1]
    # [b, t, 1, h]
    hidden_vecs = input_batch[0][..., tf.newaxis, :]

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

    # [b, 1, 1, n, h]
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

  def call(self, input_batch):
    # [b, t, n,  h]
    hidden_vecs = input_batch

    x = hidden_vecs
    # [b, t, n, 5]
    for layer in self.dense_layers:
      x = layer(x)
    pred = x
    """predictions = {
        'agents/position': pred[..., 0:3],
        'agents/position/raw_scale_tril': pred[..., 3:5],
    }"""
    return pred

