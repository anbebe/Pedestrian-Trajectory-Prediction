import os
import datetime
import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_probability as tfp
from preprocess_data import load_data
import logging
from layers import *
from metrics import *
from losses import *

logging.getLogger().setLevel(logging.INFO)

class HST(tf.keras.Model):
    def __init__(self, input_length):
        super().__init__()

        hidden_size = 128
        self.preprocess_layer = PreprocessLayer() 
    
        self.agent_encoder = FeatureAttnAgentEncoderLearnedLayer(input_length=input_length)
        self.align_layer = AgentSelfAlignmentLayer()
        self.tranformer1 = SelfAttnTransformerLayer(mask=True)
        self.tranformer2 = SelfAttnTransformerLayer(mask=True)
        self.multimodality_induction = MultimodalityInduction()
        self.tranformer3= SelfAttnTransformerLayer(mask=True, multimodality_induced=True)
        self.tranformer4 = SelfAttnModeTransformerLayer()
        self.tranformer5 = SelfAttnTransformerLayer(mask=True, multimodality_induced=True)
        self.tranformer6= SelfAttnModeTransformerLayer()
        self.prediction_layer = Prediction2DPositionHeadLayer()

    def call(self, input_batch, training = False):
        (input_1, input_2) = input_batch
        masked_inputs, mask, targets = self.preprocess_layer((input_1, input_2)) # output shape (batch_size, 15, 3)
  
        encoded_agent, _ = self.agent_encoder((masked_inputs, mask))
        self_encoded_agent, _ = self.align_layer((encoded_agent, mask))
        transformed1, _ = self.tranformer1((self_encoded_agent, mask))
        transformed2, _ = self.tranformer2((transformed1, mask))
        transformed3, logits = self.multimodality_induction((transformed2, mask))
        transformed4, _ = self.tranformer3((transformed3, mask))
        transformed5, _ = self.tranformer4(transformed4)
        transformed6, _ = self.tranformer5((transformed5, mask))
        transformed7, _ = self.tranformer6(transformed6)
        pred = self.prediction_layer(transformed7)

        output_dict = {
        'mask': mask,
        'position': pred[...,0:3],
        'position_raw_scale': pred[...,3:],
        'mixture_logits': logits,
        'targets': targets
        }

        return output_dict


def build_model():
    input_length=15
    input_dim_1 = 3
    input_dim_2 = 51

    
    # Define a simple model that includes the custom layer
    input_1 = tf.keras.Input(shape=(input_length, input_dim_1))
    input_2 = tf.keras.Input(shape=(input_length, input_dim_2))

    hidden_size = 128
    masked_inputs, mask, targets = PreprocessLayer()((input_1, input_2)) # output shape (batch_size, 15, 3)
  
    encoded_agent, _ = FeatureAttnAgentEncoderLearnedLayer(input_length=input_length)((masked_inputs, mask))
    self_encoded_agent, _ = AgentSelfAlignmentLayer()((encoded_agent, mask))
    transformed1, _ = SelfAttnTransformerLayer(mask=True)((self_encoded_agent, mask))
    transformed2, _ = SelfAttnTransformerLayer(mask=True)((transformed1, mask))
    transformed3, logits = MultimodalityInduction()((transformed2, mask))
    transformed4, _ = SelfAttnTransformerLayer(mask=True, multimodality_induced=True)((transformed3, mask))
    transformed5, _ = SelfAttnModeTransformerLayer()(transformed4)
    transformed6, _ = SelfAttnTransformerLayer(mask=True, multimodality_induced=True)((transformed5, mask))
    transformed7, _ = SelfAttnModeTransformerLayer()(transformed6)
    pred = Prediction2DPositionHeadLayer()(transformed7)
    
    
    output_dict = {
        'mask': mask,
        'position': pred[...,0:3],
        'position_raw_scale': pred[...,3:],
        'mixture_logits': logits,
        'targets': targets
    }


    model = tf.keras.Model(inputs=(input_1, input_2), outputs=output_dict)

    return model

# Define Training and Eval tf.function.
@tf.function
def train_step(iterator, model, loss_obj, strategy, optimizer, train_metrics, batches_per_train_step):
    """Training function."""

    def step_fn(input_batch):
        #print("input batch ", input_batch[0].shape)
        with tf.GradientTape() as tape:
            #print("predict")
            predictions = model(input_batch, training=True)
            #print("loss")
            loss_dict = loss_obj(input_batch, predictions)
            loss = (loss_dict['loss']
                / tf.cast(strategy.num_replicas_in_sync, tf.float32))
        #print("grad")
        grads = tape.gradient(loss, model.trainable_variables)
        #print("optimizer")
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # Update the training metrics.
        # These need special treatments as they are standard keras metrics.
        #print("train metrics")
        train_metrics['loss'].update_state(loss_dict['loss'])
        #print("train metrics2")
        train_metrics['loss_position'].update_state(loss_dict['position_loss'])
        # Our own metrics.
        for key in train_metrics:
            if key in {'loss', 'loss_position', 'loss_orientation'}:
                continue
            #print("key ", key)
            train_metrics[key].update_state(input_batch, predictions)
        #print("fertig")

    for _ in tf.range(tf.constant(batches_per_train_step)):
        strategy.run(
            step_fn,
            args=(next(iterator),),
            options=tf.distribute.RunOptions(
                experimental_enable_dynamic_batch_size=False))

@tf.function
def eval_step(iterator, model, loss_obj, strategy, eval_metrics, batches_per_eval_step):

    def step_fn(input_batch):
        predictions = model(input_batch, training=False)
        loss_dict = loss_obj(input_batch, predictions)
        # Update the eval metrics.
        # These need special treatments as they are standard keras metrics.
        eval_metrics['loss'].update_state(loss_dict['loss'])
        eval_metrics['loss_position'].update_state(loss_dict['position_loss'])
        # Our own metrics.
        for key in eval_metrics:
            if key in {'loss', 'loss_position', 'loss_orientation'}:
                continue
            eval_metrics[key].update_state(input_batch, predictions)

    for _ in tf.range(tf.constant(batches_per_eval_step)):
        strategy.run(
            step_fn,
            args=(next(iterator),),
            options=tf.distribute.RunOptions(
                experimental_enable_dynamic_batch_size=False))

def _get_learning_rate_schedule(
    warmup_steps: int,
    total_steps: int,
    learning_rate: float,
    alpha: float = 0.0) -> tf.keras.optimizers.schedules.LearningRateSchedule:
  """Returns a cosine decay learning rate schedule to be used in training.

  Args:
    warmup_steps: Number of training steps to apply warmup. If global_step <
      warmup_steps, the learning rate will be `global_step/num_warmup_steps *
      init_lr`.
    total_steps: The total number of training steps.
    learning_rate: The peak learning rate anytime during the training.
    alpha: The alpha parameter forwarded to CosineDecay

  Returns:
    A CosineDecay learning schedule w/ warmup.
  """

  decay_schedule = tf.keras.optimizers.schedules.CosineDecay(
      initial_learning_rate=learning_rate, decay_steps=total_steps, alpha=alpha)
  return tfm.optimization.LinearWarmup(decay_schedule, warmup_steps, 1e-10)

def train_model():
    # prepare train dataset
    batch_size = 32
    train_dataset, test_dataset = load_data(data_path="/home/pbr-student/personal/thesis/test/PedestrianTrajectoryPrediction/final_dataset.pkl", batch_size=batch_size)
    print("loaded dataset")
    model_base_dir = ""
    dt_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    model_dir = os.path.join(model_base_dir, dt_str)
    os.makedirs(model_dir)
    ckpt_dir = os.path.join(model_dir, 'ckpts')
    os.makedirs(ckpt_dir)
    ckpt_best_dir = os.path.join(model_dir, 'ckpts_best')
    os.makedirs(ckpt_best_dir)
    checkpoint_prefix = os.path.join(ckpt_dir, 'ckpt')
    checkpoint_prefix_best = os.path.join(ckpt_best_dir, 'ckpt')
    tensorboard_dir = '/tmp/tensorboard'


    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(tensorboard_dir, 'train'))
    eval_summary_writer = tf.summary.create_file_writer(
        os.path.join(tensorboard_dir, 'eval'))


    batches_per_train_step=10 #25000
    batches_per_eval_step =1 # 2000
    eval_every_n_step = 5 #1e4

    strategy = tf.distribute.OneDeviceStrategy('cpu')

    dist_train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    dist_eval_dataset = strategy.experimental_distribute_dataset(test_dataset)

    learning_rate_schedule = _get_learning_rate_schedule(
        warmup_steps=5e4, total_steps=5e6,
        learning_rate=1e-4)

    current_global_step = 0

    with strategy.scope():
        model = HST(15)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate_schedule,
            global_clipnorm=1.)
        loss_obj = MultimodalPositionNLLLoss()#MinNLLPositionMixtureCategoricalCrossentropyLoss()
        train_metrics = {
        'loss': Mean(),
        'loss_position': Mean(),
        'min_ade': MinADE(),
        'ml_ade': MLADE(),
        'pos_nll': PositionNegativeLogLikelihood()
        }

        eval_metrics = {
        'loss': Mean(),
        'loss_position': Mean(),
        'min_ade': MinADE(),
        'ml_ade': MLADE(),
        'pos_nll': PositionNegativeLogLikelihood()
        }

    best_eval_loss = tf.Variable(tf.float32.max)
    checkpoint = tf.train.Checkpoint(model=model,
                                    optimizer=optimizer,
                                    best_eval_loss=best_eval_loss)
    checkpoint_best = tf.train.Checkpoint(model=model)
    best_checkpoint_manager = tf.train.CheckpointManager(checkpoint_best,
                                                        checkpoint_prefix_best,
                                                        max_to_keep=1)
    
     # 5) Actual Training Loop
    train_iter = iter(dist_train_dataset)
    eval_iter = iter(dist_eval_dataset)
    total_train_steps = 60 # 1e6
    num_train_iter = (
        total_train_steps // batches_per_train_step)
    current_train_iter = (
        current_global_step // batches_per_train_step)

    #print("range: ",range(current_train_iter, num_train_iter))

    logging.info('Beginning training.')
    for step in range(current_train_iter, num_train_iter):
        #print("step: ", step)
        # Actual number of SGD steps.
        actual_steps = step * batches_per_train_step
        with train_summary_writer.as_default():
            # Run training SGD over train_param.batches_per_train_step batches.
            # optimizer.iterations = step * train_param.batches_per_train_step.
            #print("train step")
            train_step(train_iter, model, loss_obj, strategy, optimizer, train_metrics, batches_per_train_step)
            #print("out")
            # Writing metrics to tensorboard.
            if step % 1 == 0:
                for key in train_metrics:
                    tf.summary.scalar(
                        key, train_metrics[key].result(), step=optimizer.iterations)

                if isinstance(optimizer, tf.keras.optimizers.experimental.Optimizer):
                    learning_rate = optimizer.learning_rate
                else:
                    learning_rate = optimizer.lr(optimizer.iterations)
                tf.summary.scalar(
                    'learning_rate',
                    learning_rate,
                    step=optimizer.iterations)
                logging.info('Training step %d', step)
                logging.info('Training loss: %.4f, ADE: %.4f',
                                train_metrics['loss'].result().numpy(),
                                train_metrics['min_ade'].result().numpy())
                # Reset metrics.
                for key in train_metrics:
                    train_metrics[key].reset_states()

        # Evaluation.
        if actual_steps % eval_every_n_step == 0:
            logging.info('Evaluating step %d over %d random eval samples', step,
                        batches_per_eval_step * batch_size)
            with eval_summary_writer.as_default():
                eval_step(eval_iter, model, loss_obj, strategy, eval_metrics, batches_per_eval_step)
                for key in eval_metrics:
                    tf.summary.scalar(
                        key, eval_metrics[key].result(), step=optimizer.iterations)
                logging.info('Eval loss: %.4f, ADE: %.4f',
                                eval_metrics['loss'].result().numpy(),
                                eval_metrics['min_ade'].result().numpy())

                if eval_metrics['loss'].result() < best_eval_loss:
                    best_eval_loss.assign(eval_metrics['loss'].result())
                    best_checkpoint_manager.save()

                # Reset metrics.
                for key in eval_metrics:
                    eval_metrics[key].reset_states()

                # Save model.
                checkpoint_name = checkpoint.save(checkpoint_prefix)
                logging.info('Saved checkpoint to %s', checkpoint_name)
    


def test_model():
    batch_size = 32
    train_dataset, test_dataset = load_data(data_path="/home/pbr-student/personal/thesis/test/PedestrianTrajectoryPrediction/final_dataset.pkl", batch_size=batch_size)
    
    model = build_model()

    for (batch_x1, batch_x2) in train_dataset.take(1):
        output2 = model.predict((batch_x1, batch_x2))
        break

if __name__ == "__main__":
    train_model()