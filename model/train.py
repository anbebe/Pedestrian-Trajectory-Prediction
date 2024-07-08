import tensorflow as tf
from preprocess_data import load_data
from layers import *


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

def train_model(data_path):
    # prepare train dataset
    batch_size = 32
    train_dataset, test_dataset = load_data(data_path="../final_dataset.pkl", batch_size=batch_size)
    


def test_model():
    batch_size = 32
    train_dataset, test_dataset = load_data(data_path="/home/pbr-student/personal/thesis/test/PedestrianTrajectoryPrediction/final_dataset.pkl", batch_size=batch_size)
    
    model = build_model()

    for (batch_x1, batch_x2) in train_dataset.take(1):
        output2 = model.predict((batch_x1, batch_x2))
        break

if __name__ == "__main__":
    test_model()