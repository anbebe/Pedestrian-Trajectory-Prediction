from preprocess_data import load_data

def train_model(data_path):
    # prepare train dataset
    batch_size = 32
    train_data, test_data, val_data = load_data(data_path="../final_dataset.pkl")
    
