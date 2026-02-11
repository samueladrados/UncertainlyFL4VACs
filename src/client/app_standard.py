import hydra
from omegaconf import DictConfig
import os
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras import mixed_precision

from src.models.architecture import CLR_BNN
from src.training.trainer import CLR_BNN_Trainer
from src.data.loader import DataLoaderGenerator
from src.utils.logger import FederatedLogger

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# GPU Config for the Worker
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def run_client(cfg: DictConfig):
    
    client_id = cfg.client.id
    round_num = cfg.federated.get("round", 0) 
    manifest_path = cfg.client.manifest_path
    weights_dir = cfg.strategy.standard_fedavg.weights_dir
    logs_dir = cfg.federated.logs_dir
    data_root = cfg.data.root
    num_classes = cfg.data.num_classes
    num_anchors = cfg.data.num_anchors
    batch_size = cfg.client.batch_size
    epochs = cfg.client.epochs
    learning_rate = cfg.client.learning_rate
    
    print(f"🚀 [Worker] Starting Client {client_id} for Round {round_num}")
    
    # 1. Read the Manifest (JSON) containing specific files for this client
    print(f"    📖 Reading manifest: {manifest_path}")
    with open(manifest_path, 'r') as f:
        my_files = json.load(f)
    # Safety Check
    if 'train' not in my_files or 'val' not in my_files:
        raise ValueError("El JSON del cliente no tiene claves 'train' o 'val'")
    
    # 2. Build Local Model
    local_model = CLR_BNN(num_classes=num_classes, num_anchors=num_anchors)
    # Build graph with dummy input
    dummy = [tf.zeros((1, 320, 320, 3)), tf.zeros((1, 320, 320, 2)), tf.zeros((1, 320, 320, 2))]
    local_model(dummy)
    
    # 3. Load Global Weights
    global_weights_path = os.path.join(weights_dir, "global_weights.weights.h5")
    print(f"    📥 Loading Global Weights: {global_weights_path}")
    local_model.load_weights(global_weights_path)
    
    # 4. Data Generators
    # Change: We use 'data_root' from args and 'specific_files' from JSON
    print(f"    📊 Loading Data from: {data_root}")
    
    train_gen = DataLoaderGenerator(
        data_root=data_root, 
        specific_files=my_files['train'], 
        batch_size=batch_size
    )
    
    val_gen = DataLoaderGenerator(
        data_root=data_root, 
        specific_files=my_files['val'], 
        batch_size=batch_size
    )
    
    print(f"    📊 Training on {len(my_files['train'])} files (Batch Size: {batch_size})")

    # 5. Initialize Trainer
    trainer = CLR_BNN_Trainer(
        local_model, 
        num_classes=num_classes, 
        num_anchors=num_anchors,
        initial_lr=learning_rate, 
        current_round=round_num,
        steps_per_epoch=len(train_gen),
        train_dataset_size=len(my_files['train'])
    )
    
    logger = FederatedLogger(
        client_id=f"client_{client_id}",
        round_num=round_num,
        log_dir=logs_dir
    )
    
    # 6. Train (Local Epochs)
    trainer.fit(train_gen, None, epochs=epochs, callbacks=[logger]) 
    
    # 7. Save to Disk
    if not os.path.exists(weights_dir): os.makedirs(weights_dir)
    save_path = os.path.join(weights_dir, f"client_{client_id}_round_{round_num}.weights.h5")
    local_model.save_weights(save_path)
    print(f"    💾 Saved local weights to: {save_path}")
    
if __name__ == "__main__":
    run_client()