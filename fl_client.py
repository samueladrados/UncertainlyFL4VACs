import argparse
import os
import tensorflow as tf
import numpy as np
from clr_bnn import CLR_BNN
from train_clr import CLR_BNN_Trainer
from dataloader import DataLoaderGenerator
import json
from tensorflow.keras import mixed_precision
from fl_logger import FederatedLogger

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# --- CONFIGURATION ---
BATCH_SIZE = 10
NUM_CLASSES = 14
NUM_ANCHORS = 9
WEIGHTS_DIR = "./fl_weights_temp"

# GPU Config for the Worker
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

def run_client(client_id, round_num, global_weights_path, manifest_path, data_root):
    print(f"🚀 [Worker] Starting Client {client_id} for Round {round_num}")
    
    # 1. Read the Manifest (JSON) containing specific files for this client
    print(f"    📖 Reading manifest: {manifest_path}")
    with open(manifest_path, 'r') as f:
        my_files = json.load(f)
    # Safety Check
    if 'train' not in my_files or 'val' not in my_files:
        raise ValueError("El JSON del cliente no tiene claves 'train' o 'val'")
    
    # 2. Build Local Model
    local_model = CLR_BNN(num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS)
    # Build graph with dummy input
    dummy = [tf.zeros((1, 320, 320, 3)), tf.zeros((1, 320, 320, 2)), tf.zeros((1, 320, 320, 2))]
    local_model(dummy)
    
    # 3. Load Global Weights
    print(f"    📥 Loading Global Weights: {global_weights_path}")
    local_model.load_weights(global_weights_path)
    
    # 4. Data Generators
    # Change: We use 'data_root' from args and 'specific_files' from JSON
    print(f"    📊 Loading Data from: {data_root}")
    
    train_gen = DataLoaderGenerator(
        data_root=data_root, 
        specific_files=my_files['train'], 
        batch_size=BATCH_SIZE
    )
    
    val_gen = DataLoaderGenerator(
        data_root=data_root, 
        specific_files=my_files['val'], 
        batch_size=BATCH_SIZE
    )
    
    print(f"    📊 Training on {len(my_files['train'])} files (Batch Size: {BATCH_SIZE})")

    # 5. Initialize Trainer
    trainer = CLR_BNN_Trainer(
        local_model, 
        num_classes=NUM_CLASSES, 
        num_anchors=NUM_ANCHORS,
        initial_lr=0.0001, 
        current_round=round_num,
        steps_per_epoch=len(train_gen),
        train_dataset_size=len(my_files['train'])
    )
    
    logger = FederatedLogger(
        client_id=f"client_{client_id}",
        round_num=round_num,
        log_dir="fl_results_logs"
    )
    
    # 6. Train (Local Epochs)
    trainer.fit(train_gen, None, epochs=5, callbacks=[logger]) 
    
    # 7. Save to Disk
    if not os.path.exists(WEIGHTS_DIR): os.makedirs(WEIGHTS_DIR)
    save_path = os.path.join(WEIGHTS_DIR, f"client_{client_id}_round_{round_num}.weights.h5")
    local_model.save_weights(save_path)
    print(f"    💾 Saved local weights to: {save_path}")

if __name__ == "__main__":
    # Parse arguments sent by the Master script
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--global_path", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()
    
    try:
        run_client(args.client, args.round, args.global_path, args.manifest, args.data_root)
    except Exception as e:
        print(f"❌ [Worker Error] Client {args.client} failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1) # Return error code 1 to master so it knows this client crashed