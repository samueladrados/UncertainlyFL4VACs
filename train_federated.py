import os
import tensorflow as tf
import numpy as np
from dataloader import DataLoaderGenerator
from clr_bnn import CLR_BNN
from train_clr import CLR_BNN_Trainer 
from fl_utils import partition_data_dual_source
import gc 

# --- FIX MEMORY ---
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- FL CONFIGURATION ---
ROOT_TRAIN_VAL = "/mnt/c/Users/USUARIO/Desktop/enviroments/Paper_NOUS/datasets/nuscenes_preprocessed"
ROOT_TEST = "/mnt/c/Users/USUARIO/Desktop/enviroments/Paper_NOUS/datasets/nuscenes_test_preprocessed"

NUM_CLIENTS = 10
FL_ROUNDS = 5
LOCAL_EPOCHS = 3 
BATCH_SIZE = 6
LR = 1e-5
NUM_CLASSES = 14


# Directory for temporary weight storage (Disk Swap)
WEIGHTS_DIR = "./fl_weights_temp"
if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)


# --- ISOLATED FUNCTION (KEY FOR MEMORY RELEASE) ---
def train_single_client(client_id, round_num, files, global_weights_path):
    """
    Trains a single client and saves the weights to disk.
    Because this is a separate function, Python forces garbage collection 
    of all local variables (model, trainer, data) upon return.
    """
    print(f"\n   👤 Training client_{client_id} (Round {round_num+1})...")
    
    # 1. Clear Keras Session within this scope
    tf.keras.backend.clear_session()
    
    # 2. Build Local Model
    # Initialize dummy input to build the graph
    dummy = [tf.zeros((1, 320, 320, 3)), tf.zeros((1, 320, 320, 2)), tf.zeros((1, 320, 320, 2))]
    local_model = CLR_BNN(num_classes=NUM_CLASSES)
    local_model(dummy) 
    
    # 3. Load Global Weights
    local_model.load_weights(global_weights_path)
    
    # 4. Create Data Generator
    # IMPORTANT: Explicit batch_size
    train_gen = DataLoaderGenerator(ROOT_TRAIN_VAL, specific_files=files['train'], batch_size=BATCH_SIZE, split='train')
    
    # 5. Create Trainer
    trainer = CLR_BNN_Trainer(
        local_model, 
        num_classes=NUM_CLASSES,
        initial_lr=1e-4, 
        steps_per_epoch=len(train_gen),
        train_dataset_size=len(files['train']) # Used for KL weighting
    )
    
    # 6. Start Training
    try:
        trainer.fit(train_gen, None, epochs=LOCAL_EPOCHS)
        
        # 7. Save weights to disk and return path
        client_path = os.path.join(WEIGHTS_DIR, f"client_{client_id}_round_{round_num}.weights.h5")
        local_model.save_weights(client_path)
        print(f"      💾 Weights saved to {client_path}")
        return client_path
        
    except Exception as e:
        print(f"❌ Error in client {client_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def fed_avg(weight_files):
    """
    Averages weights by reading files from disk one by one to save VRAM.
    """
    print("☁️  [Server] Aggregating weights from disk...")
    
    # 1. Clear session to ensure clean slate
    tf.keras.backend.clear_session()
    
    # 2. Build a dummy model to get the architecture structure
    base_model = CLR_BNN(num_classes=NUM_CLASSES)
    # Initialize with dummy input to build layers
    dummy = [tf.zeros((1, 320, 320, 3)), tf.zeros((1, 320, 320, 2)), tf.zeros((1, 320, 320, 2))]
    base_model(dummy)
    
    # Weight accumulator (System RAM, Numpy array, NOT VRAM)
    avg_weights = None
    
    for f_path in weight_files:
        print(f"    Loading: {f_path}")
        # Load weights into the model
        base_model.load_weights(f_path)
        # Extract weights as numpy arrays
        current_weights = base_model.get_weights()
        
        if avg_weights is None:
            # First client initializes the sum
            avg_weights = [w.astype(np.float32) for w in current_weights]
        else:
            # Subsequent clients add to the sum
            for i, w in enumerate(current_weights):
                avg_weights[i] += w
                
        # Fast cleanup
        del current_weights
        gc.collect()

    # Average: Divide sum by number of clients
    n = len(weight_files)
    avg_weights = [w / n for w in avg_weights]
    
    # Clean up the base model used for aggregation
    del base_model
    tf.keras.backend.clear_session()
    
    return avg_weights

def main_fl():
    # --- 1. Prepare Data ---
    print("🔵 [FL] Starting Data Partitioning...")
    # Use the function from fl_utils
    client_files = partition_data_dual_source(ROOT_TRAIN_VAL, ROOT_TEST, NUM_CLIENTS)
    
    # --- 2. Initialize Global Model ---
    print("🔵 [FL] Initializing Global Weights...")
    tf.keras.backend.clear_session()
    global_model = CLR_BNN(num_classes=NUM_CLASSES)
    dummy = [tf.zeros((1, 320, 320, 3)), tf.zeros((1, 320, 320, 2)), tf.zeros((1, 320, 320, 2))]
    global_model(dummy)
    
    global_weights_path = os.path.join(WEIGHTS_DIR, "global_weights.weights.h5")
    global_model.save_weights(global_weights_path)
    
    del global_model
    tf.keras.backend.clear_session()
    gc.collect()

    # --- 3. MAIN LOOP ---
    for round_num in range(FL_ROUNDS):
        print(f"\n🌍 ================= ROUND {round_num+1}/{FL_ROUNDS} ================= 🌍")
        
        client_weight_files = []

        for client_id in range(NUM_CLIENTS):
            # A) Aggressive Pre-Cleanup
            gc.collect()
            tf.keras.backend.clear_session()
            
            # B) Call Isolated Function
            # This creates the model, trains it, saves it, and DESTROYS it upon return
            w_path = train_single_client(client_id, round_num, client_files[client_id], global_weights_path)
            
            if w_path:
                client_weight_files.append(w_path)
            
            # C) Aggressive Post-Cleanup
            # Ensures any residuals are cleared
            gc.collect() 
            tf.keras.backend.clear_session()
        
        # --- 4. AGGREGATION ---
        if len(client_weight_files) > 0:
            new_global_weights = fed_avg(client_weight_files)
            
            if new_global_weights is not None:
                print("💾 Saving new Global Model...")
                tf.keras.backend.clear_session()
                temp_model = CLR_BNN(num_classes=NUM_CLASSES)
                temp_model(dummy)
                temp_model.set_weights(new_global_weights)
                temp_model.save_weights(global_weights_path)
                
                del temp_model
                del new_global_weights
                gc.collect()
                tf.keras.backend.clear_session()
            
        else:
            print("⚠️ No updates received this round.")

    print("\n✅ Federated Training Finished!")

if __name__ == "__main__":
    try:
        main_fl()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()