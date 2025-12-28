import os
import subprocess
import tensorflow as tf
import numpy as np
from clr_bnn import CLR_BNN
from inject_imagenet import inject_imagenet_prior_only
import json
from fl_utils import load_federated_partition
import sys
import shutil
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# --- PARAMETERS ---
NUM_CLIENTS = 10
FL_ROUNDS = 10
NUM_CLASSES = 14
WEIGHTS_DIR = "./fl_weights_temp"
DATA_ROOT = "/mnt/c/Users/USUARIO/Desktop/enviroments/Paper_NOUS/datasets/nuscenes_preprocessed"
SPLIT_JSON_PATH = "splits_federated/federated_split.json"
META_DIR = "./fl_meta_temp"

# FIXED SEED
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

if not os.path.exists(WEIGHTS_DIR): os.makedirs(WEIGHTS_DIR)
if not os.path.exists(META_DIR): os.makedirs(META_DIR)

def generate_client_manifests():
    """
    Loads the fixed Federated Split JSON and creates individual 
    instruction files (manifests) for each client process.
    """
    print("📋 [Master] Loading Federated Partition from JSON...")
    
    # 1. Load the map (Client ID -> List of Files)
    # global_test_files is returned but not used by clients directly
    client_data, _ = load_federated_partition(DATA_ROOT, SPLIT_JSON_PATH)
    
    manifest_paths = {}
    
    for i in range(NUM_CLIENTS):
        if i not in client_data:
            print(f"❌ Error: Client {i} not found in JSON partition!")
            sys.exit(1)

        # Save the specific dict {'train': [...], 'val': [...]} for this client
        fname = os.path.join(META_DIR, f"client_{i}_manifest.json")
        
        with open(fname, 'w') as f:
            json.dump(client_data[i], f)
            
        manifest_paths[i] = fname
        
    print("✅ Client Manifests generated successfully.")
    return manifest_paths

def init_global_model():
    print("🔵 [Master] Initializing Global Model...")
    
    # Initialize and save the very first model
    model = CLR_BNN(num_classes=NUM_CLASSES)
    
    # Dummy pass to build graph
    dummy = [tf.zeros((1, 320, 320, 3)), tf.zeros((1, 320, 320, 2)), tf.zeros((1, 320, 320, 2))]
    model(dummy)
    inject_imagenet_prior_only(model)
    
    # Save initial weights
    path = os.path.join(WEIGHTS_DIR, "global_weights.weights.h5")
    model.save_weights(path)
    
    # Keep a copy of the very first initialization for reference
    path_initial = os.path.join(WEIGHTS_DIR, "global_weights_initial.weights.h5")
    shutil.copy2(path, path_initial)
    
    print(f"    💾 Global initialized at {path}")
    return path

def aggregate_weights(round_num):
    print("☁️  [Master] Aggregating Weights (Weighted FedAvg)...")
    
    # --- STEP 1: Calculate Data Distribution from Manifests ---
    # We need to know how many samples each client has to calculate 'alpha'
    client_samples = {}
    total_samples = 0
    
    for i in range(NUM_CLIENTS):
        manifest_path = os.path.join(META_DIR, f"client_{i}_manifest.json")
        with open(manifest_path, 'r') as f:
            data = json.load(f)
            n = len(data['train']) # We weight based on training set size
            client_samples[i] = n
            total_samples += n
    
    print(f"    📊 Total training samples in federation: {total_samples}")

    # --- STEP 2: Aggregation Loop ---
    avg_weights = None
    
    # Create temporary model just to load weights
    tf.keras.backend.clear_session()
    model = CLR_BNN(num_classes=NUM_CLASSES)
    dummy = [tf.zeros((1, 320, 320, 3)), tf.zeros((1, 320, 320, 2)), tf.zeros((1, 320, 320, 2))]
    model(dummy)

    # Track effective weight sum (in case a client crashed and is missing)
    current_total_weight = 0.0
    files_found = 0

    for i in range(NUM_CLIENTS):
        f_path = os.path.join(WEIGHTS_DIR, f"client_{i}_round_{round_num}.weights.h5")
        
        if os.path.exists(f_path):
            # Calculate Importance Factor (Alpha) = n_i / N_total
            alpha = client_samples[i] / total_samples
            
            print(f"    ➕ Adding Client {i} (samples={client_samples[i]}, importance={alpha:.4f})")
            
            model.load_weights(f_path)
            w = model.get_weights()
            
            # WEIGHTING: Multiply weights by their importance (alpha)
            weighted_w = [layer * alpha for layer in w]
            
            if avg_weights is None:
                # Initialize accumulator
                avg_weights = weighted_w
            else:
                # Add to accumulator
                for idx, layer in enumerate(weighted_w):
                    avg_weights[idx] += layer
            
            current_total_weight += alpha
            files_found += 1
        else:
            print(f"    ⚠️ Missing weights for client {i} (Skipping contribution)")

    # --- STEP 3: Normalization & Saving ---
    if avg_weights and files_found > 0:
        
        # SAFETY CHECK: If a client crashed, the alphas sum to < 1.0. 
        # We must re-normalize so the weights don't vanish.
        if current_total_weight < 0.999:
             normalization_factor = 1.0 / current_total_weight
             print(f"    ⚠️ Normalizing weights (Factor: {normalization_factor:.4f}) due to missing clients.")
             avg_weights = [x * normalization_factor for x in avg_weights]

        # Save New Global Model
        model.set_weights(avg_weights)
        path = os.path.join(WEIGHTS_DIR, "global_weights.weights.h5")
        model.save_weights(path)
        path_round = os.path.join(WEIGHTS_DIR, f"global_weights_round_{round_num}.weights.h5")
        model.save_weights(path_round)
        print("✅ New Global Model Saved (Weighted Average).")
    else:
        print("❌ Aggregation failed: No weights found.")
    
    # Cleanup
    del model
    tf.keras.backend.clear_session()

def main():
    # 1. Initialize Global Model & Data
    global_path = init_global_model()
    generate_client_manifests()
    # global_path = os.path.join(WEIGHTS_DIR, "global_weights.weights.h5")
    # 2. Main Federated Loop
    for r in range(FL_ROUNDS):
        print(f"\n🌍 ================= ROUND {r+1}/{FL_ROUNDS} ================= 🌍")
        
        for c in range(NUM_CLIENTS):
            print(f"\n▶️  Launching subprocess for Client {c}...")
            
            json_path = os.path.join(META_DIR, f"client_{c}_manifest.json")
            
            # --- SYSTEM CALL TO FL_CLIENT ---
            # Calls fl_client.py passing the specific JSON manifest
            cmd = [
                "python", "fl_client.py",
                "--client", str(c),
                "--round", str(r),
                "--global_path", global_path,
                "--manifest", json_path,
                "--data_root", DATA_ROOT
            ]
            
            # Run and wait
            result = subprocess.run(cmd)
            
            if result.returncode != 0:
                print(f"❌ CRITICAL: Client {c} crashed!")
            else:
                print(f"✅ Client {c} finished successfully.")

        # 3. Aggregate
        aggregate_weights(r)

    print("\n🎉 Federated Learning Simulation Complete!")

if __name__ == "__main__":
    main()