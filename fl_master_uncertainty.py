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
WEIGHTS_DIR = "./fl_weights_uncertainty"
DATA_ROOT = "../datasets/nuscenes_preprocessed"
SPLIT_JSON_PATH = "./splits_federated/federated_split.json"
META_DIR = "./fl_meta_uncertainty"

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
    return manifest_paths, client_data

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
    """
    Performs Weighted Aggregation using Double Dynamic Normalization.
    Prioritizes Epistemic (Novelty) and penalizes Aleatoric (Noise).
    """
    print("☁️  [Master] Aggregating Weights (Uncertainty-Aware)...")
    
    client_metrics=[]
    
    for i in range(NUM_CLIENTS):
        m_path = os.path.join(WEIGHTS_DIR, f"client_{i}_round_{round_num}_metrics.json")
        w_path = os.path.join(WEIGHTS_DIR, f"client_{i}_round_{round_num}.weights.h5")
        
        if os.path.exists(m_path) and os.path.exists(w_path):
            with open(m_path, 'r') as f:
                data = json.load(f)
            client_metrics.append({
                'id': i,
                'path': w_path,
                'n': data['num_samples'],
                'nov_raw': data['novelty'], # Epistemic (Variance)
                'noise_raw': data['noise']  # Aleatoric (Entropy)
            })
    
    if not client_metrics: return

    # 2. Compute Dynamic Ranges (Min/Max) for this round
    # This solves the scale issue (e.g., Novelty being 1e-6 and Noise being 2.5)
    novs = [c['nov_raw'] for c in client_metrics]
    noises = [c['noise_raw'] for c in client_metrics]
    
    min_nov, max_nov = min(novs), max(novs)
    min_noise, max_noise = min(noises), max(noises)
    
    # Epsilon to avoid division by zero
    denom_nov = (max_nov - min_nov) + 1e-9
    denom_noise = (max_noise - min_noise) + 1e-9
    
    print(f"    📊 Round Stats | Novelty Range: [{min_nov:.2e}, {max_nov:.2e}] | Noise Range: [{min_noise:.2f}, {max_noise:.2f}]")
    
    client_updates = []
    total_score = 0.0
    
    # Hyperparameters for the weighting formula
    GAMMA = 2.0  # Strength of Novelty Reward (Exponential)
    BETA = 1.0   # Strength of Noise Penalty (Linear)

    # 3. Calculate Normalized Scores
    for c in client_metrics:
        # A. Normalize to  relative to the current group
        norm_nov = (c['nov_raw'] - min_nov) / denom_nov
        norm_noise = (c['noise_raw'] - min_noise) / denom_noise
        
        # B. Apply Formula: Ratio = Exp(Novelty) / (1 + Noise)
        # Numerator: Boosts highest novelty (norm=1) by e^GAMMA (e.g., e^2 = 7.3x)
        numerator = np.exp(norm_nov * GAMMA)
        
        # Denominator: Penalizes highest noise (norm=1) by (1 + BETA) (e.g., /2.0)
        denominator = 1.0 + (norm_noise * BETA)
        
        quality_ratio = numerator / denominator
        
        # Final raw alpha = data_size * quality_ratio
        alpha_raw = c['n'] * quality_ratio
        
        print(f"    🔍 Client {c['id']}: N={c['n']} | "
              f"NormNov={norm_nov:.2f} | NormNoise={norm_noise:.2f} | "
              f"QualityRatio={quality_ratio:.3f}")
        
        total_score += alpha_raw
        client_updates.append({'path': c['path'], 'score': alpha_raw})

    # 4. Perform Weighted Aggregation
    tf.keras.backend.clear_session()
    model = CLR_BNN(num_classes=NUM_CLASSES)
    dummy = [tf.zeros((1, 320, 320, 3)), tf.zeros((1, 320, 320, 2)), tf.zeros((1, 320, 320, 2))]
    model(dummy)
    
    avg_weights = None
    for update in client_updates:
        # Normalize weights to sum to 1.0 (Softmax logic)
        weight = update['score'] / total_score
        
        model.load_weights(update['path'])
        current_weights = model.get_weights()
        
        # Weighted sum
        weighted_w = [w * weight for w in current_weights]
        
        if avg_weights is None:
            avg_weights = weighted_w
        else:
            for idx, lay in enumerate(weighted_w):
                avg_weights[idx] += lay
                
    model.set_weights(avg_weights)
    
    # Save Global Model
    path = os.path.join(WEIGHTS_DIR, "global_weights.weights.h5")
    model.save_weights(path)
    
    # Save Round Checkpoint
    path_round = os.path.join(WEIGHTS_DIR, f"global_weights_round_{round_num}.weights.h5")
    model.save_weights(path_round)
    
    print("✅ New Global Model Saved.")
    del model
    tf.keras.backend.clear_session()

def main():
    # 1. Init
    global_path = init_global_model()
    generate_client_manifests()
    with open(SPLIT_JSON_PATH, 'r') as f: 
        splits = json.load(f)

    # 2. Main Loop
    for r in range(FL_ROUNDS):
        print(f"\n🌍 ================= ROUND {r+1}/{FL_ROUNDS} ================= 🌍")
            
            # ⚠️ SPECIAL LOGIC FOR ROUND 0
        # if r == 0:
            # print("⏩ SKIPPING TRAINING for Round 0 (Using Recycled FedAvg Weights).")
            # We assume recycle_fedavg_round0.py has already filled the folder
            # So we go straight to aggregation
        #  else:
        for c in range(NUM_CLIENTS):
            
            # if not os.path.exists(os.path.join(WEIGHTS_DIR, f"client_{c}_round_{r}.weights.h5")):

                print(f"\n▶️  Launching subprocess for Client {c}...")
                
                json_path = os.path.join(META_DIR, f"client_{c}_manifest.json")
                # --- THE MAGIC: SYSTEM CALL ---
                # This launches a completely new Python process.
                # When this process finishes, Windows/Linux reclaims ALL memory.
                cmd = [
                    "python", "fl_client_uncertanity.py",
                    "--client", str(c),
                    "--round", str(r),
                    "--global_path", global_path,
                    "--manifest", json_path,
                    "--data_root", DATA_ROOT,
                    "--sc_val_size", str(len(splits['clients'][f'client_{c}']['val']))
                ]
                
                # Run and wait for it to finish
                result = subprocess.run(cmd)
                
                if result.returncode != 0:
                    print(f"❌ CRITICAL: Client {c} crashed!")
                else:
                    print(f"✅ Client {c} finished successfully.")
            # else:
                # print(f"\n✅ Client {c} Round {r} already exists.")
            # 3. Aggregate
        aggregate_weights(r)

    print("\n🎉 Federated Learning Simulation Complete!")

if __name__ == "__main__":
    main()