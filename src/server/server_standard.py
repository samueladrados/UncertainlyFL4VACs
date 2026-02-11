import hydra
from omegaconf import DictConfig
import os
import subprocess
import tensorflow as tf
import numpy as np
import json
import shutil
import sys
from tensorflow.keras import mixed_precision

from src.models.architecture import CLR_BNN
from src.models.transfer import inject_imagenet_prior_only
from src.utils.file_utils import load_federated_partition

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def generate_client_manifests(cfg, split_json_abs_path, meta_dir_abs_path):
    """
    Loads the fixed Federated Split JSON and creates individual 
    instruction files (manifests) for each client process.
    """
    print("📋 [Master] Loading Federated Partition from JSON...")
    
    # 1. Load the map (Client ID -> List of Files)
    # global_test_files is returned but not used by clients directly
    client_data, _ = load_federated_partition(cfg.data.root, split_json_abs_path)
    
    manifest_paths = {}
    
    if not os.path.exists(meta_dir_abs_path):
        os.makedirs(meta_dir_abs_path)
    
    for i in range(cfg.federated.num_clients):
        if i not in client_data:
            print(f"❌ Error: Client {i} not found in JSON partition!")
            sys.exit(1)

        # Save the specific dict {'train': [...], 'val': [...]} for this client
        fname = os.path.join(meta_dir_abs_path, f"client_{i}_manifest.json")
        
        with open(fname, 'w') as f:
            json.dump(client_data[i], f)
            
        manifest_paths[i] = fname
        
    print("✅ Client Manifests generated successfully.")
    return manifest_paths, client_data

def init_global_model(cfg, weights_dir):
    print("🔵 [Master] Initializing Global Model...")
    
    # Initialize and save the very first model
    model = CLR_BNN(num_classes=cfg.data.num_classes)
    
    # Dummy pass to build graph
    dummy = [tf.zeros((1, 320, 320, 3)), tf.zeros((1, 320, 320, 2)), tf.zeros((1, 320, 320, 2))]
    model(dummy)
    
    if cfg.model.inject_priors:
        inject_imagenet_prior_only(model)
    
    # Save initial weights
    path = os.path.join(weights_dir, "global_weights.weights.h5")
    model.save_weights(path)
    
    # Keep a copy of the very first initialization for reference
    path_initial = os.path.join(weights_dir, "global_weights_initial.weights.h5")
    shutil.copy2(path, path_initial)
    
    print(f"    💾 Global initialized at {path}")
    return path

def aggregate_weights(cfg, round_num, weights_dir, client_counts):
    """
    Standard Federated Averaging (FedAvg).
    """
    print("☁️  [Master] Aggregating Weights (Weighted FedAvg)...")
    
    # --- STEP 1: Calculate Data Distribution from Manifests ---
    # We need to know how many samples each client has to calculate 'alpha'
    total_samples = sum(client_counts.values())
    print(f"    📊 Total training samples: {total_samples}")

    # --- STEP 2: Aggregation Loop ---
    avg_weights = None
    
    # Create temporary model just to load weights
    tf.keras.backend.clear_session()
    model = CLR_BNN(num_classes=cfg.data.num_classes)
    dummy = [tf.zeros((1, 320, 320, 3)), tf.zeros((1, 320, 320, 2)), tf.zeros((1, 320, 320, 2))]
    model(dummy)

    # Track effective weight sum (in case a client crashed and is missing)
    current_total_weight = 0.0
    files_found = 0

    for i in range(cfg.federated.num_clients):
        f_path = os.path.join(weights_dir, f"client_{i}_round_{round_num}.weights.h5")
        
        if os.path.exists(f_path):
            # Calculate Importance Factor (Alpha) = n_i / N_total
            n_i = client_counts[i]
            alpha = n_i / total_samples
            
            print(f"    ➕ Adding Client {i} (samples={n_i}, importance={alpha:.4f})")
            
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
        path = os.path.join(weights_dir, "global_weights.weights.h5")
        model.save_weights(path)
        path_round = os.path.join(weights_dir, f"global_weights_round_{round_num}.weights.h5")
        model.save_weights(path_round)
        print("✅ New Global Model Saved (Weighted Average).")
    else:
        print("❌ Aggregation failed: No weights found.")
    
    # Cleanup
    del model
    tf.keras.backend.clear_session()

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    
    set_seed(cfg.system.seed)
    
    orig_cwd = hydra.utils.get_original_cwd()
    SPLIT_JSON_PATH = os.path.join(orig_cwd, cfg.data.split_path)
    
    strat_conf = cfg.strategy.standard_fedavg
    
    WEIGHTS_DIR_ABS = os.path.join(orig_cwd, strat_conf.weights_dir)
    META_DIR_ABS = os.path.join(orig_cwd, strat_conf.meta_dir)
    LOGS_DIR_ABS = os.path.join(orig_cwd, cfg.federated.logs_dir)
    
    if not os.path.exists(WEIGHTS_DIR_ABS): os.makedirs(WEIGHTS_DIR_ABS)
    if not os.path.exists(META_DIR_ABS): os.makedirs(META_DIR_ABS)
    
    # 1. Initialize Global Model & Data
    init_global_model(cfg, WEIGHTS_DIR_ABS)
    _, client_data = generate_client_manifests(cfg, SPLIT_JSON_PATH, META_DIR_ABS)
    client_counts = {i: len(client_data[i]['train']) for i in range(cfg.federated.num_clients)}
    
    # global_path = os.path.join(WEIGHTS_DIR, "global_weights.weights.h5")
    # 2. Main Federated Loop
    for r in range(cfg.federated.num_rounds):
        print(f"\n🌍 ================= ROUND {r+1}/{cfg.federated.num_rounds} ================= 🌍")
        
        for c in range(cfg.federated.num_clients):
            print(f"\n▶️  Launching subprocess for Client {c}...")
            
            json_path = os.path.join(META_DIR_ABS, f"client_{c}_manifest.json")
            
            # --- SYSTEM CALL TO FL_CLIENT ---
            # Calls fl_client.py passing the specific JSON manifest
            cmd = [
                "python", "-m", "src.client.app_standard",
                f"client.id={c}",
                f"+federated.round={r}",
                f"client.manifest_path={json_path}",
                f"strategy.standard_fedavg.weights_dir={WEIGHTS_DIR_ABS}",
                f"federated.logs_dir={LOGS_DIR_ABS}"
            ]
            
            # Run and wait
            result = subprocess.run(cmd)
            
            if result.returncode != 0:
                print(f"❌ CRITICAL: Client {c} crashed!")
            else:
                print(f"✅ Client {c} finished successfully.")

        # 3. Aggregate
        aggregate_weights(cfg, r, WEIGHTS_DIR_ABS, client_counts)

    print("\n🎉 Federated Learning Simulation Complete!")

if __name__ == "__main__":
    main()