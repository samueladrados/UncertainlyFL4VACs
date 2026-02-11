import hydra
from omegaconf import DictConfig
import argparse
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


def select_diverse_samples(file_list, n_samples=20):
    """
    Selects a subset of images maximizing scene diversity to ensure
    representative uncertainty metrics.
    """
    total_files = len(file_list)
    
    # Enforce a statistical lower bound: we need at least 20 samples 
    # to calculate meaningful variance/entropy scores.
    if n_samples < 20:
        n_samples = 20
        
    # Boundary check: If the dataset is smaller than the request, 
    # return the entire dataset as no selection logic is needed.
    if total_files <= n_samples:
        return file_list

    # ---------------------------------------------------------
    # DIVERSITY STRATEGY (Prioritize unique contexts)
    # ---------------------------------------------------------
    
    # Parsing step: Isolate the 'Scene ID' from filenames (e.g., 'S001_...') 
    # to cluster images that belong to the same recording sequence.
    scenes = {}
    for f in file_list:
        parts = f.split('_') 
        # Fallback to 'unknown' if filename format is unexpected
        scene_id = parts[0] if len(parts) > 0 else "unknown"
        
        if scene_id not in scenes: scenes[scene_id] = []
        scenes[scene_id].append(f)
    
    unique_ids = list(scenes.keys())
    diverse_files = []
    
    # SCENARIO A: High Diversity Availability
    # The number of unique scenes exceeds the requested sample size.
    # We select one image per scene to maximize data independence.
    if len(unique_ids) >= n_samples:
        selected_scenes = np.random.choice(unique_ids, n_samples, replace=False)
        for s_id in selected_scenes:
            f = np.random.choice(scenes[s_id])
            diverse_files.append(f)
            
    # SCENARIO B: Limited Diversity Availability
    # We have fewer unique scenes than requested samples.
    # We must reuse scenes, but we prioritize covering every scene at least once.
    else:
        # Step 1: Guarantee coverage by picking one image from every available scene.
        for s_id in unique_ids:
            f = np.random.choice(scenes[s_id])
            diverse_files.append(f)
            
        # Step 2: Calculate the deficit to reach the target 'n_samples'.
        missing = n_samples - len(diverse_files)
        
        # Prepare the pool of unselected images (Total - Already Picked) to avoid duplicates.
        all_files_set = set(file_list)
        selected_set = set(diverse_files)
        remaining_pool = list(all_files_set - selected_set)
        
        # Step 3: Backfill the remaining quota randomly from the unselected pool.
        if len(remaining_pool) >= missing:
            extras = np.random.choice(remaining_pool, missing, replace=False)
            diverse_files.extend(extras.tolist())
        else:
            # Edge case: If even the remaining pool isn't enough, take everything left.
            diverse_files.extend(remaining_pool)

    return diverse_files

def calculate_metrics(model, data_generator, num_anchors, num_classes, num_samples=5):
    """
    Calculates uncertainty using the Kendall & Gal decomposition (Information Theory).
    
    Decomposition Logic:
    Total Uncertainty (Predictive Entropy) = Aleatoric (Expected Entropy) + Epistemic (Mutual Information)
    
    Args:
        model: The Bayesian Neural Network (CLR_BNN).
        data_generator: Iterator yielding (images, labels).
        num_batches: Number of batches to process (optional limit).
        num_samples: Number of Monte Carlo forward passes (T).
        
    Returns:
        final_epi: Global Epistemic Uncertainty Score (Novelty).
        final_ale: Global Aleatoric Uncertainty Score (Noise).
    """
    print(f"    🤔 Calculating Uncertainty Metrics ({num_samples} MC passes)...")
    
    epistemic_scores = []
    aleatoric_scores = []
    epsilon = 1e-7  # Numerical stability
    
    expected_channels = num_anchors * num_classes
    
    for i, (images, _) in enumerate(data_generator):
        mc_preds_per_scale = {}
        
        # ---------------------------------------------------------
        # 1. MONTE CARLO SAMPLING
        # ---------------------------------------------------------
        for _ in range(num_samples):
            # ⚠️ CRITICAL: training=True forces stochastic behavior (Dropout/Flipout)
            outputs = model(images, training=True)
            
            # Handle model output structure (Tuple vs Single Tensor)
            if isinstance(outputs, (list, tuple)): 
                cls_outputs = outputs[0] # Index 0 is usually Classification
            else: 
                cls_outputs = [outputs]
            
            # Iterate through Feature Pyramid Levels (P3, P4, P5...)
            for scale_idx, logits in enumerate(cls_outputs):
                # Verify we are processing the Classification Head (Channels = Num_Classes)
                if logits.shape[-1] == expected_channels:
                    shape = tf.shape(logits)
                    # Reshape: (Batch, H, W, Anchors, Classes)
                    reshaped_logits = tf.reshape(logits, (shape[0], shape[1], shape[2], num_anchors, num_classes))
                    
                    # Convert Logits to Probabilities (Softmax)
                    probs = tf.nn.softmax(tf.cast(reshaped_logits, tf.float32), axis=-1)
                    
                    if scale_idx not in mc_preds_per_scale: 
                        mc_preds_per_scale[scale_idx] = []
                    mc_preds_per_scale[scale_idx].append(probs)

        # ---------------------------------------------------------
        # 2. METRIC CALCULATION (Kendall Decomposition)
        # ---------------------------------------------------------
        batch_epistemic = []
        batch_aleatoric = []
        
        for _, sample_list in mc_preds_per_scale.items():
            # Stack samples: [Samples, Batch, H, W, Anchors, Classes]
            p_stack = tf.stack(sample_list, axis=0)
            
            # --- A) TOTAL UNCERTAINTY (Predictive Entropy) ---
            # H(Mean Probability): Entropy of the averaged prediction
            p_mean = tf.reduce_mean(p_stack, axis=0) 
            total_uncertainty = -tf.reduce_sum(p_mean * tf.math.log(p_mean + epsilon), axis=-1)
            
            # --- B) ALEATORIC UNCERTAINTY (Expected Entropy) ---
            # Mean of H(Individual Probabilities): Captures Data Noise
            # 1. Calculate entropy for each individual sample
            entropies = -tf.reduce_sum(p_stack * tf.math.log(p_stack + epsilon), axis=-1)
            # 2. Average the entropies
            aleatoric = tf.reduce_mean(entropies, axis=0)
            
            # --- C) EPISTEMIC UNCERTAINTY (Mutual Information) ---
            # MI = Total Uncertainty - Aleatoric Uncertainty
            # Captures Model Ignorance / Novelty
            epistemic = total_uncertainty - aleatoric
            
            # Enforce non-negativity (handling floating-point precision errors)
            epistemic = tf.maximum(epistemic, 0.0)

            # --- D) SPATIAL AGGREGATION ---
            # Note: Averaging over H, W, and Anchors dilutes the score due to 
            # the dominant background class, but it serves as a valid relative 
            # score for Federated Learning client weighting.
            batch_aleatoric.append(tf.reduce_mean(aleatoric))
            batch_epistemic.append(tf.reduce_mean(epistemic))

        # Average across pyramid scales
        if batch_epistemic:
            epistemic_scores.append(tf.reduce_mean(batch_epistemic).numpy())
            aleatoric_scores.append(tf.reduce_mean(batch_aleatoric).numpy())

    # Final Global Average
    final_epi = float(np.mean(epistemic_scores)) if epistemic_scores else 0.0
    final_ale = float(np.mean(aleatoric_scores)) if aleatoric_scores else 0.0
    
    print(f"    ✅ Calculated (Information Theoretic): Epistemic={final_epi:.6f} | Aleatoric={final_ale:.6f}")
    
    return final_epi, final_ale

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def run_client(cfg: DictConfig):
    
    client_id = cfg.client.id
    round_num = cfg.federated.get("round", 0)
    manifest_path = cfg.client.manifest_path
    sc_val_size = cfg.client.diversity_sample_size
    weights_dir = cfg.strategy.uncertainty_weighted.weights_dir
    data_root = cfg.data.root
    num_classes = cfg.data.num_classes
    num_anchors = cfg.data.num_anchors
    batch_size = cfg.client.batch_size
    val_batch_size = cfg.client.val_batch_size
    mc_samples = cfg.client.mc_samples
    
    print(f"🚀 [Worker] Starting Client {client_id} for Round {round_num}")
    
    # 1. Deterministic Partitioning (Same seed every time so clients get same data)
    print(f"    📖 Reading manifest: {manifest_path}")
    with open(manifest_path, 'r') as f:
        my_files = json.load(f)
    
    diverse_files = select_diverse_samples(my_files['val'], n_samples=sc_val_size)

    # 2. Data Generator
    val_gen = DataLoaderGenerator(data_root, specific_files=diverse_files, batch_size=val_batch_size)
    train_gen = DataLoaderGenerator(data_root, specific_files=my_files['train'], batch_size=batch_size)

    # 3. Build Model
    local_model = CLR_BNN(num_classes=num_classes, num_anchors=num_anchors)
    # Build graph with dummy input
    dummy = [tf.zeros((1, 320, 320, 3)), tf.zeros((1, 320, 320, 2)), tf.zeros((1, 320, 320, 2))]
    local_model(dummy)
    
    global_weights_path = os.path.join(weights_dir, "global_weights.weights.h5")
    
    # 4. Load Global Weights
    print(f"    📥 Loading Global Weights: {global_weights_path}")
    local_model.load_weights(global_weights_path)
    
    # 5. Pre-Train: Measure Novelty (Epistemic)
    print("    📊 Measuring Pre-Training Novelty...")
    u_novelty, _ = calculate_metrics(local_model, val_gen, num_anchors, num_classes, mc_samples)
    print(f"       -> Novelty Score: {u_novelty:.6f}")
    
    print(f"    📊 Training on {len(my_files['train'])} files (Batch Size: {batch_size})")

    # 6. Initialize Trainer
    trainer = CLR_BNN_Trainer(
        local_model, 
        num_classes=num_classes,
        num_anchors=num_anchors,
        initial_lr=cfg.client.learning_rate, 
        current_round=round_num,
        steps_per_epoch=len(train_gen),
        train_dataset_size=len(my_files['train'])
    )
    
    logger = FederatedLogger(
        client_id=f"client_{client_id}",
        round_num=round_num,
        log_dir=cfg.federated.logs_dir
    )

    # 7. Train (Local Epochs)
    trainer.fit(train_gen, None, epochs=cfg.client.epochs, callbacks=[logger]) 
    
    # 8. Post-Train: Measure Noise (Aleatoric)
    print("    📊 Measuring Post-Training Noise...")
    _, u_noise = calculate_metrics(local_model, val_gen, num_anchors, num_classes, mc_samples)
    print(f"       -> Noise Score: {u_noise:.6f}")
    
    # 9. Save to Disk
    if not os.path.exists(weights_dir): os.makedirs(weights_dir)
    save_path = os.path.join(weights_dir, f"client_{client_id}_round_{round_num}.weights.h5")
    local_model.save_weights(save_path)
    
    metrics = {
        "num_samples": len(my_files['train']),
        "novelty": u_novelty,
        "noise": u_noise
    }
    metrics_path = os.path.join(weights_dir, f"client_{client_id}_round_{round_num}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    print(f"    💾 Saved local weights and metrics to: {save_path}")

if __name__ == "__main__":    
    try:
        run_client()
    except Exception as e:
        print(f"❌ [Worker Error] Failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)