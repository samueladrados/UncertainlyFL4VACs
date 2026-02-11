import hydra
from omegaconf import DictConfig
import os
import json
import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm
from tensorflow.keras import mixed_precision
import gc

from src.models.architecture import CLR_BNN
from src.data.loader import DataLoaderGenerator, AnchorBox


policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


# GPU Config
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"❌ GPU Config Error: {e}")

# --- STREAMER CLASS (To keep RAM empty) ---
class PredictionStreamer:
    def __init__(self, temp_dir, scenarios):
        self.temp_dir = temp_dir
        self.files = {}
        self.metrics_accumulators = {}
        for sc in scenarios:
            path = os.path.join(temp_dir, f"{sc}_preds.json")
            # "w" resets file, buffering=1 writes line by line
            self.files[sc] = open(path, "w", buffering=1) 
            self.metrics_accumulators[sc] = {'nll_sum': 0.0, 'nll_count': 0, 'obj_count': 0, 'ece_list': []}
            
    def save_batch(self, scenario_name, gt_batch, pred_batch, nll=None, n_objects=0, ece_m=None, ece_g=None):
        if nll is not None:
            self.metrics_accumulators[scenario_name]['nll_sum'] += nll
            self.metrics_accumulators[scenario_name]['nll_count'] += 1
            self.metrics_accumulators[scenario_name]['obj_count'] += n_objects
            self.metrics_accumulators[scenario_name]['ece_list'].extend([ece_m, ece_g])
        
        # Convert numpy to list for JSON serialization
        entry = {
            "gt": [arr.tolist() for arr in gt_batch],
            "pred": [arr.tolist() for arr in pred_batch]
        }
        self.files[scenario_name].write(json.dumps(entry) + "\n")

    def close(self):
        for f in self.files.values(): f.close()


def generate_expanded_test_manifests(split_json_path, output_dir, data_root):
    """
    1. Maps Scene IDs (e.g., 30) to filenames based on existing .npz labels.
    2. Reads 'federated_splits.json'.
    3. Expands 'global_test' (day, rain, night) into file lists.
    4. Saves 'test_clear.json', 'test_rain.json', etc.
    """
    print(f"\n🔄 Generating EXPANDED test manifests from: {split_json_path}")
    
    if not os.path.exists(split_json_path):
        print(f"❌ Error: Split file not found: {split_json_path}")
        exit(1)

    # --- STEP 1: MAP SCENES TO FILES (LOGIC FROM fl_utils.py) ---
    # We look into 'labels' folder to ensure ground truth exists
    labels_dir = os.path.join(data_root, 'labels')
    print(f"   📂 Mapping scenes from: {labels_dir} ...")
    
    if not os.path.exists(labels_dir):
        print(f"❌ Error: Labels directory not found at {labels_dir}")
        exit(1)

    # List all .npz files
    all_files = sorted([f.split('.')[0] for f in os.listdir(labels_dir) if f.endswith('.npz')])
    
    scene_map = {}
    count_files = 0
    
    for f in all_files:
        # Expected format: S001_K005_...
        try:
            parts = f.split('_')
            if parts[0].startswith('S'):
                scene_tag = parts[0]       # "S001"
                scene_idx = int(scene_tag[1:]) # 1
                
                if scene_idx not in scene_map: 
                    scene_map[scene_idx] = []
                
                scene_map[scene_idx].append(f)
                count_files += 1
        except ValueError:
            continue

    print(f"   ✅ Mapped {count_files} files to {len(scene_map)} unique scenes.")

    # --- STEP 2: LOAD SPLITS ---
    with open(split_json_path, 'r') as f:
        splits = json.load(f)

    # Access the structured global_test dictionary
    g_test = splits.get('global_test', {})
    
    # Get Scene IDs (Lists of integers)
    day_scenes = g_test.get('day', [])
    rain_scenes = g_test.get('rain', [])
    night_scenes = g_test.get('night', [])

    # --- STEP 3: EXPAND SCENES TO FILES ---
    def expand_scenes(scene_list):
        expanded_files = []
        for sid in scene_list:
            if sid in scene_map:
                expanded_files.extend(scene_map[sid])
            else:
                # Optional warning if a split scene is missing from disk
                # print(f"      ⚠️ Warning: Scene {sid} in splits but not on disk.")
                pass
        return expanded_files

    # Create file lists
    day_files = expand_scenes(day_scenes)
    rain_files = expand_scenes(rain_scenes)
    night_files = expand_scenes(night_scenes)
    
    # Combine all for "ALL"
    all_files = day_files + rain_files + night_files

    # --- STEP 4: SAVE MANIFESTS ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def save_json(filename, data_list):
        path = os.path.join(output_dir, filename)
        with open(path, 'w') as f:
            # Saving with "test" key as requested
            json.dump({"test": data_list}, f, indent=4)
        print(f"   -> 💾 Generated {filename}: {len(data_list)} files")

    # Mapping Day -> CLEAR, etc.
    save_json("test_clear.json", day_files)
    save_json("test_rain.json", rain_files)
    save_json("test_night.json", night_files)
    save_json("test_all.json", all_files)
    print("="*40 + "\n")

# --- DECODING UTILITIES ---

def decode_boxes(encoded_boxes, anchors):
    """
    Inverses the logic of AnchorBox.encode:
    t_x = (x - xa) / wa  -->  x = t_x * wa + xa
    t_y = (y - ya) / ha  -->  y = t_y * ha + ya
    t_w = log(w / wa)    -->  w = exp(t_w) * wa
    t_h = log(h / ha)    -->  h = exp(t_h) * ha
    """
    # anchors shape: [N, 4] -> (x, y, w, h)
    # encoded_boxes shape: [N, 4] -> (tx, ty, tw, th)
    
    t_x, t_y, t_w, t_h = encoded_boxes[:, 0], encoded_boxes[:, 1], encoded_boxes[:, 2], encoded_boxes[:, 3]
    a_x, a_y, a_w, a_h = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]

    t_w = np.clip(t_w, -6.0, 6.0)
    t_h = np.clip(t_h, -6.0, 6.0)

    ox = t_x * a_w + a_x
    oy = t_y * a_h + a_y
    ow = np.exp(t_w) * a_w
    oh = np.exp(t_h) * a_h

    # Convert from [cx, cy, w, h] to [x1, y1, x2, y2] for easier IoU calculation
    x1 = ox - ow / 2
    y1 = oy - oh / 2
    x2 = ox + ow / 2
    y2 = oy + oh / 2

    return np.stack([x1, y1, x2, y2], axis=-1)

def compute_iou(boxA, boxB):
    """Calculates IoU between one box and a list of boxes."""
    # boxA: [4]
    # boxB: [M, 4]
    xA = np.maximum(boxA[0], boxB[:, 0])
    yA = np.maximum(boxA[1], boxB[:, 1])
    xB = np.minimum(boxA[2], boxB[:, 2])
    yB = np.minimum(boxA[3], boxB[:, 3])

    interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[:, 2] - boxB[:, 0]) * (boxB[:, 3] - boxB[:, 1])

    iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)
    return iou

def calculate_ap(precisions, recalls):
    """Calculates Average Precision (Area under the PR curve)."""
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i-1] = np.maximum(precisions[i-1], precisions[i])
        
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap

def expected_calibration_error(y_true_cls, y_pred_cls, num_classes, n_bins=10):
    # Flatten to evaluate global calibration
    y_true = np.argmax(y_true_cls.reshape(-1, num_classes), axis=-1)
    y_pred = y_pred_cls.reshape(-1, num_classes)
    
    # Filter background (dummy class 0 if using pure one-hot, but your dataloader
    # seems to return one-hot only on positive anchors).
    # We evaluate only where there are real objects to avoid saturation with background.
    mask = np.sum(y_true_cls.reshape(-1, num_classes), axis=-1) > 0
    
    if np.sum(mask) == 0: return 0.0
    
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    pred_y = np.argmax(y_pred, axis=-1)
    confidences = np.max(y_pred, axis=-1)
    correct = (pred_y == y_true)

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(correct[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece


# --- COMPILED INFERENCE FUNCTION (FIXES MEMORY LEAK) ---
@tf.function(jit_compile=False)
def predict_step(model, images, training):
    return model(images, training=training)

# --- METRICS (NLL / ECE / mAP) ---
def evaluate_full_metrics(model, test_gen, anchors, scenarios_map, num_classes, num_anchors, temp_dir, mc_samples=1):
    if hasattr(test_gen, 'shuffle') and test_gen.shuffle:
        raise ValueError("❌ El generador 'test_gen' debe tener shuffle=False para evaluar métricas mapeadas por archivo.")
    print(f"    🧪 Running Evaluation (NLL, ECE & mAP)...")
    
    target_scenarios = list(scenarios_map.keys()) + ['ALL']
    streamer = PredictionStreamer(temp_dir, target_scenarios)
    
    total_batches = len(test_gen)
    
    # We access the file list to map Batch -> Filename -> Scenario
    all_filenames = test_gen.tokens

    pbar = tqdm(test_gen, total=total_batches, desc="    ⏳ Progress", unit="batch")
    
    # Global index to track which files we are processing
    global_idx = 0
    
    for batch_idx, ((X_img, X_lid, X_rad), (Y_cls, Y_box)) in enumerate(pbar):
        current_batch_size = X_img.shape[0]
        if current_batch_size == 0:
            print(f"⚠️ Skipping empty batch {batch_idx}, Image shape: {X_img.shape[0]}, Lidar shape: {X_lid.shape[0]}, Radar shape: {X_rad.shape[0]}")
            continue
        
        # Identify which files are in this batch
        batch_filenames = all_filenames[global_idx : global_idx + current_batch_size]
        global_idx += current_batch_size
        
        # --- 1. MONTE CARLO INFERENCE ---
        # We want to obtain: (Samples, Batch, Total_Anchors, Classes)
        mc_cls_flat = [] 
        mc_box_flat = []
        
        for _ in range(mc_samples):
            # training=True enables Dropout for MC Sampling
            outputs = predict_step(model, [X_img, X_lid, X_rad], training=(mc_samples > 1))
            raw_cls, raw_box = outputs[0], outputs[1]

            # B. Process each scale of the Feature Pyramid
            batch_cls_per_scale = []
            batch_box_per_scale = []

            for scale_idx in range(len(raw_cls)):
                # --- CLASSIFICATION ---
                logits = raw_cls[scale_idx]
                
                # Correct Reshape + Softmax (Batch, H, W, 9, 14)
                shape = tf.shape(logits)
                reshaped_logits = tf.reshape(logits, (shape[0], shape[1], shape[2], num_anchors, num_classes))
                reshaped_logits = tf.cast(reshaped_logits, tf.float32)
                probs_scale = tf.nn.sigmoid(reshaped_logits)
                
                # Flatten spatially to concatenate later: (Batch, N_Anchors_Scale, 14)
                # N_Anchors_Scale = H * W * 9
                probs_flat = tf.reshape(probs_scale, (shape[0], -1, num_classes))
                batch_cls_per_scale.append(probs_flat)

                # --- REGRESSION (BOXES) ---
                # Assume raw_box[scale_idx] is a distribution or a tensor
                box_out = raw_box[scale_idx]

                # If it's a TFP Distribution, take the mean
                if hasattr(box_out, 'mean'):
                    box_out = box_out.mean()

                # Flatten boxes: (Batch, H, W, 36) -> (Batch, N_Anchors_Scale, 4)
                box_shape = tf.shape(box_out)
                # Intermediate reshape to separate anchors: (B, H, W, 9, 4)
                box_reshaped = tf.reshape(box_out, (box_shape[0], box_shape[1], box_shape[2], num_anchors, 4))
                box_flat = tf.reshape(box_reshaped, (box_shape[0], -1, 4))
                batch_box_per_scale.append(box_flat)

            # C. Concatenate all scales (Reconstruct the giant anchor vector)
            # Result: (Batch, Total_Anchors, 14)
            full_cls_sample = tf.concat(batch_cls_per_scale, axis=1)
            full_box_sample = tf.concat(batch_box_per_scale, axis=1)
            
            mc_cls_flat.append(full_cls_sample.numpy())
            mc_box_flat.append(full_box_sample.numpy())
            
            del outputs, raw_cls, raw_box

        # --- 2. BAYESIAN AVERAGING ---
        # Now we can safely take the mean because all have shape (Batch, Total_Anchors, ...)
        avg_cls = np.mean(mc_cls_flat, axis=0) # (B, Total, 14)
        avg_box = np.mean(mc_box_flat, axis=0) # (B, Total, 4)
        
        del mc_cls_flat, mc_box_flat

        # --- DISTRIBUTION TO BUCKETS ---
        # Iterate element by element within the batch to assign to its scenario
        for i in range(current_batch_size):
            filename = batch_filenames[i]
            
            # Prepare individual data
            y_cls_i = Y_cls[i] # (Total_Anchors, 14)
            y_box_i = Y_box[i]
            avg_cls_i = avg_cls[i]
            avg_box_i = avg_box[i]
             
            # --- 3. NLL and ECE CALCULATION ---
            # Filter only anchors that contain objects (Ground Truth)
            # Y_cls is one-hot, sum(-1) > 0 tells us where there is an object
            obj_mask_i = np.sum(y_cls_i, axis=-1) > 0
            
            # --- NLL / ECE CALCULATION (Per Image) ---
            curr_nll = 0.0
            curr_ece_masked = 0.0
            curr_ece_global = 0.0
            has_obj = np.sum(obj_mask_i) > 0
            num_gt_objects = np.sum(obj_mask_i)
            gt_final = []
            pred_final = []
            
            if has_obj:
                # Flatten arrays based on mask
                y_true_masked = y_cls_i[obj_mask_i]     # (N_Obj, 14)
                y_pred_masked = avg_cls_i[obj_mask_i]   # (N_Obj, 14)
                
                # NLL Calculation
                # Probability assigned to the correct class
                prob_correct = np.sum(y_true_masked * y_pred_masked, axis=-1)
                prob_correct = np.clip(prob_correct, 1e-7, 1-1e-7) # Prevent log(0)
                curr_nll += -np.sum(np.log(prob_correct))
                
                # ECE Calculation
                curr_ece_masked = expected_calibration_error(y_true_masked, y_pred_masked)
                
                # ECE (current batch)
                curr_ece_global = expected_calibration_error(y_cls_i, avg_cls_i)

                # GT Boxes
                gt_indices = np.where(obj_mask_i)[0]
                gt_boxes_enc = y_box_i[gt_indices]
                gt_classes = np.argmax(y_cls_i[gt_indices], axis=-1)
                gt_anchors = anchors[gt_indices]
                gt_dec = decode_boxes(gt_boxes_enc, gt_anchors)
                gt_final = np.concatenate([gt_dec, gt_classes[:, None]], axis=-1)
        
                # Pred Decode
                scores = np.max(avg_cls_i, axis=-1)
                pred_idx = np.where(scores > 0.05)[0]
                if len(pred_idx) > 0:
                    p_boxes_enc = avg_box_i[pred_idx]
                    p_anchors = anchors[pred_idx]
                    p_scores = scores[pred_idx]
                    p_classes = np.argmax(avg_cls_i[pred_idx], axis=-1)
                    p_dec = decode_boxes(p_boxes_enc, p_anchors)
                    
                    # Local NMS
                    local_preds = []
                    for c in range(num_classes):
                        cls_mask = p_classes == c
                        if np.sum(cls_mask) == 0: continue
                        c_boxes = p_dec[cls_mask]
                        c_scores = p_scores[cls_mask]
                        c_boxes_tf = tf.stack([c_boxes[:, 1], c_boxes[:, 0], c_boxes[:, 3], c_boxes[:, 2]], axis=1)
                        
                        sel_idx = tf.image.non_max_suppression(c_boxes, c_scores, 50, 0.5).numpy()
                        for idx in sel_idx:
                            box = c_boxes[idx]
                            local_preds.append([box[0], box[1], box[2], box[3], c, c_scores[idx]])
                    pred_final = np.array(local_preds)
                else:
                    pred_final = np.array([])                    
            else:
                pred_final = np.array([])
                
            # --- ASSIGN TO BUCKETS (STREAM TO DISK) ---
            active_scenarios = ['ALL']
            for sc_name, sc_set in scenarios_map.items():
                if filename in sc_set:
                    active_scenarios.append(sc_name)
            
            for b_name in active_scenarios:
                gt_list = [gt_final] if len(gt_final) > 0 else []
                pred_list = [pred_final] if len(pred_final) > 0 else []
                
                # Write to disk via streamer
                streamer.save_batch(b_name, gt_list, pred_list, 
                                    nll=curr_nll if has_obj else 0,
                                    n_objects=num_gt_objects,
                                    ece_m=curr_ece_masked, ece_g=curr_ece_global)

        del avg_cls, avg_box, X_img, X_lid, X_rad, Y_cls, Y_box
        gc.collect()

    streamer.close()

    # --- FINAL mAP CALCULATION ---
    print("\n    📉 Calculating final metrics for all scenarios...")
    final_results = {}
    
    for name in target_scenarios:
        acc = streamer.metrics_accumulators[name]
                  
        # NLL & ECE
        total_objects = acc['obj_count']
        if total_objects > 0:
            mean_nll = acc['nll_sum'] / total_objects
        else:
            mean_nll = 0.0
        mean_ece = np.mean(acc['ece_list']) if len(acc['ece_list']) > 0 else 0.0
    
        # mAP Calculation
        print(f"      📐 Calc mAP for: {name}...")
        aps = []
        path = os.path.join(temp_dir, f"{name}_preds.json")
    
        # Load accumulated data from disk just for this calculation
        all_gt = []
        all_pred = []
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if len(data['gt']) > 0: all_gt.append(np.array(data['gt'][0]))
                else: all_gt.append(np.array([]))
                if len(data['pred']) > 0: all_pred.append(np.array(data['pred'][0]))
                else: all_pred.append(np.array([]))
                
        for c in range(num_classes):
            true_positives = []
            scores_list = []
            num_annotations = 0
            
            for i in range(len(all_gt)):
                gt = all_gt[i]
                pred = all_pred[i]
                
                if len(gt) > 0:
                    gt_c = gt[gt[:, 4] == c]
                    num_annotations += len(gt_c)
                else:
                    gt_c = []
                    
                if len(pred) > 0:
                    pred_c = pred[pred[:, 4] == c]
                else:
                    pred_c = []
                    
                if len(pred_c) == 0: continue
                
                # Sort preds by score
                pred_c = pred_c[np.argsort(-pred_c[:, 5])]
                
                detected_gt = np.zeros(len(gt_c))
                
                for p in pred_c:
                    scores_list.append(p[5])
                    if len(gt_c) == 0:
                        true_positives.append(0)
                        continue
                    
                    ious = compute_iou(p[:4], gt_c[:, :4])
                    max_iou = np.max(ious)
                    max_idx = np.argmax(ious)
                    
                    if max_iou >= 0.5:
                        if detected_gt[max_idx] == 0:
                            true_positives.append(1)
                            detected_gt[max_idx] = 1 # Match
                        else:
                            true_positives.append(0) # Duplicate
                    else:
                        true_positives.append(0) # FP
            
            if num_annotations == 0: continue
            
            # Compute AP
            tp = np.array(true_positives)
            sc = np.array(scores_list)
            
            # Sort total results by score
            idx = np.argsort(-sc)
            tp = tp[idx]
            
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(1 - tp)
            
            rec = cum_tp / num_annotations
            prec = cum_tp / (cum_tp + cum_fp + 1e-6)
            
            aps.append(calculate_ap(prec, rec))


        mean_ap = np.mean(aps) if len(aps) > 0 else 0.0
        final_results[name] = {'mAP': mean_ap, 'NLL': mean_nll, 'ECE': mean_ece}
        
        del all_gt, all_pred
        gc.collect()
    
    return final_results


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    orig_cwd = hydra.utils.get_original_cwd()
    DATA_ROOT = cfg.data.root
    SPLIT_JSON = os.path.join(orig_cwd, cfg.data.split_path)
    strategy_name = cfg.federated.get("strategy_name", "uncertainty_weighted")
    WEIGHTS_DIR = os.path.join(orig_cwd, cfg.strategy[strategy_name].weights_dir)
    RESULTS_DIR = os.path.join(orig_cwd, "evaluation_results")
    TEMP_DIR = os.path.join(RESULTS_DIR, "temp")
    META_TEST_DIR = os.path.join(RESULTS_DIR, "meta_test")
    MODEL_WEIGHTS = os.path.join(WEIGHTS_DIR, "global_weights.weights.h5")
    
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"❌ Cannot find model weight file: {MODEL_WEIGHTS}")
        print(f"   Searched path: {WEIGHTS_DIR}")
        return
    
    generate_expanded_test_manifests(SPLIT_JSON, META_TEST_DIR, DATA_ROOT)
    
    # 1. LOAD SCENARIO MAPS (FILENAMES ONLY)
    scenarios_def = {
        "CLEAR": "test_clear.json",
        "RAIN": "test_rain.json",
        "NIGHT": "test_night.json"
    }

    # Sets for fast O(1) lookup
    scenarios_map = {} 
    
    print("📋 Loading Scenario Manifests for filtering...")
    for name, json_file in scenarios_def.items():
        path = os.path.join(META_TEST_DIR, json_file)
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                files = data['test'] if 'test' in data else data
                scenarios_map[name] = set(files) # Convert to SET is key for speed
                print(f"   -> {name}: {len(scenarios_map[name])} files loaded.")
        else:
            print(f"   ⚠️ Warning: {json_file} not found.")
    
    # 2. LOAD THE FULL DATASET (ALL)
    print("🌍 Loading FULL Test Dataset Generator...")
    full_manifest_path = os.path.join(META_TEST_DIR, "test_all.json")
    with open(full_manifest_path, 'r') as f:
        data = json.load(f)
        all_files = data['test'] if 'test' in data else data

    test_gen = DataLoaderGenerator(DATA_ROOT, specific_files=all_files, batch_size=cfg.client.batch_size)

    results = {}

    # 3. BUILD MODEL (ONLY ONCE)
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Generate Anchors
    anchor_gen = AnchorBox(input_shape=cfg.data.input_shape)
    anchors = anchor_gen.anchors 

    print(f"🏗️ Loading CLR-BNN model...")
    model = CLR_BNN(um_classes=cfg.data.num_classes, num_anchors=cfg.data.num_anchors)
    dummy_dtype = policy.compute_dtype
    dummy = [
        tf.zeros((1, 320, 320, 3), dtype=dummy_dtype), # Image
        tf.zeros((1, 320, 320, 2), dtype=dummy_dtype), # LiDAR
        tf.zeros((1, 320, 320, 2), dtype=dummy_dtype)  # Radar
    ]
    model(dummy) # Build
    
    try:
        print(f"📥 Loading weights: {MODEL_WEIGHTS}")
        model.load_weights(MODEL_WEIGHTS)
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        exit(-1)

    # 4. EXECUTE EVALUATION
    results = evaluate_full_metrics(model, test_gen, anchors, scenarios_map, cfg.data.num_classes, cfg.data.num_anchors, TEMP_DIR, mc_samples=cfg.client.mc_samples)
    
    # 5. SAVE AND REPORT
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    final_report = {
        "model_name": strategy_name,
        "weights_path": MODEL_WEIGHTS,
        "evaluation_date": timestamp,
        "metrics": results
    }
    json_path = os.path.join(RESULTS_DIR, f"{strategy_name}_Evaluation_{timestamp}.json")
    
    with open(json_path, 'w') as f:
        json.dump(final_report, f, indent=4)
        
    print("\n" + "="*60)
    print("📝 SINGLE PASS EVALUATION SUMMARY")
    print("="*60)
    print(f"{'Scenario':<20} | {'mAP':<10} | {'NLL':<10} | {'ECE':<10}")
    print("-" * 55)
    # Print nicely ordered
    for name in ['ALL', 'CLEAR', 'RAIN', 'NIGHT']:
        if name in results:
            m = results[name]
            print(f"{name:<20} | {m['mAP']:.4f}     | {m['NLL']:.4f}     | {m['ECE']:.4f}")
    print("="*60)
    
    # Final cleanup
    del model, test_gen
    gc.collect()

if __name__ == "__main__":
    main()