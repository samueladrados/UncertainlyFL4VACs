import os
import json

def get_scene_file_map(data_root):
    """
    Maps Scene ID (int) -> List of filenames (strings)
    Example: 1 -> ['S001_K005_CAM_FRONT.npz', ...]
    """
    labels_dir = os.path.join(data_root, 'labels')
    if not os.path.exists(labels_dir):
        print(f"❌ Error: Labels directory not found at {labels_dir}")
        return {}

    # List all .npz files
    all_files = sorted([f.split('.')[0] for f in os.listdir(labels_dir) if f.endswith('.npz')])
    
    scene_map = {}
    for f in all_files:
        # Expected format: S001_K005_...
        try:
            scene_tag = f.split('_')[0]   # "S001"
            scene_idx = int(scene_tag[1:]) # 1
            
            if scene_idx not in scene_map: 
                scene_map[scene_idx] = []
            scene_map[scene_idx].append(f)
        except ValueError: 
            continue
            
    return scene_map

def load_federated_partition(data_root, split_json_path="splits_federated/federated_split.json"):
    """
    Loads the JSON and returns dictionaries containing LISTS OF FILES ready for the dataloader.
    Returns:
        client_data (dict): {0: {'train': [...], 'val': [...]}, ...}
        global_test_files (list): [...]
    """
    if not os.path.exists(split_json_path):
        raise FileNotFoundError(f"❌ Split file not found: {split_json_path}. Run create_splits.py first!")
    
    print(f"📄 Loading Federated Splits from: {split_json_path}")
    with open(split_json_path, 'r') as f: 
        splits = json.load(f)
    
    print("📂 Mapping files from disk...")
    scene_map = get_scene_file_map(data_root)
    
    client_data = {}
    
    # Build data for each client
    for client_key, info in splits['clients'].items():
        # client_key is "client_0", extract ID -> 0
        c_id = int(client_key.split('_')[1])
        
        # Translate scene indices to file lists
        train_files = []
        for s in info['train']: 
            if s in scene_map: train_files.extend(scene_map[s])
            
        val_files = []
        for s in info['val']:
            if s in scene_map: val_files.extend(scene_map[s])
            
        client_data[c_id] = {
            'train': train_files, 
            'val': val_files, 
            'metadata': info['metadata']
        }
        print(f"   👤 Client {c_id} ({info['metadata']['type']}): {len(train_files)} Train | {len(val_files)} Val files")

# Build Global Test Set
    global_test_files = []
    
    # ✅ FIX: Handle dictionary structure {"day": [...], "rain": [...], "night": [...]}
    test_structure = splits['global_test']
    
    if isinstance(test_structure, dict):
        # Flatten all categories into one global list
        all_test_scenes = []
        for category in test_structure:
            all_test_scenes.extend(test_structure[category])
    else:
        # Fallback if it's still a list
        all_test_scenes = test_structure

    # Map the flattened scenes to files
    for s in all_test_scenes:
        if s in scene_map: 
            global_test_files.extend(scene_map[s])
        
    print(f"   🌍 Global Test: {len(global_test_files)} files (from {len(all_test_scenes)} scenes)")
    
    return client_data, global_test_files