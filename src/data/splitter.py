import numpy as np
import json
import os
import random

# --- CONFIGURATION ---
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
OUTPUT_DIR = "splits_federated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_federated_splits():
    # 1. Define Ranges (NuScenes v1.0-trainval)
    rain_indices = list(range(354, 391)) + list(range(450, 507)) + \
                   list(range(622, 639)) + list(range(655, 702))
    night_indices = list(range(751, 850))
    
    all_indices = set(range(850))
    rain_set = set(rain_indices)
    night_set = set(night_indices)
    day_indices = list(all_indices - rain_set - night_set)
    
    print(f"☀️ Total Day: {len(day_indices)} | 🌧️ Total Rain: {len(rain_indices)} | 🌙 Total Night: {len(night_indices)}")
    
    # 2. GLOBAL TEST SET (Stratified 20% original)
    np.random.seed(42) 
    
    test_day = np.random.choice(day_indices, size=int(len(day_indices)*0.2), replace=False)
    test_rain = np.random.choice(rain_indices, size=int(len(rain_indices)*0.2), replace=False)
    test_night = np.random.choice(night_indices, size=int(len(night_indices)*0.2), replace=False)
    
    
    # 3. Remaining Pools for Clients
    # Convert to numpy array to use array_split easily
    pool_day = np.array(list(set(day_indices) - set(test_day)))
    pool_rain = np.array(list(set(rain_indices) - set(test_rain)))
    pool_night = np.array(list(set(night_indices) - set(test_night)))
    
    # Shuffle randomly before splitting
    np.random.shuffle(pool_day)
    np.random.shuffle(pool_rain)
    np.random.shuffle(pool_night)

    # ==============================================================================
    # --- ✂️ DATA REDUCTION (1/10 - 10%) ---
    # We cut arrays to keep only 10% of the available data for SUPER FAST testing.
    # ==============================================================================
    print("\n✂️ REDUCING DATASET TO 1/10 (10%) FOR DEBUGGING...")
    
    REDUCTION_FACTOR = 0.10 # 1/10
    
    # Reduce Global Test
    test_day = test_day[:max(2, int(len(test_day) * REDUCTION_FACTOR))]
    test_rain = test_rain[:max(2, int(len(test_rain) * REDUCTION_FACTOR))]
    test_night = test_night[:max(2, int(len(test_night) * REDUCTION_FACTOR))]
    
    global_test = np.concatenate([test_day, test_rain, test_night])
    print(f"   -> Global Test Size: {len(global_test)} (Day: {len(test_day)}, Rain: {len(test_rain)}, Night: {len(test_night)})")
    
    # Reduce Training Pools
    # We enforce a minimum number of scenes to prevent crashes due to empty lists
    pool_day = pool_day[:max(45, int(len(pool_day) * REDUCTION_FACTOR))] 
    pool_rain = pool_rain[:max(4, int(len(pool_rain) * REDUCTION_FACTOR))]
    pool_night = pool_night[:max(4, int(len(pool_night) * REDUCTION_FACTOR))]
    
    print(f"   -> Reduced Pool Day: {len(pool_day)}")
    print(f"   -> Reduced Pool Rain: {len(pool_rain)}")
    print(f"   -> Reduced Pool Night: {len(pool_night)}")
    # ==============================================================================

    # --- DISTRIBUTION LOGIC ---
    # We divide resources equally to avoid quantity skew
    
    # A. DAY POOL MANAGEMENT
    # We reserve scenes as "filler" for Rain/Night clients.
    
    filler_count = 40
    # Safety check: if the reduction left us with very few Day scenes
    if len(pool_day) < 50: 
        filler_count = int(len(pool_day) * 0.4) # Use 40% for filler
        print(f"⚠️ Warning: Low data regime. Filler reduced to {filler_count} scenes.")

    filler_day = pool_day[:filler_count]
    main_day = pool_day[filler_count:]

    # Split the main day pool into 6 EQUAL chunks for clients 0-5
    chunks_day = np.array_split(main_day, 6)
    
    # Split the filler day pool into 4 EQUAL chunks for clients 6-9
    chunks_filler = np.array_split(filler_day, 4)

    # B. RAIN POOL (Split into 2 equal chunks for clients 6-7)
    chunks_rain = np.array_split(pool_rain, 2)
    
    # C. NIGHT POOL (Split into 2 equal chunks for clients 8-9)
    chunks_night = np.array_split(pool_night, 2)
    
    clients = {}
    
    for i in range(10):
        client_scenes = []
        c_type = ""
        
        if i < 6: # Day Clients (0-5)
            # They take their equal chunk of day scenes
            client_scenes.extend(chunks_day[i].tolist())
            c_type = "Day"
            
        elif i < 8: # Rain Clients (6-7)
            idx = i - 6
            # Add equal chunk of Rain + equal chunk of Filler Day
            if len(chunks_rain) > idx: client_scenes.extend(chunks_rain[idx].tolist())
            if len(chunks_filler) > idx: client_scenes.extend(chunks_filler[idx].tolist())
            c_type = "Rain"
            
        else: # Night Clients (8-9)
            idx = i - 8
            # Add equal chunk of Night + equal chunk of Filler Day
            if len(chunks_night) > idx: client_scenes.extend(chunks_night[idx].tolist())
            if len(chunks_filler) > 2 + idx: client_scenes.extend(chunks_filler[2 + idx].tolist())
            c_type = "Night"
        
        # Internal Split Train/Val (80/20)
        client_scenes = [int(x) for x in client_scenes]
        np.random.shuffle(client_scenes)
        
        split_pt = int(len(client_scenes) * 0.8)
        # Ensure at least 1 val scene if possible
        if split_pt == len(client_scenes) and len(client_scenes) > 1:
            split_pt -= 1
        
        clients[f"client_{i}"] = {
            "train": client_scenes[:split_pt],
            "val": client_scenes[split_pt:],
            "metadata": {"type": c_type, "total_scenes": len(client_scenes)}
        }
        
        print(f"👤 Client {i} ({c_type}): {len(client_scenes)} scenes")

    # Reconstruct Global Test Categorization
    data = {
        "global_test": {
            "day": [int(x) for x in global_test if x in day_indices],
            "rain": [int(x) for x in global_test if x in rain_indices],
            "night": [int(x) for x in global_test if x in night_indices]
        },
        "clients": clients
    }
    
    # Write to new file
    json_path = os.path.join(OUTPUT_DIR, "federated_split.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
        
    print(f"\n✅ REDUCED (1/10) JSON generated at {json_path}")

if __name__ == "__main__":
    create_federated_splits()