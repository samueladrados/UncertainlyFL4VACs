import hydra
from omegaconf import DictConfig
import os
import cv2
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

# INTELLIGENT MAPPING: Which radars to project onto which camera (based on overlap)
CAMERA_RADAR_MAPPING = {
    'CAM_FRONT':       ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT'],
    'CAM_FRONT_LEFT':  ['RADAR_FRONT_LEFT', 'RADAR_FRONT', 'RADAR_BACK_LEFT'],
    'CAM_FRONT_RIGHT': ['RADAR_FRONT_RIGHT', 'RADAR_FRONT', 'RADAR_BACK_RIGHT'],
    'CAM_BACK':        ['RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'],
    'CAM_BACK_LEFT':   ['RADAR_BACK_LEFT', 'RADAR_FRONT_LEFT'],
    'CAM_BACK_RIGHT':  ['RADAR_BACK_RIGHT', 'RADAR_FRONT_RIGHT']
}

CLASS_MAP = {
    'vehicle.car': 0, 'vehicle.emergency.police': 0, 'vehicle.emergency.ambulance': 0,
    'vehicle.truck': 1, 'vehicle.bus.rigid': 2, 'vehicle.bus.bendy': 2,
    'vehicle.trailer': 3, 'vehicle.construction': 4,
    'human.pedestrian.adult': 5, 'human.pedestrian.child': 5, 
    'human.pedestrian.construction_worker': 5, 'human.pedestrian.police_officer': 5,
    'human.pedestrian.wheelchair': 5, 'human.pedestrian.stroller': 5, 'human.pedestrian.personal_mobility': 5,
    'vehicle.motorcycle': 6, 'vehicle.bicycle': 7,
    'movable_object.trafficcone': 8, 'movable_object.barrier': 9,
    'movable_object.pushable_pullable': 10, 'movable_object.debris': 11,
    'static_object.bicycle_rack': 12, 'animal': 13
}


# --- GEOMETRIC FUNCTIONS ---
def map_pointcloud_to_image(nusc, pointsensor_token, camera_token, original_shape, min_dist=1.0):
    cam_data = nusc.get('sample_data', camera_token)
    pointsensor_data = nusc.get('sample_data', pointsensor_token)
    pcl_path = os.path.join(nusc.dataroot, pointsensor_data['filename'])
    
    # Robust Detection (Check channel name or file extension to avoid .pcd vs .bin crash)
    is_lidar = 'LIDAR' in pointsensor_data['channel'] or pcl_path.endswith('.bin')
    
    if is_lidar:
        pc = LidarPointCloud.from_file(pcl_path)
    else:
        pc = RadarPointCloud.from_file(pcl_path)

    # Feature Extraction (Before rotation)
    extra_feature = None
    if is_lidar: 
        extra_feature = pc.points[3, :] # Intensity
    elif pc.points.shape[0] >= 6: 
        extra_feature = pc.points[5, :] # RCS (Radar Cross Section)

    # Geometric Transformations (Sensor -> Ego -> Global -> Ego -> Camera)
    cs_record = nusc.get('calibrated_sensor', pointsensor_data['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))
    
    poserecord = nusc.get('ego_pose', pointsensor_data['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))
    
    cam_poserecord = nusc.get('ego_pose', cam_data['ego_pose_token'])
    pc.translate(-np.array(cam_poserecord['translation']))
    pc.rotate(Quaternion(cam_poserecord['rotation']).rotation_matrix.T)
    
    cam_cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    pc.translate(-np.array(cam_cs_record['translation']))
    pc.rotate(Quaternion(cam_cs_record['rotation']).rotation_matrix.T)

    depths = pc.points[2, :]
    intrinsic = np.array(cam_cs_record['camera_intrinsic'])
    points = view_points(pc.points[:3, :], intrinsic, normalize=True)

    # Filtering (Remove points behind camera or outside image bounds)
    H, W = original_shape
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 0)
    mask = np.logical_and(mask, points[0, :] < W)
    mask = np.logical_and(mask, points[1, :] > 0)
    mask = np.logical_and(mask, points[1, :] < H)

    if extra_feature is not None: return points[:, mask], depths[mask], extra_feature[mask]
    else: return points[:, mask], depths[mask], np.zeros(np.sum(mask))

def create_sparse_map(points, depths, feature_1, target_shape, original_shape):
    channels = 2 # [Depth, Feature]
    map_img = np.zeros(target_shape + (channels,), dtype=np.float32)
    
    scale_x = target_shape[1] / original_shape[1]
    scale_y = target_shape[0] / original_shape[0]
    
    u = np.clip((points[0, :] * scale_x).astype(int), 0, target_shape[1] - 1)
    v = np.clip((points[1, :] * scale_y).astype(int), 0, target_shape[0] - 1)
    
    # Channel 0: Depth
    map_img[v, u, 0] = depths
    # Channel 1: Feature (Intensity or RCS)
    if feature_1 is not None: map_img[v, u, 1] = feature_1
    
    return map_img

def draw_debug_image(img, points, color, radius, original_shape):
    overlay = img.copy()
    scale_x = img.shape[1] / original_shape[1]
    scale_y = img.shape[0] / original_shape[0]
    for i in range(points.shape[1]):
        u = int(points[0, i] * scale_x)
        v = int(points[1, i] * scale_y)
        u = max(0, min(u, img.shape[1] - 1))
        v = max(0, min(v, img.shape[0] - 1))
        cv2.circle(overlay, (u, v), radius, color, -1)
    return overlay

# --- MAIN LOOP (SCENE -> SAMPLE -> CAMERA) ---
def process_scene_batch(nusc, output_root, input_shape, original_shape, cameras_list):
    # Create output directories
    for subdir in ['images', 'lidar', 'radar', 'labels', 'debug_visualizations']:
        os.makedirs(os.path.join(output_root, subdir), exist_ok=True)

    print(f"🚀 Starting processing of {len(nusc.scene)} scenes...")

    # ITERATE THROUGH SCENES (S001, S002...)
    for scene_idx, scene in enumerate(tqdm(nusc.scene, desc="Scenes")):
        
        # Get first sample token of the scene
        sample_token = scene['first_sample_token']
        sample_idx = 0 # Counter for samples within the scene
        
        # WHILE LOOP TO TRAVERSE CHAINED SAMPLES
        while sample_token != "":
            sample = nusc.get('sample', sample_token)
            
            # ITERATE THROUGH THE 6 CAMERAS
            for cam_name in cameras_list:
                if cam_name not in sample['data']:
                    continue
                
                # --- GENERATE READABLE FILENAME ---
                # Format: S{Scene}_K{Sample}_{Camera}
                # Example: S001_K005_CAM_FRONT
                file_id = f"S{scene_idx:03d}_K{sample_idx:03d}_{cam_name}"
                
                cam_token = sample['data'][cam_name]
                
                # 1. IMAGE PROCESSING
                img_path = os.path.join(nusc.dataroot, nusc.get('sample_data', cam_token)['filename'])
                
                # Check existence (in case of partial download vs full dataset)
                if not os.path.exists(img_path):
                    continue 

                img_orig = cv2.imread(img_path)
                img_resized = cv2.resize(img_orig, (input_shape[1], input_shape[0]))
                cv2.imwrite(os.path.join(output_root, 'images', f"{file_id}.jpg"), img_resized)
                
                img_debug = img_resized.copy()
                
                # 2. LIDAR PROCESSING (LIDAR_TOP projected onto THIS camera)
                pts_l, d_l, int_l = map_pointcloud_to_image(nusc, sample['data']['LIDAR_TOP'], cam_token, original_shape)
                lidar_map = create_sparse_map(pts_l, d_l, int_l, input_shape, original_shape)
                np.savez_compressed(os.path.join(output_root, 'lidar', f"{file_id}.npz"), data=lidar_map)
                
                img_debug = draw_debug_image(img_debug, pts_l, (0, 255, 0), 1, original_shape)
                
                # 3. RADAR PROCESSING (Using Intelligent Mapping)
                rad_accum = np.zeros((input_shape[0], input_shape[1], 2), dtype=np.float32)
                
                # Select only relevant radars for THIS camera
                relevant_radars = CAMERA_RADAR_MAPPING.get(cam_name, [])
                
                for r_chan in relevant_radars:
                    if r_chan in sample['data']:
                        pts_r, d_r, rcs_r = map_pointcloud_to_image(nusc, sample['data']['LIDAR_TOP'], cam_token, original_shape)
                        curr_map = create_sparse_map(pts_r, d_r, rcs_r, input_shape, original_shape)
                        # Fusion: Keep maximum value (overlap handling)
                        rad_accum = np.maximum(rad_accum, curr_map)
                        
                        img_debug = draw_debug_image(img_debug, pts_r, (0, 0, 255), 2, original_shape)
                        
                np.savez_compressed(os.path.join(output_root, 'radar', f"{file_id}.npz"), data=rad_accum)
                
                # 4. LABEL PROCESSING (Ground Truth)
                _, boxes, _ = nusc.get_sample_data(cam_token, box_vis_level=0)
                final_boxes, final_classes = [], []
                
                cam_sd = nusc.get('sample_data', cam_token)
                cs_rec = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
                intrinsic = np.array(cs_rec['camera_intrinsic'])

                for box in boxes:
                    detected_class = -1
                    for k, v in CLASS_MAP.items():
                        if box.name.startswith(k):
                            detected_class = v
                            break
                        
                        if detected_class != -1:
                            corners = view_points(box.corners(), intrinsic, normalize=True)[:2, :]
                            xmin = np.min(corners[0, :])
                            xmax = np.max(corners[0, :])
                            ymin = np.min(corners[1, :])
                            ymax = np.max(corners[1, :])
                            
                            # Scale factors
                            sx = input_shape[1] / original_shape[1]
                            sy = input_shape[0] / original_shape[0]
                            
                            xmin, xmax = xmin * sx, xmax * sx
                            ymin, ymax = ymin * sy, ymax * sy
                            
                            # Clip to new image boundaries
                            xmin, ymin = max(0, xmin), max(0, ymin)
                            xmax, ymax = min(input_shape[1], xmax), min(input_shape[0], ymax)
                            
                            if (xmax - xmin) > 1 and (ymax - ymin) > 1:
                                # Convert to [cx, cy, w, h]
                                final_boxes.append([(xmin+xmax)/2, (ymin+ymax)/2, xmax-xmin, ymax-ymin])
                                final_classes.append(detected_class)
                                
                                # Draw debug box (Yellow)
                                cv2.rectangle(img_debug, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 255), 1)
                
                np.savez_compressed(
                    os.path.join(output_root, 'labels', f"{file_id}.npz"), 
                    boxes=np.array(final_boxes, dtype=np.float32), 
                    classes=np.array(final_classes, dtype=np.int32)
                )
                
                # Save Debug Image (Optional: Consumes space, comment out if not needed)
                cv2.imwrite(os.path.join(output_root, 'debug_visualizations', f"{file_id}_debug.jpg"), img_debug)

            # --- ADVANCE TO NEXT SAMPLE ---
            sample_token = sample['next']
            sample_idx += 1


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    raw_root = cfg.data.raw_root
    output_root = cfg.data.root
    version = cfg.data.version
    
    input_shape = tuple(cfg.data.input_shape[:2]) 
    original_shape = tuple(cfg.data.original_shape)
    cameras_list = cfg.data.cameras
    
    print(f"📂 Loading NuScenes from: {raw_root}")
    print(f"💾 Saving to: {output_root}")
    print(f"📏 Shapes -> Original: {original_shape}, Target: {input_shape}")

    try:
        nusc = NuScenes(version=version, dataroot=raw_root, verbose=True)
    except Exception as e:
        print(f"❌ Error loading NuScenes: {e}")
        return

    process_scene_batch(nusc, output_root, input_shape, original_shape, cameras_list)
    print("\n✅ Preprocessing Complete!")

if __name__ == "__main__":
    main()