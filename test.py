import os
import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

# --- CONFIGURACIÓN ---
NUSC_ROOT = "/mnt/c/Users/USUARIO/Desktop/enviroments/Paper_NOUS/datasets/nuscenes"
OUTPUT_ROOT = "/mnt/c/Users/USUARIO/Desktop/enviroments/Paper_NOUS/datasets/nuscenes_alignment_test"
INPUT_SHAPE = (320, 320)
ORIGINAL_SHAPE = (900, 1600)
RADARS_TO_FUSE = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT']

# --- FUNCIONES MATEMÁTICAS (Idénticas al anterior) ---
def map_pointcloud_to_image(nusc, pointsensor_token, camera_token, min_dist=1.0):
    cam_data = nusc.get('sample_data', camera_token)
    pointsensor_data = nusc.get('sample_data', pointsensor_token)
    pcl_path = os.path.join(nusc.dataroot, pointsensor_data['filename'])
    
    # --- CORRECCIÓN AQUÍ: Detección robusta del tipo de sensor ---
    # Usamos el nombre del canal ('LIDAR_TOP') o la extensión del archivo
    is_lidar = 'LIDAR' in pointsensor_data['channel'] or pcl_path.endswith('.bin')
    
    try:
        if is_lidar:
            # Intenta cargar sweeps para LiDAR (más denso)
            pc, _ = LidarPointCloud.from_file_multisweep(nusc, nusc.get('sample', pointsensor_data['sample_token']), 
                                                         pointsensor_data['channel'], pointsensor_data['channel'], nsweeps=10)
        else:
            # Intenta cargar sweeps para Radar
            pc, _ = RadarPointCloud.from_file_multisweep(nusc, nusc.get('sample', pointsensor_data['sample_token']), 
                                                         pointsensor_data['channel'], pointsensor_data['channel'], nsweeps=5)
    except:
        # Fallback si no hay carpeta sweeps (Tu caso con Part 1)
        if is_lidar:
            pc = LidarPointCloud.from_file(pcl_path) # LidarPointCloud lee .bin correctamente
        else:
            pc = RadarPointCloud.from_file(pcl_path)

    # --- El resto de la función sigue igual ---
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

    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 0)
    mask = np.logical_and(mask, points[0, :] < ORIGINAL_SHAPE[1])
    mask = np.logical_and(mask, points[1, :] > 0)
    mask = np.logical_and(mask, points[1, :] < ORIGINAL_SHAPE[0])

    return points[:, mask], depths[mask]

def draw_points_on_image(img_bgr, points, color=(0, 255, 0), radius=1):
    overlay = img_bgr.copy()
    scale_x = img_bgr.shape[1] / ORIGINAL_SHAPE[1]
    scale_y = img_bgr.shape[0] / ORIGINAL_SHAPE[0]

    for i in range(points.shape[1]):
        u = int(points[0, i] * scale_x)
        v = int(points[1, i] * scale_y)
        u = max(0, min(u, img_bgr.shape[1] - 1))
        v = max(0, min(v, img_bgr.shape[0] - 1))
        cv2.circle(overlay, (u, v), radius, color, -1)
    return overlay

# --- LÓGICA DE UN SOLO SAMPLE ---
def process_single_sample(nusc):
    os.makedirs(f"{OUTPUT_ROOT}/debug_visualizations", exist_ok=True)

    # 1. OBTENER SOLO EL PRIMER SAMPLE DE LA PRIMERA ESCENA
    print("🔍 Buscando la primera muestra de la primera escena...")
    first_scene = nusc.scene[0]
    first_sample_token = first_scene['first_sample_token']
    sample = nusc.sample[255]
    
    print(f"🎯 Procesando Sample: {sample['token']}")
    print(f"   Escena: {first_scene['name']}")
    print(f"   Descripción: {first_scene['description']}")

    cam_token = sample['data']['CAM_FRONT']
    
    # 2. CARGAR IMAGEN ORIGINAL
    img_path = os.path.join(nusc.dataroot, nusc.get('sample_data', cam_token)['filename'])
    if not os.path.exists(img_path):
        print(f"❌ Error: No encuentro la imagen en {img_path}")
        return

    img_orig = cv2.imread(img_path)
    img_debug = cv2.resize(img_orig, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
    
    # 3. PROYECTAR Y PINTAR LIDAR (VERDE)
    print("   - Proyectando LiDAR...")
    pts_lidar, _ = map_pointcloud_to_image(nusc, sample['data']['LIDAR_TOP'], cam_token)
    img_debug = draw_points_on_image(img_debug, pts_lidar, color=(0, 255, 0), radius=1)

    # 4. PROYECTAR Y PINTAR RADARES FUSIONADOS (ROJO)
    print("   - Proyectando y Fusionando Radares...")
    for r_chan in RADARS_TO_FUSE:
        if r_chan in sample['data']:
            pts_r, _ = map_pointcloud_to_image(nusc, sample['data'][r_chan], cam_token)
            img_debug = draw_points_on_image(img_debug, pts_r, color=(0, 0, 255), radius=2)

    # 5. GUARDAR RESULTADO
    output_path = f"{OUTPUT_ROOT}/debug_visualizations/TEST_ALIGNMENT.jpg"
    cv2.imwrite(output_path, img_debug)
    
    print("-" * 30)
    print(f"✅ ¡ÉXITO! Imagen guardada en:")
    print(f"   {output_path}")
    print("   Ábrela para verificar que los puntos coinciden con los coches.")

if __name__ == "__main__":
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=False)
    process_single_sample(nusc)