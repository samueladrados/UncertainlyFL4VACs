import os
import tensorflow as tf
import numpy as np
import cv2

# --- ANCHOR BOX LOGIC (Integrated) ---
class AnchorBox:
    def __init__(self, input_shape=(320, 320), min_size=320):
        self.input_shape = input_shape
        self.min_size = min_size
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1/3, 2/3]]
        self.num_anchors = len(self.aspect_ratios) * len(self.scales)
        self.strides = [4, 8, 16, 32, 64]
        self.areas = [16**2, 32**2, 64**2, 128**2, 256**2]
        self.anchors = self._generate_anchors()

    def _generate_anchors(self):
        anchor_list = []
        for stride, area in zip(self.strides, self.areas):
            base_anchor_size = np.sqrt(area)
            y_centers = np.arange(stride / 2, self.input_shape[0], stride)
            x_centers = np.arange(stride / 2, self.input_shape[1], stride)
            x_centers, y_centers = np.meshgrid(x_centers, y_centers)
            
            widths, heights = [], []
            for scale in self.scales:
                for ratio in self.aspect_ratios:
                    w = np.sqrt(area / ratio)
                    h = w * ratio
                    widths.append(w * scale)
                    heights.append(h * scale)
            
            centers = np.stack([x_centers, y_centers], axis=-1).reshape(-1, 2)
            wh = np.stack([np.array(widths), np.array(heights)], axis=-1)
            
            centers = np.repeat(centers, self.num_anchors, axis=0)
            wh = np.tile(wh, (len(x_centers.flatten()), 1))
            anchor_list.append(np.concatenate([centers, wh], axis=-1))
            
        return np.concatenate(anchor_list, axis=0).astype(np.float32)

    def encode(self, gt_boxes, gt_classes):
        targets_cls = np.zeros((self.anchors.shape[0],), dtype=np.int32)
        targets_box = np.zeros((self.anchors.shape[0], 4), dtype=np.float32)
        
        if len(gt_boxes) == 0:
            return np.zeros((self.anchors.shape[0], 14), dtype=np.float32), targets_box

        # IoU Calculation
        anchors_min = self.anchors[:, :2] - self.anchors[:, 2:] / 2
        anchors_max = self.anchors[:, :2] + self.anchors[:, 2:] / 2
        anchors_area = self.anchors[:, 2] * self.anchors[:, 3]
        
        gt_min = gt_boxes[:, :2] - gt_boxes[:, 2:] / 2
        gt_max = gt_boxes[:, :2] + gt_boxes[:, 2:] / 2
        gt_area = gt_boxes[:, 2] * gt_boxes[:, 3]
        
        inter_min = np.maximum(anchors_min[:, None, :], gt_min[None, :, :])
        inter_max = np.minimum(anchors_max[:, None, :], gt_max[None, :, :])
        inter_dim = np.maximum(0, inter_max - inter_min)
        inter_area = inter_dim[:, :, 0] * inter_dim[:, :, 1]
        iou = inter_area / (anchors_area[:, None] + gt_area[None, :] - inter_area + 1e-6)

        max_iou_ind = np.argmax(iou, axis=1)
        max_iou = np.max(iou, axis=1)
        
        # Assign Targets: <0.4 BG, 0.4-0.5 Ignore, >0.5 Object
        targets_cls[max_iou < 0.4] = 0 
        targets_cls[(max_iou >= 0.4) & (max_iou < 0.5)] = -1
        pos_mask = max_iou >= 0.5
        targets_cls[pos_mask] = gt_classes[max_iou_ind[pos_mask]] + 1 # Class 1..14

        assigned_boxes = gt_boxes[max_iou_ind]
        targets_box[:, 0] = (assigned_boxes[:, 0] - self.anchors[:, 0]) / self.anchors[:, 2]
        targets_box[:, 1] = (assigned_boxes[:, 1] - self.anchors[:, 1]) / self.anchors[:, 3]
        targets_box[:, 2] = np.log(np.maximum(assigned_boxes[:, 2] / self.anchors[:, 2], 1e-6))
        targets_box[:, 3] = np.log(np.maximum(assigned_boxes[:, 3] / self.anchors[:, 3], 1e-6))
        targets_box[~pos_mask] = 0
        
        # One-Hot Encoding (14 Classes)
        final_cls_onehot = np.zeros((self.anchors.shape[0], 14), dtype=np.float32)
        pos_indices = np.where(targets_cls > 0)[0]
        if len(pos_indices) > 0:
            final_cls_onehot[pos_indices, targets_cls[pos_indices] - 1] = 1.0
            
        return final_cls_onehot, targets_box

# --- DATALOADER ---
class DataLoaderGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_root, specific_files=None, batch_size=4, input_shape=(320, 320)):
        """
        Args:
            data_root: Path to the dataset root.
            specific_files: EXACT LIST of filenames (without extension) to load.
                            This list comes from fl_utils.load_federated_partition().
        """
        self.data_root = data_root
        self.batch_size = batch_size
        
        # ✅ CRITICAL: min_size=0 ensures we don't filter out valid objects
        self.encoder = AnchorBox(input_shape=input_shape, min_size=0) 

        # FEDERATED MODE: We use exactly the files provided
        if specific_files is not None:
            self.tokens = specific_files
        else:
            # Fallback (only if you run it without federation)
            print("⚠️ WARNING: Loading ALL files from directory (Non-Federated Mode)")
            self.tokens = sorted([f.split('.')[0] for f in os.listdir(os.path.join(data_root, 'labels')) if f.endswith('.npz')])
            
        self.indexes = np.arange(len(self.tokens))

    def __len__(self):
        return int(np.floor(len(self.tokens) / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X_img, X_lid, X_rad, Y_cls, Y_box = [], [], [], [], []

        for k in indexes:
            token = self.tokens[k]
            
            # Load Data paths
            img_path = f"{self.data_root}/images/{token}.jpg"
            lid_path = f"{self.data_root}/lidar/{token}.npz"
            rad_path = f"{self.data_root}/radar/{token}.npz"
            lbl_path = f"{self.data_root}/labels/{token}.npz"
            
            # Basic safety check
            if not os.path.exists(lbl_path): continue
            
            # Read files
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            lid = np.load(lid_path)['data']
            rad = np.load(rad_path)['data']
            lbl = np.load(lbl_path)
            
            # Encode Targets
            yc, yb = self.encoder.encode(lbl['boxes'], lbl['classes'])
            
            X_img.append(img); X_lid.append(lid); X_rad.append(rad)
            Y_cls.append(yc); Y_box.append(yb)

        return (np.array(X_img), np.array(X_lid), np.array(X_rad)), (np.array(Y_cls), np.array(Y_box))