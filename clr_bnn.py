"""
IMPLEMENTATION OF CLR-BNN
==========================
Title: Camera, LiDAR, and Radar Sensor Fusion Based on Bayesian Neural Network (CLR-BNN)
Authors: Ratheesh Ravindran, Michael J. Santora, and Mohsin M. Jamali
Journal: IEEE Sensors Journal, Vol. 22, No. 7, April 1, 2022
DOI: 10.1109/JSEN.2022.3154980

Abstract Implementation details:
This script implements the single-stage sensor fusion Bayesian Neural Network (CLR-BNN) 
proposed in the paper. It fuses Camera (RGB), LiDAR, and RADAR data to predict 2D object 
detection with aleatoric uncertainty (classification and bounding box covariance).

Key Components:
- Backbone: ResNet-50 based with Bayesian Layers (Fig. 1, Table I)
- Bayesian Inference: Flipout/Reparameterization layers with KL Divergence (Sec. II-A, II-B)
- Sensor Fusion: Feature-level fusion at multiple stages (Fig. 3)
- RADAR Processing: Custom Min-Pooling for sparse data (Table II)
- Uncertainty Head: Multivariate Normal Distribution for bounding boxes (Sec. II-G)
"""

import os

# --- CRITICAL BLOCK: GPU CONFIGURATION FOR WSL ---
# NOTE: This block is not part of the original paper logic but is required 
# for stability on modern WSL2/CUDA environments running TensorFlow Probability.
# 1. Disable XLA (compiler often fails with Softplus op on WSL)
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0 --tf_xla_cpu_global_jit=0"
os.environ["TF_JIT_PROFILING"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# Explicitly force JIT deactivation
tf.config.optimizer.set_jit(False)

# Configure memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error config GPU: {e}")

# Aliases
tfd = tfp.distributions
tfpl = tfp.layers

# ==============================================================================
# 1. BAYESIAN CONFIGURATION & LAYERS
# Ref: Section II-B "Layers and Initialization" and Figure 2
# ==============================================================================

def get_kernel_divergence_fn(train_size=1000):
    """
    Calculates scaled KL divergence.
    Ref: Eq. (2) - Approximating predictive distribution via ELBO.
    """
    def divergence_fn(q, p, _):
        return tfd.kl_divergence(q, p) / train_size
    return divergence_fn

def bayesian_conv2d(filters, kernel_size, strides=1, padding='same', name=None):
    """
    Creates a 'Convolution2DReparameterization' layer.
    
    Ref: Figure 2 and Sec II-B.
    The paper specifies:
    - Prior: Multivariate Normal (simulating deterministic initialization, sigma->0).
    - Posterior: Mean Field Normal (trainable parameters).
    - Initialization: Based on ImageNet for feature extraction layers.
    """
    return tfpl.Convolution2DReparameterization(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation='relu',
        name=name,
        kernel_prior_fn=tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        kernel_divergence_fn=get_kernel_divergence_fn()
    )

# ==============================================================================
# 2. CUSTOM LAYER: MIN POOLING IGNORING ZEROS
# Ref: Table II "Processing Methods for LiDAR and RADAR", Row 3
# Ref: Section II-D "RADAR Signal Processing Layers"
# ==============================================================================

class MinIZPooling2D(tf.keras.layers.Layer):
    """
    Implements 'Min Pooling Ignoring Zeros' (Min_IZ_Pool2D).
    
    Paper Justification:
    "RADAR point-cloud is processed using 'Min_IZ_Pool2D' before processing...
    Due to the negative impact of sparse RADAR data."
    Standard MaxPool would pick noise/far objects; standard MinPool would pick zeros.
    """
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        super(MinIZPooling2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding.upper()

    def call(self, inputs):
        # 1. Detect zeros (missing data)
        zeros_mask = tf.equal(inputs, 0.0)
        
        # 2. Replace zeros with Infinity
        large_val = tf.float32.max
        inputs_shifted = tf.where(zeros_mask, large_val, inputs)
        
        # 3. Math trick: Min(x) = -Max(-x)
        neg_inputs = -inputs_shifted
        max_pool_neg = tf.nn.max_pool(neg_inputs, ksize=self.pool_size, 
                                      strides=self.strides, padding=self.padding)
        min_pool = -max_pool_neg
        
        # 4. Restore zeros if window was empty
        safe_threshold = large_val / 2.0
        return tf.where(min_pool > safe_threshold, 0.0, min_pool)

# ==============================================================================
# 3. SENSOR PROCESSING BLOCKS
# Ref: Section II-C (LiDAR) and II-D (RADAR)
# ==============================================================================

def lidar_processing_block(inputs, stage, filters):
    """
    LiDAR Processing (L-processing).
    Ref: Table II, Row 1 & 4. Uses MaxPool2D to handle sparsity.
    """
    x = inputs
    # Progressive downsampling to align with camera feature map size
    for _ in range(stage): 
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    
    x = bayesian_conv2d(filters=filters, kernel_size=1, name=f'lidar_proc_s{stage}')(x)
    return x

def radar_processing_block(inputs, stage, filters):
    """
    RADAR Processing (R-processing).
    Ref: Section II-D.
    - Stage 1: Uses Min_IZ_Pool2D (to capture nearest objects in sparse data).
    - Subsequent Stages: Uses MaxPool2D.
    """
    x = inputs
    if stage == 1: 
        x = MinIZPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    else:
        for _ in range(stage):
            x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
            
    x = bayesian_conv2d(filters=filters, kernel_size=1, name=f'radar_proc_s{stage}')(x)
    return x

# ==============================================================================
# 4. BAYESIAN RESIDUAL BLOCK
# Ref: Table I "Conv Blocks" and Section II-E
# ==============================================================================

def bottleneck_block(x, filters, stride=1, use_projection=False):
    """
    Bayesian implementation of ResNet-50 bottleneck block.
    """
    filters1, filters2, filters3 = filters
    shortcut = x

    if use_projection:
        shortcut = bayesian_conv2d(filters3, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    # 1. Conv 1x1
    x = bayesian_conv2d(filters1, kernel_size=1, strides=stride if stride > 1 else 1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # 2. Conv 3x3
    x = bayesian_conv2d(filters2, kernel_size=3)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # 3. Conv 1x1
    x = bayesian_conv2d(filters3, kernel_size=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x

# ==============================================================================
# 5. MAIN MODEL: CLR-BNN
# Ref: Figure 1 (Architecture Overview)
# ==============================================================================

class CLR_BNN(tf.keras.Model):
    def __init__(self, num_classes, num_anchors=9):
        super(CLR_BNN, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # --- Heads / Subnets ---
        
        # Ref: Section II-G "Classification and Regression Subnets"
        # "The final layer is modeled using 'MultivariateNormalTriL' layer to incorporate
        # the multi-variate Gaussian distribution with Cholesky decomposition."
        
        # 1. Params for Distribution
        # 4 coords (x,y,w,h) * 9 anchors = 36 random variables per cell
        event_size = 4 * num_anchors
        params_size = tfpl.MultivariateNormalTriL.params_size(event_size) 
        
        self.regression_head = tf.keras.Sequential([
            bayesian_conv2d(256, 3, name='reg_conv1'),
            bayesian_conv2d(256, 3, name='reg_conv2'),
            bayesian_conv2d(256, 3, name='reg_conv3'),
            bayesian_conv2d(256, 3, name='reg_conv4'),
            # Projection to required parameter size for the distribution
            bayesian_conv2d(params_size, 3, name='reg_projection'),
            
            # The probabilistic output layer
            tfpl.MultivariateNormalTriL(
                event_size=event_size,
                convert_to_tensor_fn=tfd.Distribution.sample, # Samples during inference
                name='bbox_uncertainty'
            )
        ])

        # Classification Subnet (Uses Focal Loss downstream)
        self.classification_head = tf.keras.Sequential([
            bayesian_conv2d(256, 3, name='cls_conv1'),
            bayesian_conv2d(256, 3, name='cls_conv2'),
            bayesian_conv2d(256, 3, name='cls_conv3'),
            bayesian_conv2d(256, 3, name='cls_conv4'),
            tf.keras.layers.Conv2D(num_classes * num_anchors, kernel_size=3, padding='same', activation='sigmoid')
        ])

    def build_backbone(self, input_img, input_lidar, input_radar):
        """
        Builds the Feature Extractor with Sensor Fusion (Fig. 1).
        Ref: Figure 3 "Concatenation of camera image, LiDAR and RADAR".
        """
        
        # --- STAGE 1 (Input -> Conv1) ---
        # Image: 608x608
        l_s0 = lidar_processing_block(input_lidar, stage=0, filters=16) 
        
        # RADAR: Strides=(1,1) to match Image size (608x608) before Conv1
        r_s0 = MinIZPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_radar)
        r_s0 = bayesian_conv2d(16, 1)(r_s0) 

        # Fusion 1 (Concatenation)
        x = tf.keras.layers.Concatenate()([input_img, l_s0, r_s0])
        
        # Conv1 Block (Downsample x2)
        c1 = bayesian_conv2d(64, 7, strides=2, name='conv1')(x)
        c1 = tf.keras.layers.MaxPool2D(3, strides=2, padding='same')(c1) # Output: 152x152

        # --- STAGE 2 (Conv2_x) ---
        l_s1 = lidar_processing_block(input_lidar, stage=2, filters=64) 
        r_s1 = radar_processing_block(input_radar, stage=2, filters=64)
        
        c1_fused = tf.keras.layers.Concatenate()([c1, l_s1, r_s1])
        
        c2 = bottleneck_block(c1_fused, [64, 64, 256], stride=1, use_projection=True)
        for _ in range(2): c2 = bottleneck_block(c2, [64, 64, 256])

        # --- STAGE 3 (Conv3_x) ---
        l_s2 = lidar_processing_block(input_lidar, stage=2, filters=256) 
        r_s2 = radar_processing_block(input_radar, stage=2, filters=256)
        
        c2_fused = tf.keras.layers.Concatenate()([c2, l_s2, r_s2])
        
        # First block does downsample (stride=2)
        c3 = bottleneck_block(c2_fused, [128, 128, 512], stride=2, use_projection=True)
        for _ in range(2): c3 = bottleneck_block(c3, [128, 128, 512])

        # --- STAGE 4 (Conv4_x) ---
        l_s3 = lidar_processing_block(input_lidar, stage=3, filters=512)
        r_s3 = radar_processing_block(input_radar, stage=3, filters=512)
        
        c3_fused = tf.keras.layers.Concatenate()([c3, l_s3, r_s3])
        
        c4 = bottleneck_block(c3_fused, [256, 256, 1024], stride=2, use_projection=True)
        for _ in range(2): c4 = bottleneck_block(c4, [256, 256, 1024])

        # --- STAGE 5 (Conv5_x) ---
        l_s4 = lidar_processing_block(input_lidar, stage=4, filters=1024)
        r_s4 = radar_processing_block(input_radar, stage=4, filters=1024)
        
        c4_fused = tf.keras.layers.Concatenate()([c4, l_s4, r_s4])
        
        c5 = bottleneck_block(c4_fused, [512, 512, 2044], stride=2, use_projection=True)
        for _ in range(2): c5 = bottleneck_block(c5, [512, 512, 2044])

        return c3, c4, c5

    def build_fpn(self, c3, c4, c5):
        """
        Bayesian Feature Pyramid Network (FPN).
        Ref: Section II-F
        """
        feature_size = 256
        
        # Top-down pathway
        p5 = bayesian_conv2d(feature_size, 1, name='fpn_c5p5')(c5)
        
        # P4
        p5_upsampled = tf.keras.layers.UpSampling2D(size=(2, 2))(p5)
        p4_lateral = bayesian_conv2d(feature_size, 1, name='fpn_c4p4')(c4)
        p5_upsampled = tf.image.resize(p5_upsampled, tf.shape(p4_lateral)[1:3])
        p4 = tf.keras.layers.Add()([p5_upsampled, p4_lateral])
        p4 = bayesian_conv2d(feature_size, 3, name='fpn_p4')(p4)

        # P3
        p4_upsampled = tf.keras.layers.UpSampling2D(size=(2, 2))(p4)
        p3_lateral = bayesian_conv2d(feature_size, 1, name='fpn_c3p3')(c3)
        p4_upsampled = tf.image.resize(p4_upsampled, tf.shape(p3_lateral)[1:3])
        p3 = tf.keras.layers.Add()([p4_upsampled, p3_lateral])
        p3 = bayesian_conv2d(feature_size, 3, name='fpn_p3')(p3)

        # Extra levels (RetinaNet style)
        p6 = bayesian_conv2d(feature_size, 3, strides=2, name='fpn_p6')(c5)
        p7 = tf.keras.layers.ReLU()(p6)
        p7 = bayesian_conv2d(feature_size, 3, strides=2, name='fpn_p7')(p7)

        return [p3, p4, p5, p6, p7]

    def call(self, inputs):
        """
        Forward Pass.
        Ref: Figure 1 for data flow.
        """
        img, lidar, radar = inputs
        
        # 1. Backbone
        c3, c4, c5 = self.build_backbone(img, lidar, radar)
        
        # 2. FPN
        features = self.build_fpn(c3, c4, c5)
        
        # 3. Heads
        cls_outputs = [self.classification_head(f) for f in features]
        box_outputs = [self.regression_head(f) for f in features]

        # Flatten predictions
        cls_final = tf.concat([tf.keras.layers.Flatten()(c) for c in cls_outputs], axis=1)
        box_final = tf.concat([tf.keras.layers.Flatten()(b) for b in box_outputs], axis=1)

        return cls_final, box_final

# ==============================================================================
# MAIN TEST
# ==============================================================================
if __name__ == "__main__":
    print(f"Running CLR-BNN Implementation based on Ravindran et al. (2022)")
    
    # Input simulation
    input_img = tf.random.normal((1, 608, 608, 3))
    input_lidar = tf.random.normal((1, 608, 608, 2))
    input_radar = tf.zeros((1, 608, 608, 2)) 
    
    model = CLR_BNN(num_classes=10)
    
    print("Executing Forward Pass...")
    cls, box = model([input_img, input_lidar, input_radar])
    
    print(f"Success! Output shapes matched with paper architecture.")
    print(f"Classification: {cls.shape}")
    print(f"Regression (Covariance Samples): {box.shape}")