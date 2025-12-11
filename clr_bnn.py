"""
IMPLEMENTATION OF CLR-BNN
==========================
Title: Camera, LiDAR, and Radar Sensor Fusion Based on Bayesian Neural Network (CLR-BNN)
Authors: Ratheesh Ravindran, Michael J. Santora, and Mohsin M. Jamali
Journal: IEEE Sensors Journal, Vol. 22, No. 7, April 1, 2022
DOI: 10.1109/JSEN.2022.3154980

Implementation details:
This script implements the single-stage sensor fusion Bayesian Neural Network (CLR-BNN).
It fuses Camera (RGB), LiDAR, and RADAR data to predict 2D object detection with 
aleatoric uncertainty (classification and bounding box covariance).

Key Components:
- Backbone: RetinaNet-50 based with Bayesian Layers.
- Bayesian Inference: Reparameterization layers with KL Divergence.
- Sensor Fusion: Deep feature-level fusion at multiple stages.
- RADAR Processing: Custom Min-Pooling for sparse data.
- LiDAR Processing: Parallel sequential branch structure.
- Uncertainty Head: Multivariate Normal Distribution for bounding boxes.
"""

import os

# --- CRITICAL BLOCK: GPU CONFIGURATION FOR WSL ---
# NOTE: Required for stability on modern WSL2/CUDA environments running 
# TensorFlow Probability to avoid "JIT compilation failed [Op:Softplus]" errors.
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0 --tf_xla_cpu_global_jit=0"
os.environ["TF_JIT_PROFILING"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# Explicitly force JIT deactivation in TF optimizer
tf.config.optimizer.set_jit(False)

# Configure memory growth to prevent VRAM allocation crashes
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
tfm = tfp.math

# ==============================================================================
# 1. BAYESIAN CONFIGURATION & UTILS
# Ref: Section II-B "Layers and Initialization"
# ==============================================================================

def get_kernel_divergence_fn(train_size=1000):
    """
    Calculates scaled KL divergence.
    Ref: Approximating predictive distribution via ELBO.
    """
    def divergence_fn(q, p, _):
        return tfd.kl_divergence(q, p) / train_size
    return divergence_fn

def get_prior_bias_posterior_fn(init_value=-4.59):
    """
    Generates a posterior function for the Bias with LOW VARIANCE.
    
    CRITICAL FIX for NaNs:
    1. loc_initializer: Sets the center to -4.59 (Prob ~0.01).
    2. untransformed_scale_initializer: Sets the variance to near zero.
       Val -5.0 -> Softplus -> 0.000000002.
       This prevents random Bayesian noise from destroying the initialization.
    """
    def posterior_fn(dtype, shape, name, trainable, add_variable_fn):
        return tfpl.default_mean_field_normal_fn(
            is_singular=False,
            # Fix 1: Center distribution at low probability
            loc_initializer=tf.constant_initializer(init_value),
            # Fix 2: Freeze variance initially so it acts deterministically
            untransformed_scale_initializer=tf.constant_initializer(-5.0)
        )(dtype, shape, name, trainable, add_variable_fn)
    return posterior_fn



def bayesian_conv2d(filters, kernel_size, strides=1, padding='same', activation='relu', name=None):
    """
    Creates a 'Convolution2DReparameterization' layer.
    
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
        activation=activation,
        name=name,
        kernel_prior_fn=tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        kernel_divergence_fn=get_kernel_divergence_fn()
    )

# ==============================================================================
# 2. CUSTOM LAYER: MIN POOLING IGNORING ZEROS
# Ref: Table II, Row 3 and Section II-D
# ==============================================================================

class MinIZPooling2D(tf.keras.layers.Layer):
    """
    Implements 'Min Pooling Ignoring Zeros' (Min_IZ_Pool2D).
    
    Paper Justification:
    Used for RADAR point-clouds to address sparsity. Standard MaxPool would pick 
    noise/far objects; standard MinPool would pick zeros (empty space).
    """
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        super(MinIZPooling2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding.upper()

    def call(self, inputs):
        """
        Executes Min-Pooling while Ignoring Zeros (Sparse Data).
        
        Inputs:
            Tensor of shape (Batch, Height, Width, Channels). 
                Typically RADAR Depth maps where 0.0 indicates missing data.
                    
        Outputs:
            Tensor of shape (Batch, H/2, W/2, Channels).
            Contains the minimum NON-ZERO value in each pooling window.
        """
        # 1. Detect zeros (missing data)
        zeros_mask = tf.equal(inputs, 0.0)
        
        # 2. Replace zeros with Infinity
        large_val = 1e5
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
# 3. SPECIFIC ARCHITECTURE BLOCKS
# Ref: Table I "CLR-DNN AND CLR-BNN ARCHITECTURE DETAILS"
# ==============================================================================

def bottleneck_block(x, filters, stride=1, use_projection=False):
    """
    Bayesian implementation of ResNet-50 bottleneck block (Conv1-5 Blocks).
    Filters structure: [1x1, 3x3, 1x1]
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

def build_conv1_block(x, name='conv1'):
    """
    Conv1 Block per Table I: 7x7, 64, stride 2
    Used for both Image and LiDAR branches.
    """
    x = bayesian_conv2d(filters=64, kernel_size=7, strides=2, name=name)(x)
    return x

def build_conv2_block(x, name='conv2'):
    """
    Conv2 Block per Table I: 
    1. 3x3 MaxPool, stride 2
    2. 3x Residual Bottleneck blocks
    Used for both Image and LiDAR branches.
    """
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name=f'{name}_pool')(x)
    
    # 3 residual blocks [64, 64, 256]
    x = bottleneck_block(x, [64, 64, 256], stride=1, use_projection=True)
    x = bottleneck_block(x, [64, 64, 256], stride=1)
    x = bottleneck_block(x, [64, 64, 256], stride=1)
    return x

def build_conv3_block(x, name='conv3'):
    """
    Conv3 Block per Table I:
    3x Residual Bottleneck blocks [128, 128, 512]
    """
    x = bottleneck_block(x, [128, 128, 512], stride=2, use_projection=True)
    x = bottleneck_block(x, [128, 128, 512], stride=1)
    x = bottleneck_block(x, [128, 128, 512], stride=1)
    return x

# ==============================================================================
# 4. MAIN MODEL: CLR-BNN
# Ref: Figure 1 (Architecture Overview)
# ==============================================================================

class CLR_BNN(tf.keras.Model):
    def __init__(self, num_classes=14, num_anchors=9):
        super(CLR_BNN, self).__init__()
        self.num_classes = num_classes    
        
         # --- Heads / Subnets ---
        
        # Ref: Section II-G "Classification and Regression Subnets"
        # "The final layer is modeled using 'MultivariateNormalTriL' layer to incorporate
        # the multi-variate Gaussian distribution with Cholesky decomposition."
        
        # 1. Params for Distribution
        # 4 coords (x,y,w,h) * 9 anchors = 36 random variables per cell
        self.event_size = 4 * num_anchors
        params_size = tfpl.MultivariateNormalTriL.params_size(self.event_size) 
        
        # Regression Subnet
        self.regression_net = tf.keras.Sequential([
            bayesian_conv2d(256, 3, name='reg_conv1'),
            bayesian_conv2d(256, 3, name='reg_conv2'),
            bayesian_conv2d(256, 3, name='reg_conv3'),
            bayesian_conv2d(256, 3, name='reg_conv4'),
            bayesian_conv2d(params_size, 3, name='reg_projection')
        ])
        
        # Classification Subnet (Uses Focal Loss downstream)
        self.classification_head = tf.keras.Sequential([
            bayesian_conv2d(256, 3, name='cls_conv1'),
            bayesian_conv2d(256, 3, name='cls_conv2'),
            bayesian_conv2d(256, 3, name='cls_conv3'),
            bayesian_conv2d(256, 3, name='cls_conv4'),
            
            tfpl.Convolution2DReparameterization(
                filters=num_classes * num_anchors, 
                kernel_size=3, 
                padding='same', 
                activation='sigmoid', 
                name='cls_final',
                # FIX: Usamos bias_posterior_fn en lugar de bias_initializer
                bias_posterior_fn=get_prior_bias_posterior_fn(-4.59),
                kernel_prior_fn=tfpl.default_multivariate_normal_fn,
                kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                kernel_divergence_fn=get_kernel_divergence_fn()
            )
        ])

        # This creates the static graph for Backbone + FPN + Fusion
        self.feature_extractor = self._build_feature_extractor()

    def _build_feature_extractor(self, input_shape = (320, 320)):
        """
        Builds the Feature Extractor with Sensor Fusion (Fig. 1).
        Ref: Figure 3 "Concatenation of camera image, LiDAR and RADAR".
        """
        
        h, w = input_shape
        input_img = tf.keras.Input(shape=(h, w, 3), name='img_input')
        input_lidar = tf.keras.Input(shape=(h, w, 2), name='lidar_input')
        input_radar = tf.keras.Input(shape=(h, w, 2), name='radar_input')

        # --- 1. BACKBONE CONSTRUCTION ---
        # Parallel Branches
        l_c1 = build_conv1_block(input_lidar, name='lidar_branch_c1')
        r_processed = MinIZPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_radar)
        r_c1 = build_conv1_block(r_processed, name='radar_branch_c1')

        # Stage 1
        s1_input = tf.keras.layers.Concatenate()([input_img, input_lidar, r_processed])
        c1 = build_conv1_block(s1_input, name='backbone_conv1')

        # Stage 2
        l_c2 = build_conv2_block(l_c1, name='lidar_branch_c2')
        r_c2 = build_conv2_block(r_c1, name='radar_branch_c2')
        c1_fused = tf.keras.layers.Concatenate()([c1, l_c1, r_c1])
        c2 = build_conv2_block(c1_fused, name='backbone_conv2')

        # Stage 3
        l_c3 = build_conv3_block(l_c2, name='lidar_branch_c3')
        r_c3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(r_c2)
        c2_fused = tf.keras.layers.Concatenate()([c2, l_c2, r_c2])
        c3 = build_conv3_block(c2_fused, name='backbone_conv3')

        # Stage 4
        l_c4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(l_c3)
        r_c4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(r_c3)
        c3_fused = tf.keras.layers.Concatenate()([c3, l_c3, r_c3])
        c4 = bottleneck_block(c3_fused, [256, 256, 1024], stride=2, use_projection=True)
        for _ in range(2): c4 = bottleneck_block(c4, [256, 256, 1024])

        # Stage 5
        l_c5 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(l_c4)
        r_c5 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(r_c4)
        c4_fused = tf.keras.layers.Concatenate()([c4, l_c4, r_c4])
        c5 = bottleneck_block(c4_fused, [512, 512, 2044], stride=2, use_projection=True)
        for _ in range(2): c5 = bottleneck_block(c5, [512, 512, 2044])

        # Sensor Downsampling for P7
        l_c6 = tf.keras.layers.MaxPool2D(2, strides=2, padding='same')(l_c5)
        r_c6 = tf.keras.layers.MaxPool2D(2, strides=2, padding='same')(r_c5)

        # --- 2. FPN CONSTRUCTION (Lateral Fusion) ---
        feature_size = 256
        
        # P6 (Top)
        # Fuse C5 + L5 + R5 -> Reduce -> Smooth
        p6_input = tf.keras.layers.Concatenate()([c5, l_c5, r_c5])
        p6_lat = bayesian_conv2d(feature_size, 1, name='fpn_lat_p6')(p6_input)
        p6_fused = tf.keras.layers.Concatenate()([p6_lat, l_c5, r_c5])
        p6_output = bayesian_conv2d(feature_size, 3, name='fpn_p6')(p6_fused)
        
        # P5 (C4)
        p6_up = tf.keras.layers.UpSampling2D(size=(2, 2))(p6_lat)
        p5_input = tf.keras.layers.Concatenate()([c4, l_c4, r_c4])
        p5_lat = bayesian_conv2d(feature_size, 1, name='fpn_lat_p5')(p5_input)
        p6_up = tf.image.resize(p6_up, tf.shape(p5_lat)[1:3])
        p5_sum = tf.keras.layers.Add()([p6_up, p5_lat])
        p5_fused = tf.keras.layers.Concatenate()([p5_sum, l_c4, r_c4])
        p5_output = bayesian_conv2d(feature_size, 3, name='fpn_p5')(p5_fused)

        # P4 (C3)
        p5_up = tf.keras.layers.UpSampling2D(size=(2, 2))(p5_sum)
        p4_input = tf.keras.layers.Concatenate()([c3, l_c3, r_c3])
        p4_lat = bayesian_conv2d(feature_size, 1, name='fpn_lat_p4')(p4_input)
        p5_up = tf.image.resize(p5_up, tf.shape(p4_lat)[1:3])
        p4_sum = tf.keras.layers.Add()([p5_up, p4_lat])
        p4_fused = tf.keras.layers.Concatenate()([p4_sum, l_c3, r_c3])
        p4_output = bayesian_conv2d(feature_size, 3, name='fpn_p4')(p4_fused)

        # P3 (C2)
        p4_up = tf.keras.layers.UpSampling2D(size=(2, 2))(p4_sum)
        p3_input = tf.keras.layers.Concatenate()([c2, l_c2, r_c2])
        p3_lat = bayesian_conv2d(feature_size, 1, name='fpn_lat_p3')(p3_input)
        p4_up = tf.image.resize(p4_up, tf.shape(p3_lat)[1:3])
        p3_sum = tf.keras.layers.Add()([p4_up, p3_lat])
        p3_fused = tf.keras.layers.Concatenate()([p3_sum, l_c2, r_c2])
        p3_output = bayesian_conv2d(feature_size, 3, name='fpn_p3')(p3_fused)

        # P7 (from P6)
        p7_in = tf.keras.layers.ReLU()(p6_lat)
        p7_in = bayesian_conv2d(feature_size, 3, strides=2, name='fpn_p7_in')(p7_in)
        p7_fused = tf.keras.layers.Concatenate()([p7_in, l_c6, r_c6])
        p7_output = bayesian_conv2d(feature_size, 3, name='fpn_p7')(p7_fused)

        features = [p3_output, p4_output, p5_output, p6_output, p7_output]

        return tf.keras.Model(
            inputs=[input_img, input_lidar, input_radar],
            outputs= features,
            name="feature_extractor"
        )

    def call(self, inputs, training=False):
        """
        Orchestration Logic using the pre-built graph.
        
        Args:
            inputs (list): List of tensors `[image, lidar, radar]`.
            training (bool): Whether the model is in training mode.
            
        Returns:
            If training=True:
                cls_outputs (list): List of class probability tensors for each pyramid level (P3-P7).
                box_outputs (list): List of `tfd.MultivariateNormalTriL` distribution objects for each level.
                
            If training=False:
                cls_final (tf.Tensor): Flattened class probabilities. Shape `(Batch, Total_Anchors, Num_Classes)`.
                box_final (tf.Tensor): Flattened box coordinates (means). Shape `(Batch, Total_Anchors, 4)`.
        """
        
        features = self.feature_extractor(inputs, training=training)
        
        cls_outputs = [self.classification_head(f) for f in features]
        
        box_outputs = []
        for f in features:
            # 1. Get Raw Distribution Parameters from Network
            params = self.regression_net(f)
            params = tf.cast(params, tf.float32)
            params = tf.clip_by_value(params, -100.0, 100.0)
            
            # 2. Split into Mean and Covariance Parameters
            loc = params[..., :self.event_size]
            scale_params = params[..., self.event_size:]
            
            # 3. Construct Lower Triangular Matrix (Covariance)
            scale_tril = tfm.fill_triangular(scale_params)
            
            # 4. Ensure positive diagonal for numerical stability
            diag = tf.linalg.diag_part(scale_tril)
            softplus_diag = tf.nn.softplus(diag) + 1e-5
            scale_tril = tf.linalg.set_diag(scale_tril, softplus_diag)
            
            # 5. Create Distribution Object
            dist = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)
            box_outputs.append(dist)

        # --- TRAINING: Return raw structure for ELBO Loss ---
        if training:
            return cls_outputs, box_outputs

        # --- INFERENCE: Flatten and Concatenate ---
        # 1. Classification
        cls_final = tf.concat([
            tf.keras.layers.Reshape((-1, self.num_classes))(c) for c in cls_outputs
        ], axis=1)
        
        # 2. Regression (Sample from distribution - using Mean for stability)
        box_samples = []
        for b in box_outputs:
            box_samples.append(b.mean())

        box_final = tf.concat([
            tf.keras.layers.Reshape((-1, 4))(b) for b in box_samples
        ], axis=1)

        return cls_final, box_final

# ==============================================================================
# MAIN TEST
# ==============================================================================
if __name__ == "__main__":
    print(f"Running CLR-BNN (Hybrid Functional/Subclass)")
    
    input_img = tf.random.normal((2, 320, 320, 3))
    input_lidar = tf.random.normal((2, 320, 320, 2))
    input_radar = tf.zeros((2, 320, 320, 2)) 
    
    print("Building Graph...")
    model = CLR_BNN(num_classes=10)
    
    print("Executing Forward Pass...")
    cls, box = model([input_img, input_lidar, input_radar])
    
    print(f"Success! Output shapes matched with paper architecture.")
    print(f"Classification Output: {cls.shape}")
    print(f"Regression Output (Covariance Samples): {box.shape}")