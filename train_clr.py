import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa 
from tensorflow.keras import mixed_precision
from tqdm import tqdm
import numpy as np
from clr_bnn import CLR_BNN
from data_loader import DataLoaderGenerator
from metrics import ValidationCallback
import os

DATA_ROOT = "/mnt/c/Users/USUARIO/Desktop/enviroments/Paper_NOUS/datasets/nuscenes_preprocessed"
BATCH_SIZE = 8 
EPOCHS = 30
LR = 1e-4
NUM_CLASSES = 14

tfd = tfp.distributions

class CLR_BNN_Trainer:
    """
    Trainer class encapsulating the training logic for CLR-BNN.
    """
    def __init__(self, model, num_classes=10, initial_lr=1e-5, steps_per_epoch=1000, train_dataset_size=None):
        """
        Initializes the trainer.
        
        Args:
            model (CLR_BNN): The model instance to train.
            num_classes (int): Number of object categories.
            initial_lr (float): Initial learning rate.
            steps_per_epoch (int): Steps per epoch for LR scheduling.
        """
        self.model = model
        self.num_classes = num_classes
        
        if train_dataset_size is not None and train_dataset_size > 0:
            self.kl_weight = 1.0 / float(train_dataset_size)
            print(f"✅ KL Weight ajustado automáticamente: {self.kl_weight:.6f} (1/{train_dataset_size})")
        else:
            # Fallback seguro si no sabemos el tamaño
            self.kl_weight = 1.0 / 1000.0 
            print(f"⚠️ Dataset size desconocido. Usando KL Weight por defecto: {self.kl_weight}")
        
        # --- METRICS TRACKERS ---
        # These allow us to average the loss over the whole epoch
        self.tracker_loss = tf.keras.metrics.Mean(name="loss")
        self.tracker_cls = tf.keras.metrics.Mean(name="cls_loss")
        self.tracker_box = tf.keras.metrics.Mean(name="box_loss")
        self.tracker_kl = tf.keras.metrics.Mean(name="kl_loss")
        
        # --- OPTIMIZER SETUP ---
        # Ref: Table I "Optimizer and learning rate"
        self.lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=steps_per_epoch * 80, 
            end_learning_rate=1e-6,
            power=0.5
        )
        
        base_optimizer = tfa.optimizers.LAMB(
            learning_rate=self.lr_schedule,
            clipnorm=0.5
        )
        
        self.optimizer = mixed_precision.LossScaleOptimizer(base_optimizer)
        
        self.cls_loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
            gamma=2.0, alpha=0.25, from_logits=False, reduction=tf.keras.losses.Reduction.NONE
        )

    def split_targets_by_level(self, y_true, grid_sizes):
        """
        Splits ground truth tensors to match the pyramid levels.
        
        Args:
            y_true (tf.Tensor): Concatenated ground truth tensor (Batch, Total_Anchors, K).
            grid_sizes (list): List of grid dimensions [152, 76, ...].
            
        Returns:
            list[tf.Tensor]: List of ground truth tensors for each level.
        """
        split_sizes = [ (g**2) * 9 for g in grid_sizes ]
        return tf.split(y_true, split_sizes, axis=1)

    def compute_loss(self, y_true_cls, y_true_box, cls_outputs, box_outputs):
        """
        Computes the total loss (ELBO).
        
        Args:
            y_true_cls (tf.Tensor): Ground truth classes.
            y_true_box (tf.Tensor): Ground truth boxes.
            cls_outputs (list): Model class predictions.
            box_outputs (list): Model box distributions.
            
        Returns:
            tuple: (total_loss, cls_loss, box_loss, kl_loss).
        """
        grid_sizes = [80, 40, 20, 10, 5] 
        targets_cls_levels = self.split_targets_by_level(y_true_cls, grid_sizes)
        targets_box_levels = self.split_targets_by_level(y_true_box, grid_sizes)
        
        total_cls_loss = 0.0
        total_box_loss = 0.0
        
        for i in range(len(grid_sizes)):
            y_cls = targets_cls_levels[i]
            y_box = targets_box_levels[i]
            pred_cls = cls_outputs[i]
            pred_dist = box_outputs[i] 
            
            # --- 1. CLASSIFICATION ---
            pred_cls = tf.reshape(pred_cls, tf.shape(y_cls))
            positive_mask = tf.reduce_max(y_cls, axis=-1) > 0 
            positive_mask_float = tf.cast(positive_mask, dtype=tf.float32)
            num_positives = tf.reduce_sum(positive_mask_float)
            
            normalizer = tf.maximum(1.0, num_positives)
            if num_positives == 0:
                normalizer = tf.cast(tf.shape(y_cls)[1], tf.float32)
            
            curr_cls_loss = self.cls_loss_fn(y_cls, pred_cls)
            total_cls_loss += tf.reduce_sum(curr_cls_loss) / normalizer

            # --- 2. REGRESSION (NLL) ---
            H, W = grid_sizes[i], grid_sizes[i]
            y_box_reshaped = tf.reshape(y_box, (-1, H, W, 36))
            y_box_reshaped = tf.cast(y_box_reshaped, tf.float32)
            
            if num_positives > 0:
                nll_spatial = -pred_dist.log_prob(y_box_reshaped)
                nll_flat = tf.reshape(nll_spatial, (-1, H * W))
                
                pos_mask_reshaped = tf.reshape(positive_mask, (-1, H * W, 9))
                pos_mask_cell = tf.reduce_any(pos_mask_reshaped, axis=-1)
                pos_mask_cell_float = tf.cast(pos_mask_cell, dtype=tf.float32)
                
                nll_safe = tf.where(tf.math.is_finite(nll_flat), nll_flat, tf.zeros_like(nll_flat))
                
                curr_box_loss = tf.reduce_sum(nll_safe * pos_mask_cell_float)
                total_box_loss += curr_box_loss / normalizer

        raw_kl = tf.cast(tf.reduce_sum(self.model.losses), tf.float32)
        kl_loss = raw_kl * self.kl_weight
        total_loss = total_cls_loss + total_box_loss + kl_loss
        return total_loss, total_cls_loss, total_box_loss, kl_loss

    @tf.function
    def train_step(self, images, lidar, radar, target_cls, target_box):
        """
        Runs a single training step.
        
        Args:
            images (tf.Tensor): Batch of images.
            lidar (tf.Tensor): Batch of LiDAR data.
            radar (tf.Tensor): Batch of RADAR data.
            target_cls (tf.Tensor): Classification labels.
            target_box (tf.Tensor): Box targets.
            
        Returns:
            dict: Loss metrics for the step.
        """
        with tf.GradientTape() as tape:
            cls_outs, box_outs = self.model([images, lidar, radar], training=True)
            loss, l_c, l_b, l_kl = self.compute_loss(target_cls, target_box, cls_outs, box_outs)
            scaled_loss = self.optimizer.get_scaled_loss(loss)
        
        scaled_grads = tape.gradient(scaled_loss, self.model.trainable_variables)
        grads = self.optimizer.get_unscaled_gradients(scaled_grads)
        
        grads_ok = [tf.where(tf.math.is_nan(g), tf.zeros_like(g), g) if g is not None else g for g in grads]
        self.optimizer.apply_gradients(zip(grads_ok, self.model.trainable_variables))
        
        # Update metrics tracking
        self.tracker_loss.update_state(loss)
        self.tracker_cls.update_state(l_c)
        self.tracker_box.update_state(l_b)
        self.tracker_kl.update_state(l_kl)
        
        return {"loss": loss, "cls": l_c, "box": l_b, "kl": l_kl}

    def fit(self, train_ds, val_ds, epochs=1):
        """
        Main Training Loop with Progress Bar.
        """
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Reset metrics at start of epoch
            self.tracker_loss.reset_states()
            self.tracker_cls.reset_states()
            self.tracker_box.reset_states()
            self.tracker_kl.reset_states()
            self.tracker_val_loss.reset_states()
            
            # --- TRAINING ---
            pbar = tqdm(train_ds, unit="batch")
            
            for inputs, targets in pbar:
                self.train_step(inputs, targets)
                pbar.set_postfix({
                    "L": f"{self.tracker_loss.result():.2f}",
                    "C": f"{self.tracker_cls.result():.2f}",
                    "B": f"{self.tracker_box.result():.2f}",
                    "KL": f"{self.tracker_kl.result():.2f}"
                })
                            
                # --- VALIDATION ---
                print("Running Validation...")
                for i, (inputs, targets) in enumerate(val_ds):
                    self.val_step(inputs, targets)
                    if i >= 50: break
                    
                print(f"End of Epoch {epoch+1} -> Train Loss: {self.tracker_loss.result():.4f} | Val Loss: {self.tracker_val_loss.result():.4f}")
                
                # Guardar pesos
                os.makedirs("checkpoints", exist_ok=True)
                self.model.save_weights(f"checkpoints/clr_bnn_epoch_{epoch+1}.h5")
            # End of Epoch Summary
            print(f"End of Epoch {epoch+1}: Total Loss: {self.tracker_loss.result():.4f}")    


if __name__ == "__main__":
    print("Loading Datasets...")
    train_gen = DataLoaderGenerator(DATA_ROOT, batch_size=BATCH_SIZE, split='train')
    val_gen = DataLoaderGenerator(DATA_ROOT, batch_size=BATCH_SIZE, split='val')
    print(f"   Train Batches: {len(train_gen)} | Val Batches: {len(val_gen)}")
    
    print("Initializing Bayesian Model and Trainer...")
    model = CLR_BNN(num_classes=NUM_CLASSES)
    dummy_in = [tf.zeros((1, 320, 320, 3)), tf.zeros((1, 320, 320, 2)), tf.zeros((1, 320, 320, 2))]
    model(dummy_in)
    
    trainer = CLR_BNN_Trainer(
        model, 
        num_classes=NUM_CLASSES, 
        initial_lr=LR, 
        steps_per_epoch=len(train_gen),
        train_dataset_size=len(train_gen) * BATCH_SIZE
    )
    
    print("Starting Training Loop...")
    try:
        trainer.fit(train_gen, val_gen, epochs=EPOCHS)
        print("\nTraining Finished Successfully!")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()