import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa 
from tqdm import tqdm
import numpy as np
import os
import json
from tensorflow.keras import mixed_precision

from src.models.architecture import CLR_BNN
from src.data.loader import DataLoaderGenerator

tfd = tfp.distributions

class CLR_BNN_Trainer:
    """
    Trainer class encapsulating the training logic for CLR-BNN.
    """
    def __init__(self, model, num_classes=14, num_anchors=9, initial_lr=0.002, steps_per_epoch=1000, epochs=1, current_round=0, train_dataset_size=None):
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
        self.num_anchors = num_anchors
        self.grid_sizes = [80, 40, 20, 10, 5]
        self.current_round = current_round
        self.train_dataset_size = train_dataset_size
        
        if train_dataset_size is not None and train_dataset_size > 0:
            target_kl_weight = 1.0 / float(train_dataset_size)
            if current_round < 5: 
                self.kl_weight = 0.0
            else:
                progress = min(1.0, (current_round - 5) / 15.0)
                self.kl_weight = target_kl_weight * progress
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
        self.tracker_val_loss = tf.keras.metrics.Mean(name="val_loss")
        
        # --- OPTIMIZER SETUP ---
        # Ref: Table I "Optimizer and learning rate"
        self.lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=max(1, (steps_per_epoch * epochs)), 
            end_learning_rate=1e-5,
            power=1.0
        )
        
        self.optimizer = tfa.optimizers.LAMB(
            learning_rate=self.lr_schedule,
            clipnorm=0.5
        )
        self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer)
        
        self.cls_loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
            gamma=2.0,
            alpha=0.25,
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE
        )

    def split_targets_by_level(self, y_true):
        """
        Splits ground truth tensors to match the pyramid levels.
        
        Args:
            y_true (tf.Tensor): Concatenated ground truth tensor (Batch, Total_Anchors, K).
            grid_sizes (list): List of grid dimensions.
            
        Returns:
            list[tf.Tensor]: List of ground truth tensors for each level.
        """
        split_sizes = [ (g**2) * 9 for g in self.grid_sizes]
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
        cls_outputs = [tf.cast(c, tf.float32) for c in cls_outputs]
        
        targets_cls_levels = self.split_targets_by_level(y_true_cls)
        targets_box_levels = self.split_targets_by_level(y_true_box)
        
        total_cls_loss = tf.constant(0.0, dtype=tf.float32)
        total_box_loss = tf.constant(0.0, dtype=tf.float32)
        
        for i in range(len(self.grid_sizes)):
            y_cls = targets_cls_levels[i]
            y_box = targets_box_levels[i]
            pred_cls = cls_outputs[i]
            pred_dist = box_outputs[i] 
            
            # --- 1. CLASSIFICATION ---
            pred_cls = tf.reshape(pred_cls, (tf.shape(y_cls)[0], -1, self.num_classes))
            positive_mask = tf.reduce_max(y_cls, axis=-1) > 0 
            positive_mask_float = tf.cast(positive_mask, dtype=tf.float32)
            num_positives = tf.reduce_sum(positive_mask_float)
            
            normalizer = tf.maximum(1.0, num_positives)
            if num_positives == 0:
                normalizer = tf.cast(tf.shape(y_cls)[1], tf.float32)
            
            curr_cls_loss = self.cls_loss_fn(y_cls, pred_cls)

            loss_sum = tf.reduce_sum(curr_cls_loss)
            total_cls_loss += tf.cast(loss_sum, tf.float32) / normalizer       
           
           # --- 2. REGRESSION (Robust NLL with Smooth L1) ---
            H, W = self.grid_sizes[i], self.grid_sizes[i]
            
            # Ground Truth
            # Flatten to (-1, 4) to operate coordinate-by-coordinate
            # Note: 36 channels = 9 anchors * 4 coordinates
            y_box_flat = tf.reshape(y_box, (-1, 4)) 

            # Predictions (Extract Mean and StdDev from the Distribution object)
            pred_mu = pred_dist.mean()      
            pred_std = pred_dist.stddev()   
            
            pred_mu_flat = tf.reshape(pred_mu, (-1, 4))
            pred_std_flat = tf.reshape(pred_std, (-1, 4))

            # --- STABILITY CLIP ---
            # Prevents sigma from approaching zero (division by zero) or exploding.
            # 1e-3 is a safe lower bound for float32 numerical stability.
            pred_std_flat = tf.clip_by_value(pred_std_flat, 1e-2, 10.0)

            # 1. Calculate Robust Localization Error (Smooth L1 / Huber)
            diff = tf.abs(y_box_flat - pred_mu_flat)
            # Smooth L1 logic: 0.5 * x^2 if x < 1, else |x| - 0.5
            smooth_l1 = tf.where(diff < 1.0, 0.5 * tf.square(diff), diff - 0.5)

            # 2. Calculate Variance (sigma^2) and Log Variance
            sigma2 = tf.square(pred_std_flat) + 1e-6
            log_sigma2 = tf.math.log(sigma2)
            inv_sigma2 = tf.clip_by_value(1.0 / sigma2, 0.0, 500.0)
            
            # 3. Attenuated Loss Calculation (Ref: Kendall & Gal)
            # Loss = (SmoothL1 / Variance) + 0.5 * Log(Variance)
            # Term 1: Attenuates error based on uncertainty
            # Term 2: Penalty for high uncertainty (Regularization)
            term_error = smooth_l1 * inv_sigma2
            term_reg = 0.5 * log_sigma2
            
            nll_item = term_error + term_reg
            
            # Sum the 4 values (tx, ty, tw, th) to get the total loss per anchor
            nll_flat = tf.reduce_sum(nll_item, axis=-1) # Shape: (Total_Anchors,)

            # --- MASKING ---
            # Flatten the positive mask to match the anchors
            pos_mask_flat = tf.reshape(positive_mask, (-1,))
            pos_mask_float = tf.cast(pos_mask_flat, dtype=tf.float32)

            # Safety Check: Filter out potential NaNs or Infs
            nll_safe = tf.where(tf.math.is_finite(nll_flat), nll_flat, tf.zeros_like(nll_flat))
            
            # Only sum loss for positive anchors (those containing an object)
            curr_box_loss = tf.reduce_sum(nll_safe * pos_mask_float)
            
            # Add to total box loss normalized by the number of positives
            total_box_loss += tf.cast(curr_box_loss, tf.float32) / normalizer
            
        data_loss = total_cls_loss + total_box_loss
        raw_losses = self.model.losses
        safe_losses = []
        for l in raw_losses:
            l_32 = tf.cast(l, tf.float32)
            safe_l = tf.where(tf.math.is_finite(l_32), l_32, tf.zeros_like(l_32))
            safe_losses.append(safe_l)
        raw_kl_sum = tf.reduce_sum(safe_losses)
        kl_loss = raw_kl_sum * self.kl_weight

        # tf.print(raw_kl, kl_loss)
        
        # # 1. Extraer todas las capas que tienen KL (incluyendo Sequential)
        # all_bayesian_layers = []
        # for layer in self.model.layers:
        #     if isinstance(layer, tf.keras.Sequential):
        #         # Extraemos las sub-capas de las heads de regresión y clasificación
        #         all_bayesian_layers.extend([sub for sub in layer.layers if hasattr(sub, 'losses') and sub.losses])
        #     elif hasattr(layer, 'losses') and layer.losses:
        #         all_bayesian_layers.append(layer)

        # # --- FORMA CORRECTA PARA TF.FUNCTION ---
        # tf.print("📊 DESGLOSE DE KL - BATCH TOTAL:", kl_loss)
        # for layer in self.model._flatten_layers():
        #     if hasattr(layer, 'losses') and layer.losses:
        #         layer_kl = tf.reduce_sum(layer.losses) * self.kl_weight
        #         if layer_kl > 0:
        #             tf.print("Layer:", f"{layer.name:<30}", "| KL:", layer_kl)

        # tf.print("=" * 50 + "\n")
        
        total_loss = data_loss + kl_loss
        total_loss = tf.where(tf.math.is_finite(total_loss), total_loss, tf.constant(0.0, dtype=tf.float32))
        
        return (tf.cast(total_loss, tf.float32), 
                tf.cast(total_cls_loss, tf.float32), 
                tf.cast(total_box_loss, tf.float32), 
                tf.cast(kl_loss, tf.float32))

    @tf.function
    def train_step(self, inputs, targets):
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
        images, lidar, radar = inputs
        target_cls, target_box = targets
        
        with tf.GradientTape() as tape:
            cls_outs, box_outs = self.model([images, lidar, radar], training=True)
            loss, l_c, l_b, l_kl = self.compute_loss(target_cls, target_box, cls_outs, box_outs)
            scaled_loss = self.optimizer.get_scaled_loss(loss)
        
        scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables) 
        grads = self.optimizer.get_unscaled_gradients(scaled_gradients)
        grads_ok = [tf.where(tf.math.is_nan(g), tf.zeros_like(g), g) if g is not None else g for g in grads]
        self.optimizer.apply_gradients(zip(grads_ok, self.model.trainable_variables))
        
        # Update metrics tracking
        self.tracker_loss.update_state(loss)
        self.tracker_cls.update_state(l_c)
        self.tracker_box.update_state(l_b)
        self.tracker_kl.update_state(l_kl)
        
        return {"loss": loss, "cls": l_c, "box": l_b, "kl": l_kl}

    @tf.function
    def val_step(self, inputs, targets):
        img, lid, rad = inputs
        y_c, y_b = targets
        
        # Training=False uses Mean weights (Deterministic)
        cls_outs, box_outs = self.model([img, lid, rad], training=False)
        loss, _, _, _ = self.compute_loss(y_c, y_b, cls_outs, box_outs)
        self.tracker_val_loss.update_state(loss)
        return loss

    def fit(self, train_ds, val_ds, epochs=1, callbacks=None):
        """
        Main Training Loop with Progress Bar.
        Arguments:
            train_ds: Training dataset/generator
            val_ds: Validation dataset/generator (Can be None for FL clients)
            epochs: Number of epochs
        """
        callbacks = callbacks or []
        
        for epoch in range(epochs):
  
            # Reset metrics at start of epoch
            self.tracker_loss.reset_states()
            self.tracker_cls.reset_states()
            self.tracker_box.reset_states()
            self.tracker_kl.reset_states()
            self.tracker_val_loss.reset_states()
            
            # --- TRAINING ---
            pbar = tqdm(train_ds, desc=f"   Epoch {epoch+1}", unit="batch", leave=False)
            
            for batch_idx, (inputs, targets) in enumerate(pbar):
                current_batch_size = inputs[0].shape[0]
                if current_batch_size == 0:
                    print(f"⚠️ Skipping empty batch {batch_idx}, Image shape: {inputs[0].shape[0]}, Lidar shape: {inputs[1].shape[0]}, Radar shape: {inputs[2].shape[0]}")
                    continue
                self.train_step(inputs, targets)
                pbar.set_postfix({
                    "L": f"{self.tracker_loss.result():.2f}",
                    "C": f"{self.tracker_cls.result():.2f}",
                    "B": f"{self.tracker_box.result():.2f}",
                    "KL": f"{self.tracker_kl.result():.2f}"
                })
                            
            # --- VALIDATION ---
            if val_ds is not None:
                # Run validation only if dataset provided
                for i, (inputs, targets) in enumerate(val_ds):
                    self.val_step(inputs, targets)
                    if i >= 20: break # Speed limit for validation
            
                
                print(f"      End Epoch {epoch+1} -> Train Loss: {self.tracker_loss.result():.4f} | Val Loss: {self.tracker_val_loss.result():.4f}")
                
            else:
                # If no validation, just print Train Loss
                print(f"      End Epoch {epoch+1} -> Train Loss: {self.tracker_loss.result():.4f} | L: {self.tracker_loss.result():.2f}, C: {self.tracker_cls.result():.2f}, B: {self.tracker_box.result():.2f}, KL: {self.tracker_kl.result():.2f}") 

                logs = {
                    'loss': float(self.tracker_loss.result()),
                    'cls_loss': float(self.tracker_cls.result()),
                    'box_loss': float(self.tracker_box.result()),
                    'kl_loss': float(self.tracker_kl.result()),
                    'val_loss': float(self.tracker_val_loss.result()) if val_ds else 0.0
                }
                for callback in callbacks:
                    callback.on_epoch_end(epoch, logs)