import numpy as np
import tensorflow as tf
from clr_bnn import CLR_BNN
import os
import gc

def inject_imagenet_prior_only(model):
    print("\n💉 STARTING PRIOR-ONLY INJECTION (IMAGE NET KNOWLEDGE ANCHOR)...")
    
    # 1. Load Donor (Standard ResNet50 from Keras)
    resnet_donor = tf.keras.applications.ResNet50(
        weights='imagenet', include_top=False, input_shape=(320, 320, 3)
    )

    donor_weights = {
        l.name: l.get_weights() 
        for l in resnet_donor.layers 
        if ('conv' in l.name or '_bn' in l.name) and len(l.get_weights()) > 0
    }
    
    target_layers = model.feature_extractor.layers
    count_conv = 0
    count_bn = 0
    
    for layer in target_layers:
        # --- A. BAYESIAN CONVOLUTIONS ONLY ---
        if 'conv' in layer.name and hasattr(layer, 'trainable_variables'):
            
            # Mapping name to donor (Standardizing)
            search_name = layer.name
            
            if 'lidar' in layer.name or 'radar' in layer.name:
                # print(f"   Skipping branch layer: {layer.name}")
                continue
            
            if search_name in donor_weights:
                source_w = donor_weights[search_name]
                
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    
                    if len(source_w) == 4:
                        old_w = layer.get_weights()
                        layer.set_weights(source_w)
                        # print(f"\n📊 BN Sync: {layer.name}")
                        # print(f"  - Gamma (Trainable)  Old: {np.mean(old_w[0]):.6f} -> New: {np.mean(source_w[0]):.6f}")
                        # print(f"  - Beta  (Trainable)  Old: {np.mean(old_w[1]):.6f} -> New: {np.mean(source_w[1]):.6f}")
                        # print(f"  - Moving Mean (Stat) Old: {np.mean(old_w[2]):.6f} -> New: {np.mean(source_w[2]):.6f}")
                        # print(f"  - Moving Var  (Stat) Old: {np.mean(old_w[3]):.6f} -> New: {np.mean(source_w[3]):.6f}")
                        count_bn += 1
                        
                elif 'conv' in layer.name and hasattr(layer, 'trainable_variables'):
                    source_kernel = source_w[0]
                    source_bias = source_w[1] if len(source_w) > 1 else None
                    
                    target_prior_loc = None
                    bias_prior_loc = None
                    for v in layer.trainable_variables:
                        if 'kernel_prior_loc' in v.name: target_prior_loc = v
                        if 'bias_prior_loc' in v.name: bias_prior_loc = v

                    if target_prior_loc is not None:
                        old_prior_mean = np.mean(target_prior_loc.numpy())
                        target_shape = target_prior_loc.shape
                        if source_kernel.shape != target_shape:
                            new_kernel = np.zeros(target_shape, dtype=np.float32)
                            c_copy = min(source_kernel.shape[2], target_shape[2])
                            new_kernel[:, :, :c_copy, :] = source_kernel[:, :, :c_copy, :]
                            target_prior_loc.assign(new_kernel)
                        else:
                            target_prior_loc.assign(source_kernel)
                        # print(f"\n🌀 Conv Prior Sync: {layer.name}")
                        # print(f"Kernel Prior Mean Old: {old_prior_mean:.8f} -> New: {np.mean(target_prior_loc.numpy()):.8f}")
                    if bias_prior_loc is not None and source_bias is not None:
                        old_bias_mean = np.mean(bias_prior_loc.numpy())
                        bias_prior_loc.assign(source_bias)
                        # print(f"Bias Prior Mean   Old: {old_bias_mean:.8f} -> New: {np.mean(source_bias):.8f}")
                    count_conv += 1

    print(f"\n✅ Injection completed: {count_conv} Conv (Prior) y {count_bn} BN layers.")
    del resnet_donor
    gc.collect()
    
if __name__ == "__main__":
    # Ejecution
    WEIGHTS_DIR = "./fl_weights_temp"
    model_global = CLR_BNN()
    dummy = [tf.zeros((1, 320, 320, 3)), tf.zeros((1, 320, 320, 2)), tf.zeros((1, 320, 320, 2))]
    model_global(dummy)
    model_global.load_weights(os.path.join(WEIGHTS_DIR, "global_weights.weights.h5"))
    inject_imagenet_prior_only(model_global)