import tensorflow as tf
from tensorflow.keras import layers, models, losses
import numpy as np
import json
import os
import time
from scipy.ndimage import zoom

# ==========================================
# 1. CUSTOM LAYERS: FiLM (Parameter Injection)
# ==========================================

class FiLMLayer(layers.Layer):
    """
    Imparts parameters by scaling and shifting feature maps.
    As described in Baishnab et al. (2025).
    """
    def __init__(self, **kwargs):
        super(FiLMLayer, self).__init__(**kwargs)

    def call(self, inputs):
        x, conditioning = inputs # x: feature map, conditioning: [gamma, beta]
        gamma = conditioning[:, :x.shape[-1]]
        beta = conditioning[:, x.shape[-1]:]
        
        # Reshape for 3D broadcasting
        gamma = tf.reshape(gamma, (-1, 1, 1, 1, x.shape[-1]))
        beta = tf.reshape(beta, (-1, 1, 1, 1, x.shape[-1]))
        
        return (1 + gamma) * x + beta

# ==========================================
# 2. MODEL: CONDITIONAL 3D U-NET (Inflation)
# ==========================================

def build_inflation_network(volume_size=(100, 100, 100), param_dim=8):
    # Inputs
    skeleton_input = layers.Input(shape=(*volume_size, 1), name="skeleton")
    param_input = layers.Input(shape=(param_dim,), name="params")
    
    # Parameter Embedding (MLP)
    p_embed = layers.Dense(128, activation='swish')(param_input)
    p_embed = layers.Dense(256, activation='swish')(p_embed)

    # --- Encoder ---
    # Block 1: 100x100x100 -> 50x50x50
    c1 = layers.Conv3D(32, 3, padding='same', activation='relu')(skeleton_input)
    cond1 = layers.Dense(64)(p_embed) # gamma + beta
    c1 = FiLMLayer()([c1, cond1])
    p1 = layers.MaxPooling3D((2, 2, 2))(c1)

    # Block 2: 50x50x50 -> 25x25x25
    c2 = layers.Conv3D(64, 3, padding='same', activation='relu')(p1)
    cond2 = layers.Dense(128)(p_embed)
    c2 = FiLMLayer()([c2, cond2])
    p2 = layers.MaxPooling3D((2, 2, 2))(c2)

    # --- Bottleneck ---
    b = layers.Conv3D(128, 3, padding='same', activation='relu')(p2)

    # --- Decoder ---
    # Block 3: 25x25x25 -> 50x50x50
    u3 = layers.UpSampling3D((2, 2, 2))(b)
    # Ensure concatenation works on 25x25x25 upsampled to 50x50x50
    u3 = layers.concatenate([u3, c2])
    c3 = layers.Conv3D(64, 3, padding='same', activation='relu')(u3)
    
    # Block 4: 50x50x50 -> 100x100x100
    u4 = layers.UpSampling3D((2, 2, 2))(c3)
    u4 = layers.concatenate([u4, c1])
    c4 = layers.Conv3D(32, 3, padding='same', activation='relu')(u4)

    # Output Layer
    output = layers.Conv3D(1, 1, activation='sigmoid', name="volume")(c4)

    model = models.Model(inputs=[skeleton_input, param_input], outputs=output)
    return model

# ==========================================
# 3. LOSS FUNCTIONS: Strict Structure & Connectivity
# ==========================================

def dice_loss(y_true, y_pred, smooth=1e-6):
    """Strict volumetric overlap loss."""
    y_true_f = tf.cast(layers.Flatten()(y_true), tf.float32)
    y_pred_f = layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=-1)
    denominator = tf.reduce_sum(y_true_f + y_pred_f, axis=-1)
    return 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)

def skeleton_strict_coverage_loss(y_true_ske, y_pred, weight=100.0):
    """
    Extremely strict penalty. If even one voxel of the skeleton is missing 
    (pred < 1.0 at ske == 1), the loss increases significantly.
    """
    # 1. Individual Voxel Penalty (Log-space ensures they all move to 1.0)
    # Using negative log likelihood specifically for the skeleton points
    epsilon = 1e-7
    individual_loss = -tf.reduce_mean(y_true_ske * tf.math.log(y_pred + epsilon))
    
    # 2. Global Recall Penalty (Percentage of skeleton missed)
    ske_sum = tf.reduce_sum(y_true_ske) + epsilon
    recall = tf.reduce_sum(y_true_ske * y_pred) / ske_sum
    global_recall_penalty = (1.0 - recall) 
    
    # Combine with a high weight to make it "major"
    return (individual_loss + global_recall_penalty) * weight

class ConnectivityAwareLoss(losses.Loss):
    def __init__(self, name="connectivity_aware_loss"):
        super().__init__(name=name)
        self.bce = losses.BinaryCrossentropy()

    def call(self, y_true_combined, y_pred):
        # We pack [target_volume, input_skeleton] into y_true_combined
        y_true = y_true_combined[..., 0:1]
        y_ske = y_true_combined[..., 1:2]

        # 1. Standard BCE
        loss_bce = self.bce(y_true, y_pred)
        
        # 2. Dice Loss (Strict volumetric overlap)
        loss_dice = tf.reduce_mean(dice_loss(y_true, y_pred))
        
        # 3. Skeleton Coverage Penalty (Major penalty if any voxel is missing)
        loss_ske = skeleton_strict_coverage_loss(y_ske, y_pred)

        return loss_bce + loss_dice + loss_ske

# ==========================================
# 4. DATA PREPARATION (Loading from Files)
# ==========================================

def load_inflation_training_data(target_size=(100, 100, 100)):
    """
    Extracts triplets: (Skeleton, Parameters, Full Volume)
    Uses pre-generated skeletons/json and corresponding voxel volumes.
    """
    import glob
    import re
    # Directories to search
    base_dirs = glob.glob('../../microCT/S.*_fins*') + ['./ctbins']
    
    skeletons, params, targets = [], [], []
    param_keys = ["BV/TV", "porosity", "pore_size", "pBV/TV", "pTb.Th", "pTb.N", "rBV/TV", "rTb.Th"]
    
    # Cache for voxel data (to avoid reloading large files)
    voxel_cache = {}
    
    print("Finding datasets and extracting targets...")
    
    for d in base_dirs:
        # Determine voxel file for this directory
        if 'ctbins' in d:
            voxel_path = 's01_voxel.npy'
        else:
            s_match = re.search(r'(S\.\d+)', d)
            if s_match:
                s_name = s_match.group(1)
                voxel_path = f'../../microCT/{s_name}_voxel.npy'
            else:
                voxel_path = 's01_voxel.npy'
        
        if not os.path.exists(voxel_path):
            print(f"Warning: Voxel file {voxel_path} not found. Skipping directory {d}")
            continue

        # Find all ske_rand*.npy in this directory
        ske_files = glob.glob(os.path.join(d, 'ske_rand*.npy'))
        if not ske_files:
            continue
            
        # Load voxel data if not in cache
        if voxel_path not in voxel_cache:
            # Clear cache if we have too many voxels to save RAM
            if len(voxel_cache) >= 1:
                voxel_cache.clear()
            print(f"Loading voxel data: {voxel_path}")
            voxel_cache[voxel_path] = np.load(voxel_path)
        
        voxeld = voxel_cache[voxel_path]
        print(f"  Found {len(ske_files)} samples in {d}")

        for ske_path in ske_files:
            match = re.search(r'ske_(rand\d+)\.npy', os.path.basename(ske_path))
            if not match: continue
            name = match.group(1)
            quant_path = os.path.join(d, f'quant_{name}.json')
            
            if os.path.exists(quant_path):
                # Load skeleton
                ske = np.load(ske_path).astype(np.float32)
                
                # Load metadata and parameters
                with open(quant_path, 'r') as f:
                    quant = json.load(f)
                
                # Extract target volume using the range from JSON
                try:
                    r = quant['range']
                    vol_target = voxeld[r['z'][0]:r['z'][1], r['y'][0]:r['y'][1], r['x'][0]:r['x'][1]].astype(np.float32)
                    
                    # Resize both to target_size (100x100x100)
                    s_factors = [t / s for t, s in zip(target_size, ske.shape)]
                    t_factors = [t / s for t, s in zip(target_size, vol_target.shape)]
                    
                    ske_resized = zoom(ske, s_factors, order=1)
                    vol_resized = zoom(vol_target, t_factors, order=1)
                    
                    p_vec = [quant[k] for k in param_keys]
                    
                    skeletons.append(np.expand_dims(ske_resized, -1))
                    params.append(p_vec)
                    targets.append(np.expand_dims(vol_resized, -1))
                except Exception as e:
                    print(f"    Error processing {name}: {e}")
                    continue
                
                if len(skeletons) % 50 == 0:
                    print(f"  Processed {len(skeletons)} samples...")

    # Clear cache to free memory
    voxel_cache.clear()

    if not skeletons:
        return None, None, None, None, None

    skeletons = np.array(skeletons)
    params = np.array(params).astype(np.float32)
    targets = np.array(targets)
    
    # Standardize parameters
    p_mean = params.mean(axis=0)
    p_std = params.std(axis=0)
    p_std[p_std == 0] = 1.0
    params = (params - p_mean) / p_std
    
    return skeletons, params, targets, p_mean, p_std

# ==========================================
# 4. TRAINING EXECUTION
# ==========================================

if __name__ == "__main__":
    ske_data, param_data, vol_data, p_mean, p_std = load_inflation_training_data()
    
    if ske_data is not None:
        print(f"\nTraining data loaded.")
        print(f"Skeletons shape: {ske_data.shape}")
        print(f"Params shape: {param_data.shape}")
        print(f"Targets shape: {vol_data.shape}")

        # Initialize Model
        model = build_inflation_network(volume_size=(100, 100, 100))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss=ConnectivityAwareLoss(),
            metrics=['mae', 'accuracy']
        )

        # Pack Target Volume and Skeleton for loss (ConnectivityAwareLoss splits them)
        y_combined = np.concatenate([vol_data, ske_data], axis=-1)

        print("\n--- Starting Inflation Model Training ---")
        start_time = time.time()
        
        # Train
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='accuracy', 
            patience=20, 
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath="cp_inflation_ep{epoch:02d}_acc{accuracy:.2f}.keras",
            save_weights_only=False,
            monitor='accuracy',
            mode='max',
            save_freq='epoch'
        )

        history = model.fit(
            x={"skeleton": ske_data, "params": param_data},
            y=y_combined,
            # epochs=1000,
            epochs=100,
            batch_size=2,
            verbose=1,
            callbacks=[early_stopping, checkpoint]
        )

        print(f"\nTraining complete in {time.time() - start_time:.2f} seconds.")

        # Save model and normalization stats
        model.save("bone_inflation_model_tf.keras")
        np.savez("bone_inflation_stats.npz", mean=p_mean, std=p_std)
        print("Model and parameters saved.")
    else:
        print("Data extraction failed. Ensure rand*.npy/json files and s01_voxel.npy exist.")
