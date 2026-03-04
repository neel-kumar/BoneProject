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
# 3. DATA PREPARATION (Loading from Files)
# ==========================================

def load_inflation_training_data(sample_names, voxel_path='s01_voxel.npy', target_size=(100, 100, 100)):
    """
    Extracts triplets: (Skeleton, Parameters, Full Volume)
    Uses pre-generated skeletons/json and original voxel volume.
    """
    print("Loading datasets and extracting targets...")
    if not os.path.exists(voxel_path):
        print(f"Error: {voxel_path} not found.")
        return None, None, None, None, None

    voxeld = np.load(voxel_path)
    
    skeletons, params, targets = [], [], []
    param_keys = ["BV/TV", "porosity", "pore_size", "pBV/TV", "pTb.Th", "pTb.N", "rBV/TV", "rTb.Th"]
    
    for name in sample_names:
        ske_path = f'ske_{name}.npy'
        quant_path = f'quant_{name}.json'
        
        if os.path.exists(ske_path) and os.path.exists(quant_path):
            # Load skeleton
            ske = np.load(ske_path).astype(np.float32)
            
            # Load metadata and parameters
            with open(quant_path, 'r') as f:
                quant = json.load(f)
            
            # Extract target volume using the range from JSON
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
            print(f"  Processed {name}")

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
    sample_names = [f'rand{i}' for i in range(1, 34)]
    
    ske_data, param_data, vol_data, p_mean, p_std = load_inflation_training_data(sample_names)
    
    if ske_data is not None:
        print(f"\nTraining data loaded.")
        print(f"Skeletons shape: {ske_data.shape}")
        print(f"Params shape: {param_data.shape}")
        print(f"Targets shape: {vol_data.shape}")

        # Initialize Model
        model = build_inflation_network(volume_size=(100, 100, 100))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss='binary_crossentropy',
            metrics=['mae', 'accuracy']
        )

        print("\n--- Starting Inflation Model Training ---")
        start_time = time.time()
        
        # Train
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss', 
            patience=50, 
            min_delta=1e-8,
            restore_best_weights=True
        )

        history = model.fit(
            x={"skeleton": ske_data, "params": param_data},
            y=vol_data,
            epochs=1000,
            batch_size=2,
            verbose=1,
            callbacks=[early_stopping]
        )

        print(f"\nTraining complete in {time.time() - start_time:.2f} seconds.")

        # Save model and normalization stats
        model.save("bone_inflation_model_tf")
        np.savez("bone_inflation_stats.npz", mean=p_mean, std=p_std)
        print("Model and parameters saved.")
    else:
        print("Data extraction failed. Ensure rand*.npy/json files and s01_voxel.npy exist.")
