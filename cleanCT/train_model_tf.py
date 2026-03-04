import tensorflow as tf
from tensorflow.keras import layers, models, losses
import numpy as np
import json
import os
import time
from scipy.ndimage import zoom

# =================================================================
# 1. MODULE 1: THE 3D VAE (Latent Compression)
# =================================================================

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1:]
        epsilon = tf.random.normal(shape=tf.concat([[batch], dim], axis=0))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(input_shape=(100, 100, 100, 1), latent_channels=4):
    # Encoder
    encoder_inputs = layers.Input(shape=input_shape)
    x = layers.Conv3D(32, 3, strides=2, padding="same", activation="relu")(encoder_inputs) # 50x50x50
    x = layers.Conv3D(64, 3, strides=2, padding="same", activation="relu")(x) # 25x25x25
    
    z_mean = layers.Conv3D(latent_channels, 3, padding="same", name="z_mean")(x)
    z_log_var = layers.Conv3D(latent_channels, 3, padding="same", name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    
    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    
    # Decoder
    latent_inputs = layers.Input(shape=(25, 25, 25, latent_channels))
    x = layers.Conv3DTranspose(64, 3, strides=2, padding="same", activation="relu")(latent_inputs) # 50x50x50
    x = layers.Conv3DTranspose(32, 3, strides=2, padding="same", activation="relu")(x) # 100x100x100
    decoder_outputs = layers.Conv3D(1, 3, padding="same", activation="sigmoid")(x)
    
    decoder = models.Model(latent_inputs, decoder_outputs, name="decoder")
    
    # Full VAE
    outputs = decoder(encoder(encoder_inputs)[2])
    vae = models.Model(encoder_inputs, outputs, name="vae")
    
    return vae, encoder, decoder

# =================================================================
# 2. MODULE 2: THE FEATURE PREDICTOR (FP)
# =================================================================

def build_feature_predictor(latent_shape=(25, 25, 25, 4), param_dim=8):
    inputs = layers.Input(shape=latent_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(param_dim)(x)
    return models.Model(inputs, outputs, name="feature_predictor")

# =================================================================
# 3. MODULE 3: CONDITIONAL LATENT DIFFUSION (LDM)
# =================================================================

class ConditionalLDM(models.Model):
    def __init__(self, latent_ch=4, param_dim=8):
        super().__init__()
        self.time_mlp = models.Sequential([
            layers.Dense(128, activation="swish"),
            layers.Dense(128)
        ])
        self.param_mlp = models.Sequential([
            layers.Dense(128, activation="swish"),
            layers.Dense(128)
        ])
        
        self.down = layers.Conv3D(128, 3, padding="same")
        self.up = layers.Conv3D(latent_ch, 3, padding="same")
        
        self.cond_layer = layers.Dense(256)

    def call(self, x, t, p):
        # Embeddings
        t_emb = self.time_mlp(tf.expand_dims(tf.cast(t, tf.float32), -1))
        p_emb = self.param_mlp(p)
        context = tf.concat([t_emb, p_emb], axis=-1)
        
        # FiLM conditioning
        cond = self.cond_layer(context)
        cond = tf.reshape(cond, (-1, 1, 1, 1, 256))
        gamma, beta = tf.split(cond, 2, axis=-1)
        
        h = self.down(x)
        h = gamma * h + beta # Apply bone parameters + time context
        return self.up(tf.nn.relu(h))

# =================================================================
# 4. DATASET UTILITIES
# =================================================================

def load_bone_data(sample_names, target_size=(100, 100, 100)):
    all_skeletons = []
    all_params = []
    param_keys = ["BV/TV", "porosity", "pore_size", "pBV/TV", "pTb.Th", "pTb.N", "rBV/TV", "rTb.Th"]
    
    print("Loading and preprocessing data...")
    for name in sample_names:
        ske_path = f'ske_{name}.npy'
        quant_path = f'quant_{name}.json'
        
        if os.path.exists(ske_path) and os.path.exists(quant_path):
            ske = np.load(ske_path).astype(np.float32)
            with open(quant_path, 'r') as f:
                quant = json.load(f)
            
            # Resize 3D volume
            factors = [t / s for t, s in zip(target_size, ske.shape)]
            ske_resized = zoom(ske, factors, order=1)
            ske_resized = np.expand_dims(ske_resized, axis=-1) # Add channel dim
            
            params = [quant[k] for k in param_keys]
            
            all_skeletons.append(ske_resized)
            all_params.append(params)
    
    if not all_skeletons:
        return None, None, None, None

    all_skeletons = np.array(all_skeletons)
    all_params = np.array(all_params).astype(np.float32)
    
    # Standardize parameters
    p_mean = all_params.mean(axis=0)
    p_std = all_params.std(axis=0)
    p_std[p_std == 0] = 1.0
    all_params = (all_params - p_mean) / p_std
    
    return all_skeletons, all_params, p_mean, p_std

# =================================================================
# 5. TRAINING LOOP
# =================================================================

def train_pipeline(skeletons, params, epochs=100, batch_size=2):
    dataset = tf.data.Dataset.from_tensor_slices((skeletons, params)).shuffle(100).batch(batch_size)
    
    vae, encoder, decoder = build_vae()
    fp = build_feature_predictor()
    ldm = ConditionalLDM()
    
    # Phase 1: Train VAE
    print("\n--- Phase 1: Training VAE (Reconstruction) ---")
    vae_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    @tf.function
    def train_step_vae(x):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(x)
            reconstruction = decoder(z)
            recon_loss = tf.reduce_mean(losses.mse(x, reconstruction))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = recon_loss + 0.01 * kl_loss
        grads = tape.gradient(total_loss, vae.trainable_variables)
        vae_optimizer.apply_gradients(zip(grads, vae.trainable_variables))
        return total_loss

    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, _ in dataset:
            loss = train_step_vae(x_batch)
            epoch_loss += loss
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, VAE Loss: {epoch_loss/len(dataset):.6f}")

    # Phase 2: Train Feature Predictor
    print("\n--- Phase 2: Training Feature Predictor ---")
    fp_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    @tf.function
    def train_step_fp(x, p):
        z_mean, _, _ = encoder(x)
        with tf.GradientTape() as tape:
            p_pred = fp(z_mean)
            loss = tf.reduce_mean(losses.mse(p, p_pred))
        grads = tape.gradient(loss, fp.trainable_variables)
        fp_optimizer.apply_gradients(zip(grads, fp.trainable_variables))
        return loss

    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, p_batch in dataset:
            loss = train_step_fp(x_batch, p_batch)
            epoch_loss += loss
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, FP Loss: {epoch_loss/len(dataset):.6f}")

    # Phase 3: Train Latent Diffusion
    print("\n--- Phase 3: Training Latent Diffusion ---")
    ldm_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    @tf.function
    def train_step_ldm(x, p):
        z_0, _, _ = encoder(x)
        batch_size = tf.shape(z_0)[0]
        t = tf.random.uniform((batch_size,), minval=0, maxval=1000, dtype=tf.int32)
        noise = tf.random.normal(shape=tf.shape(z_0))
        
        # Simplified linear noise schedule
        alpha_t = tf.reshape(1.0 - tf.cast(t, tf.float32) / 1000.0, (-1, 1, 1, 1, 1))
        z_t = tf.sqrt(alpha_t) * z_0 + tf.sqrt(1.0 - alpha_t) * noise
        
        with tf.GradientTape() as tape:
            noise_pred = ldm(z_t, t, p)
            loss = tf.reduce_mean(losses.mse(noise, noise_pred))
        grads = tape.gradient(loss, ldm.trainable_variables)
        ldm_optimizer.apply_gradients(zip(grads, ldm.trainable_variables))
        return loss

    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, p_batch in dataset:
            loss = train_step_ldm(x_batch, p_batch)
            epoch_loss += loss
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, LDM Loss: {epoch_loss/len(dataset):.6f}")

    return vae, fp, ldm

if __name__ == "__main__":
    sample_names = [f'rand{i}' for i in range(1, 12)]
    skeletons, params, p_mean, p_std = load_bone_data(sample_names)
    
    if skeletons is not None:
        print(f"Dataset ready. Skeletons: {skeletons.shape}, Params: {params.shape}")
        vae, fp, ldm = train_pipeline(skeletons, params, epochs=100)
        
        # Save models
        vae.save("vae_bone_tf")
        fp.save("fp_bone_tf")
        # LDM is a subclassed model, saving weights
        ldm.save_weights("ldm_bone_tf_weights.h5")
        print("\nTraining complete and models saved.")
    else:
        print("No data files found. Please run thin_fin.py first.")
