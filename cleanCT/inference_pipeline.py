import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import json
import os
import argparse

# ==========================================
# 1. CUSTOM LAYERS & MODELS (Required for loading)
# ==========================================

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1:]
        epsilon = tf.random.normal(shape=tf.concat([[batch], dim], axis=0))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class FiLMLayer(layers.Layer):
    def call(self, inputs):
        x, conditioning = inputs
        gamma = conditioning[:, :x.shape[-1]]
        beta = conditioning[:, x.shape[-1]:]
        gamma = tf.reshape(gamma, (-1, 1, 1, 1, x.shape[-1]))
        beta = tf.reshape(beta, (-1, 1, 1, 1, x.shape[-1]))
        return (1 + gamma) * x + beta

class ConditionalLDM(models.Model):
    def __init__(self, latent_ch=4, param_dim=8):
        super().__init__()
        self.time_mlp = models.Sequential([layers.Dense(128, activation="swish"), layers.Dense(128)])
        self.param_mlp = models.Sequential([layers.Dense(128, activation="swish"), layers.Dense(128)])
        self.down = layers.Conv3D(128, 3, padding="same")
        self.up = layers.Conv3D(latent_ch, 3, padding="same")
        self.cond_layer = layers.Dense(256)

    def call(self, x, t, p):
        t_emb = self.time_mlp(tf.expand_dims(tf.cast(t, tf.float32), -1))
        p_emb = self.param_mlp(p)
        context = tf.concat([t_emb, p_emb], axis=-1)
        cond = self.cond_layer(context)
        cond = tf.reshape(cond, (-1, 1, 1, 1, 256))
        gamma, beta = tf.split(cond, 2, axis=-1)
        h = self.down(x)
        h = gamma * h + beta
        return self.up(tf.nn.relu(h))

# ==========================================
# 2. INFERENCE PIPELINE
# ==========================================

def run_inference(json_path, output_prefix="generated"):
    # 1. Load Parameters
    param_keys = ["BV/TV", "porosity", "pore_size", "pBV/TV", "pTb.Th", "pTb.N", "rBV/TV", "rTb.Th"]
    with open(json_path, 'r') as f:
        quant = json.load(f)
    params = np.array([[quant[k] for k in param_keys]], dtype=np.float32)

    # 2. Normalize Parameters
    if not os.path.exists("bone_inflation_stats.npz"):
        print("Error: bone_inflation_stats.npz not found.")
        return
    stats = np.load("bone_inflation_stats.npz")
    p_mean, p_std = stats['mean'], stats['std']
    params_norm = (params - p_mean) / p_std
    params_tf = tf.convert_to_tensor(params_norm)

    # 3. Load Models
    print("Loading models...")
    vae = models.load_model("vae_bone_tf.keras", custom_objects={"Sampling": Sampling})
    decoder = vae.get_layer("decoder")
    
    ldm = ConditionalLDM(latent_ch=4, param_dim=8)
    # Initialize weights by calling once
    ldm(tf.zeros((1, 25, 25, 25, 4)), tf.zeros((1,), dtype=tf.int32), tf.zeros((1, 8)))
    ldm.load_weights("ldm_bone_tf_weights.weights.h5")
    
    inflation_model = models.load_model("bone_inflation_model_tf.keras", custom_objects={"FiLMLayer": FiLMLayer})

    # 4. Step 1: LDM Sampling (Latent Generation)
    print("Generating latent representation...")
    z_t = tf.random.normal(shape=(1, 25, 25, 25, 4))
    for t_idx in reversed(range(1000)):
        t = tf.constant([t_idx], dtype=tf.int32)
        noise_pred = ldm(z_t, t, params_tf)
        
        # Simple DDIM-like step based on linear schedule alpha_t = 1 - t/1000
        alpha_t = 1.0 - t_idx / 1000.0
        if t_idx > 0:
            alpha_prev = 1.0 - (t_idx - 1) / 1000.0
            # Estimate z0
            z0_recon = (z_t - tf.sqrt(1.0 - alpha_t) * noise_pred) / tf.sqrt(alpha_t)
            # Predict z_{t-1}
            z_t = tf.sqrt(alpha_prev) * z0_recon + tf.sqrt(1.0 - alpha_prev) * noise_pred
        else:
            z_t = (z_t - tf.sqrt(1.0 - alpha_t) * noise_pred) / tf.sqrt(alpha_t)

    # 5. Step 2: Decode Latent to Skeleton
    print("Decoding skeleton...")
    skeleton = decoder(z_t).numpy()
    ske_save_path = f"{output_prefix}_skeleton.npy"
    np.save(ske_save_path, skeleton[0, ..., 0])
    print(f"Skeleton saved to {ske_save_path}")

    # 6. Step 3: Inflation (Final Structure)
    print("Generating final structure...")
    # Inflation model expects {"skeleton": ..., "params": ...}
    final_structure = inflation_model.predict({"skeleton": skeleton, "params": params_norm})
    struct_save_path = f"{output_prefix}_structure.npy"
    np.save(struct_save_path, final_structure[0, ..., 0])
    print(f"Final structure saved to {struct_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference pipeline for BoneProject")
    parser.add_argument("json_path", help="Path to the input JSON parameter file")
    parser.add_argument("--output", default="generated", help="Prefix for output files")
    args = parser.parse_args()
    
    run_inference(args.json_path, args.output)
