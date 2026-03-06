import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
import json
from scipy.ndimage import zoom
from thin_fin import render_3d_mat, scout

# ==========================================
# 1. CUSTOM LAYERS: FiLM (Required for loading)
# ==========================================

class FiLMLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(FiLMLayer, self).__init__(**kwargs)

    def call(self, inputs):
        x, conditioning = inputs 
        gamma = conditioning[:, :x.shape[-1]]
        beta = conditioning[:, x.shape[-1]:]
        gamma = tf.reshape(gamma, (-1, 1, 1, 1, x.shape[-1]))
        beta = tf.reshape(beta, (-1, 1, 1, 1, x.shape[-1]))
        return (1 + gamma) * x + beta

# ==========================================
# 2. GENERATION SCRIPT
# ==========================================

def generate():
    model_path = 'bone_inflation_model_tf.keras'
    stats_path = 'bone_inflation_stats.npz'
    sample_name = 'rand1'
    ske_path = f'ske_{sample_name}.npy'
    quant_path = f'quant_{sample_name}.json'

    # Check if files exist
    for p in [model_path, stats_path, ske_path, quant_path]:
        if not os.path.exists(p):
            print(f"Error: {p} not found.")
            return

    print("Loading model and data...")
    # Load model with custom FiLMLayer
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={'FiLMLayer': FiLMLayer},
        compile=False
    )
    
    # Load stats
    stats = np.load(stats_path)
    p_mean = stats['mean']
    p_std = stats['std']

    # Load skeleton
    ske = np.load(ske_path).astype(np.float32)
    
    # Load parameters from JSON
    with open(quant_path, 'r') as f:
        quant = json.load(f)
    param_keys = ["BV/TV", "porosity", "pore_size", "pBV/TV", "pTb.Th", "pTb.N", "rBV/TV", "rTb.Th"]
    params = np.array([[quant[k] for k in param_keys]]).astype(np.float32)
    params_norm = (params - p_mean) / p_std

    # Prepare input
    target_size = (100, 100, 100)
    factors = [t / s for t, s in zip(target_size, ske.shape)]
    ske_resized = zoom(ske, factors, order=1)
    # Threshold skeleton after resizing
    ske_binary = (ske_resized > 0.5).astype(np.float32)
    ske_input = np.expand_dims(np.expand_dims(ske_binary, -1), 0)

    print("Predicting structure...")
    structure_pred = model.predict({"skeleton": ske_input, "params": params_norm})
    structure_binary = (structure_pred[0, :, :, :, 0] > 0.5).astype(np.uint8)

    # Render Skeleton
    print("Generating SCAD for skeleton...")
    # Using smaller cube_size for better visualization in OpenSCAD if needed, 
    # but thin_fin.py default is 0.1
    ske_obj = render_3d_mat(ske_binary, cube_size=1.0)
    scout('generated_skeleton', ske_obj)

    # Render Structure
    print("Generating SCAD for structure...")
    struct_obj = render_3d_mat(structure_binary, cube_size=1.0)
    scout('generated_structure', struct_obj)

    print("\nGeneration complete.")
    print("Files created: generated_skeleton.scad, generated_structure.scad")

if __name__ == "__main__":
    generate()
