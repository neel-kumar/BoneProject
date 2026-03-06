import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
import json
import time
import subprocess
from scipy.ndimage import zoom
from solid import *

# ==========================================
# 1. RENDER FUNCTIONS (Copied from thin_fin.py)
# ==========================================

def scout(filename, obj, to_stl=False):
    st = time.time()
    # Ensure the directory exists or just save to current
    scad_render_to_file(obj, filename + '.scad')
    print('to openscad: ' + filename + '.scad')
    if to_stl:
        print("to stl")
        # Note: Path to openscad.exe might need adjustment depending on OS
        subprocess.run(['openscad', '-o', filename + '.stl', './' + filename + '.scad'])
    print(f"scout elapsed: {time.time() - st:.2f} seconds")

def render_3d_mat(mat, cube_size=1.0):
    positions = np.argwhere(mat)
    if len(positions) == 0:
        return cube(0)
    cells = None
    for z, y, x in positions:
        # We use the voxel coordinates scaled by cube_size
        c = translate([x * cube_size, y * cube_size, z * cube_size])(cube(cube_size))
        cells = c if cells is None else cells + c
    return cells

# ==========================================
# 2. CUSTOM MODEL LAYER (Copied from inflation_model_tf.py)
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
# 3. GENERATION LOGIC
# ==========================================

def generate_and_export():
    # Paths
    model_path = 'bone_inflation_model_tf.keras'
    stats_path = 'bone_inflation_stats.npz'
    # Use rand1 as the example
    sample_name = 'rand1'
    ske_path = f'ske_{sample_name}.npy'
    quant_path = f'quant_{sample_name}.json'

    # Check for required files
    for p in [model_path, stats_path, ske_path, quant_path]:
        if not os.path.exists(p):
            print(f"Critical Error: Required file {p} not found.")
            return

    print("--- Loading Model and Statistics ---")
    # Load model with custom layer mapping
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={'FiLMLayer': FiLMLayer},
        compile=False
    )
    
    # Load normalization stats
    stats = np.load(stats_path)
    p_mean = stats['mean']
    p_std = stats['std']

    print(f"--- Processing {sample_name} ---")
    # Load skeleton and parameters
    ske = np.load(ske_path).astype(np.float32)
    with open(quant_path, 'r') as f:
        quant = json.load(f)
    
    param_keys = ["BV/TV", "porosity", "pore_size", "pBV/TV", "pTb.Th", "pTb.N", "rBV/TV", "rTb.Th"]
    params = np.array([[quant[k] for k in param_keys]]).astype(np.float32)
    
    # Normalize parameters
    params_norm = (params - p_mean) / p_std

    # Prepare skeleton input (100x100x100)
    target_size = (100, 100, 100)
    factors = [t / s for t, s in zip(target_size, ske.shape)]
    ske_resized = zoom(ske, factors, order=1)
    ske_binary = (ske_resized > 0.5).astype(np.float32)
    
    # Model expects batch and channel dims: (1, 100, 100, 100, 1)
    ske_input = np.expand_dims(np.expand_dims(ske_binary, -1), 0)

    print("Predicting structure via Inflation Model...")
    structure_pred = model.predict({"skeleton": ske_input, "params": params_norm})
    # Threshold prediction to get binary volume
    structure_binary = (structure_pred[0, :, :, :, 0] > 0.5).astype(np.uint8)

    # Export Skeleton
    print("Generating Exports: skeleton...")
    # Save as .npy binary matrix
    np.save('exported_skeleton.npy', ske_binary)
    print("  - Saved: exported_skeleton.npy")
    
    # Using cube_size=1.0 for standard unit size
    ske_obj = render_3d_mat(ske_binary, cube_size=1.0)
    scout('exported_skeleton', ske_obj)

    # Export Structure
    print("Generating Exports: structure...")
    # Save as .npy binary matrix
    np.save('exported_structure.npy', structure_binary)
    print("  - Saved: exported_structure.npy")
    
    struct_obj = render_3d_mat(structure_binary, cube_size=1.0)
    scout('exported_structure', struct_obj)

    print("\nSuccess! Generated:")
    print("  - exported_skeleton.scad / .npy")
    print("  - exported_structure.scad / .npy")

if __name__ == "__main__":
    generate_and_export()
