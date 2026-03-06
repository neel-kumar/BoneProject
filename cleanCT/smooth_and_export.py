import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndimage
import torch
import os
import trimesh
from skimage import measure # Fallback for mesh if specialized GPU meshers aren't installed

def gpu_smooth_voxel_data(volume_np, sigma=1.2):
    """
    Smoothens voxel data on the GPU using a 3D Gaussian filter.
    
    Parameters:
    -----------
    volume_np : ndarray
        The 3D binary or probability volume.
    sigma : float
        The smoothing strength (higher = smoother, but loses detail).
    """
    # Move data to GPU
    volume_gpu = cp.asarray(volume_np, dtype=cp.float32)
    
    # Fast 3D Gaussian Smoothing (GPU)
    # This removes the "blocky" voxel artifacts
    smoothed_gpu = ndimage.gaussian_filter(volume_gpu, sigma=sigma)
    
    # Return to CPU for mesh extraction (most meshers need CPU pointers or PyTorch)
    return cp.asnumpy(smoothed_gpu)

def export_to_stl(volume, filename, threshold=0.5, voxel_size=1.0):
    """
    Extracts a surface mesh and saves it as an STL.
    """
    print(f"Extracting surface for {filename}...")
    
    # Marching Cubes (CPU version is standard, but we've already 
    # done the heavy lifting of smoothing on the GPU)
    verts, faces, normals, values = measure.marching_cubes(volume, level=threshold)
    
    # Create mesh object
    mesh = trimesh.Trimesh(vertices=verts * voxel_size, faces=faces)
    
    # Save as STL
    mesh.export(filename)
    print(f"Saved: {filename}")

def process_sample(sample_name, sigma=1.0):
    """
    Processes a specific sample: loads, smoothens, and exports to STL.
    """
    ske_file = f'ske_{sample_name}.npy'
    
    if not os.path.exists(ske_file):
        print(f"File {ske_file} not found.")
        return

    # 1. Load data
    data = np.load(ske_file)
    
    # 2. Smooth on GPU
    print(f"Smoothing {sample_name} on GPU (sigma={sigma})...")
    smoothed = gpu_smooth_voxel_data(data, sigma=sigma)
    
    # 3. Export to STL
    export_to_stl(smoothed, f"{sample_name}_smooth.stl")

if __name__ == "__main__":
    # Example: Process 'rand1'
    # You can loop through your files here
    process_sample("rand1", sigma=1.2)
    
    # If you have the original voxel submasks (from s01_voxel.npy), 
    # you can pass them here too.
    print("\nProcessing complete. Note: Ensure 'cupy' is installed for GPU support.")
