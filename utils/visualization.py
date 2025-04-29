import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os


def show_slice_advanced(volume, z: int):
    """
    Display the z-th slice of a 3D volume.

    Parameters:
        volume: 3D numpy array of shape (H, W, D)
        z:       Index of the slice along the z-dimension
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(volume[:, :, z], cmap='gray')
    plt.title(f"Slice {z+1}/{volume.shape[2]}")
    plt.axis('off')
    plt.show()