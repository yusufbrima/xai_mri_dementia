"""
Extended transform for frame selection in MONAI DataLoader.

This module adds a custom transform to slice 3D medical images
and select only the informative frames for training.
"""

import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split

from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
    ScaleIntensityd, 
    ToTensord,
    RandRotated,
    Resized,
    RandFlipd,
    Transform
)
from monai.utils import ensure_tuple


class SliceSelectord(Transform):
    """
    Custom transform to select specific slices from a 3D medical image,
    removing empty or non-informative frames from the beginning and end.
    """
    
    def __init__(
        self, 
        keys: Union[str, List[str]], 
        skip_x_start: int = 0,
        skip_x_end: int = 0,
        skip_y_start: int = 0,
        skip_y_end: int = 0,
        skip_z_start: int = 0,
        skip_z_end: int = 0,
        allow_missing_keys: bool = False
    ):
        """
        Initialize the slice selector transform.
        
        Args:
            keys: Keys of the corresponding items to be transformed.
            skip_x_start: Number of frames to skip from the x axis start.
            skip_x_end: Number of frames to skip from the x axis end.
            skip_y_start: Number of frames to skip from the y axis start.
            skip_y_end: Number of frames to skip from the y axis end.
            skip_z_start: Number of frames to skip from the z axis start.
            skip_z_end: Number of frames to skip from the z axis end.
            allow_missing_keys: Don't raise exception if key is missing.
        """
        super().__init__()
        self.keys = ensure_tuple(keys)
        self.skip_x_start = skip_x_start
        self.skip_x_end = skip_x_end
        self.skip_y_start = skip_y_start
        self.skip_y_end = skip_y_end
        self.skip_z_start = skip_z_start
        self.skip_z_end = skip_z_end
        self.allow_missing_keys = allow_missing_keys
    
    def __call__(self, data: Dict) -> Dict:
        """
        Apply the transform to the data dictionary.
        
        Args:
            data: Dictionary containing the data to be transformed.
            
        Returns:
            Transformed dictionary with selected slices.
        """
        d = dict(data)
        for key in self.keys:
            if key in d:
                # Get the shape of the image (assuming channel-first: [C, H, W, D])
                img = d[key]
                if len(img.shape) != 4:
                    raise ValueError(f"Expected 4D image [C, H, W, D], got {img.shape}")
                
                channels, x_dim, y_dim, z_dim = img.shape
                
                # Ensure we're not trying to skip more frames than available
                valid_x_frames = x_dim - self.skip_x_start - self.skip_x_end
                if valid_x_frames <= 0:
                    raise ValueError(
                        f"Cannot skip {self.skip_x_start} start frames and {self.skip_x_end} end frames "
                        f"from image with x_dim {x_dim}"
                    )
                
                valid_y_frames = y_dim - self.skip_y_start - self.skip_y_end
                if valid_y_frames <= 0:
                    raise ValueError(
                        f"Cannot skip {self.skip_y_start} start frames and {self.skip_y_end} end frames "
                        f"from image with y_dim {y_dim}"
                    )
                valid_z_frames = z_dim - self.skip_z_start - self.skip_z_end
                if valid_z_frames <= 0:
                    raise ValueError(
                        f"Cannot skip {self.skip_z_start} start frames and {self.skip_z_end} end frames "
                        f"from image with z_dim {z_dim}"
                    )
                # Select only the frames we want to keep
                d[key] = img[
                    :,
                    self.skip_x_start:x_dim - self.skip_x_end,
                    self.skip_y_start:y_dim - self.skip_y_end,
                    self.skip_z_start:z_dim - self.skip_z_end
                ]
          
                # Log the slicing operation
                # print(f"SliceSelectord: Image shape changed from {img.shape} to {d[key].shape}")
            elif not self.allow_missing_keys:
                raise KeyError(f"Key {key} not found in data dictionary.")
        return d

class MedicalImageTransformFactory:
    """
    A factory class for creating MONAI transform pipelines,
    driven by a config dict (e.g. parsed from your YAML).
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the transform factory.
        
        Args:
            config: Configuration dictionary, e.g.
              {
                "data": {
                  "image_size": [128,128,128] or None,
                  "augmentations": ["flip","rotate"],
                  ...
                },
                "perform_slicing": True/False,
                "skip_start_frames": int,
                "skip_end_frames": int,
              }
        """
        self.config = config or {}
        self.skip_start = self.config.get('skip_start_frames', 0)
        self.skip_end = self.config.get('skip_end_frames', 0)
        self.skip_x_start = self.config.get('skip_x_start_frames', 0)
        self.skip_x_end = self.config.get('skip_x_end_frames', 0)
        self.skip_y_start = self.config.get('skip_y_start_frames', 0)
        self.skip_y_end = self.config.get('skip_y_end_frames', 0)
        self.skip_z_start = self.config.get('skip_z_start_frames', 0)
        self.skip_z_end = self.config.get('skip_z_end_frames', 0)
    
    def _get_image_size(self) -> Optional[Tuple[int, ...]]:
        size = self.config.get("image_size", None)
        return tuple(size) if size is not None else None

    def _get_augmentations(self) -> List:
        aug_list = self.config.get("augmentations", [])
        augs = []
        if "rotate" in aug_list:
            augs.append(RandRotated(keys=["image"], range_x=0.2, prob=0.5))
            # print("Rotation augmentation applied.")
        if "flip" in aug_list:
            # flip along two axes
            augs.append(RandFlipd(keys=["image"], spatial_axis=0, prob=0.5))
            augs.append(RandFlipd(keys=["image"], spatial_axis=1, prob=0.5))
            # print("Flip augmentation applied.")
        return augs

    def get_training_transforms(self) -> Compose:
        """
        Get transforms for training data, including optional resizing,
        slicing, and augmentations.
        """
        transforms = [
            LoadImaged(keys=['image']),
            EnsureChannelFirstd(keys=['image']),
        ]
        
        
        # Optional slicing
        if self.config.get('perform_slicing', False):
            transforms.append(
                SliceSelectord(
                    keys=['image'],
                    skip_x_start=self.skip_x_start,
                    skip_x_end=self.skip_x_end,
                    skip_y_start=self.skip_y_start,
                    skip_y_end=self.skip_y_end,
                    skip_z_start=self.skip_z_start,
                    skip_z_end=self.skip_z_end

                )
            )

        # Optional resize
        img_size = self._get_image_size()
        if img_size is not None:
            transforms.append(Resized(keys=['image'], spatial_size=img_size))
        

        if self.config.get("normalization") == "minmax":
            # Intensity normalization to [0, 1]
            transforms.append(ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0))
        
        #TODO: add  other normalization methods

        # Optional augmentations
        transforms.extend(self._get_augmentations())

        # Final tensor conversion (image + label)
        transforms.append(ToTensord(keys=['image', 'label']))
        return Compose(transforms)

    def get_validation_transforms(self) -> Compose:
        """
        Get transforms for validation data: optional resize/slicing,
        then scale+to-tensor (no augmentations).
        """
        transforms = [
            LoadImaged(keys=['image']),
            EnsureChannelFirstd(keys=['image']),
        ]
        

        if self.config.get('perform_slicing', False):
            transforms.append(
                SliceSelectord(
                    keys=['image'],
                    skip_x_start=self.skip_x_start,
                    skip_x_end=self.skip_x_end,
                    skip_y_start=self.skip_y_start,
                    skip_y_end=self.skip_y_end,
                    skip_z_start=self.skip_z_start,
                    skip_z_end=self.skip_z_end
                )
            )

        img_size = self._get_image_size()
        if img_size is not None:
            transforms.append(Resized(keys=['image'], spatial_size=img_size))
        
        if self.config.get("normalization") == "minmax":
            # Intensity normalization to [0, 1]
            transforms.append(ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0))
        
        #TODO: add  other normalization methods
        
        transforms.extend([
            ToTensord(keys=['image', 'label']),
        ])
        return Compose(transforms)

    def get_test_transforms(self) -> Compose:
        """
        Get transforms for test data: optional resize/slicing,
        then scale+to-tensor (no augmentations).
        """
        transforms = [
            LoadImaged(keys=['image']),
            EnsureChannelFirstd(keys=['image']),
        ]

        if self.config.get('perform_slicing', False):
            transforms.append(
                SliceSelectord(
                    keys=['image'],
                    skip_x_start=self.skip_x_start,
                    skip_x_end=self.skip_x_end,
                    skip_y_start=self.skip_y_start,
                    skip_y_end=self.skip_y_end,
                    skip_z_start=self.skip_z_start,
                    skip_z_end=self.skip_z_end
                )
            )
        
        img_size = self._get_image_size()
        if img_size is not None:
            transforms.append(Resized(keys=['image'], spatial_size=img_size))
        
        if self.config.get("normalization") == "minmax":
            # Intensity normalization to [0, 1]
            transforms.append(ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0))
        
        #TODO: add  other normalization methods

        transforms.extend([
            ToTensord(keys=['image', 'label']),
        ])
        return Compose(transforms)


# Example usage
if __name__ == "__main__":
    # Path to config
    pass