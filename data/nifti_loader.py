"""
Extended transform for frame selection in MONAI DataLoader.

This module adds a custom transform to slice 3D medical images
and select only the informative frames for training.
"""

import pandas as pd
import torch
from typing import Dict, List, Optional, Union, Tuple, Callable
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
from monai.data import Dataset, DataLoader
from monai.utils import ensure_tuple

from config import config_loader


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



class MedicalImageDatasetSplitter:
    """
    A class responsible for loading and splitting medical image datasets.
    
    This class handles loading dataset information from CSV files and
    creating appropriate train/validation/test splits.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the dataset splitter with a configuration file.
        
        Args:
            config : Configuration dictionary containing paths and parameters.
        """
        # Load the configuration
        self.config = config
        
        # Set up paths and column names from config
        self.base_dir = Path(self.config['data']['root_dir'])
        self.file_path  = self.base_dir / self.config['data']['annotations_file']
        self.class_column = self.config['data']['class_column']
        self.filename_column = self.config['data']['filename_column']
        self.value_to_filter =  self.config['data'].get('filter_class_values', [])
        
        # Get split ratios from config or use defaults
        self.val_ratio = self.config['data'].get('val_ratio', 0.15)
        self.test_ratio = self.config['data'].get('test_ratio', 0.15)
        self.stratify = self.config['data'].get('stratify', True)
        self.random_seed = self.config['data'].get('random_seed', 42)
        self.drop_empty = self.config['data'].get('drop_na', False)
        self.annotations_file_format = self.config['data'].get('annotations_file_format', 'csv')
        
        # Dataset components
        self.df = None
        self.class_to_idx = None
        self.idx_to_class = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
        # Load and split the dataset
        self.load_and_split_dataset()
    
    def load_and_split_dataset(self) -> None:
        """
        Load the dataset information from the annotation file and create train/val/test splits.
        
        Reads the annotation file, creates a mapping from class names to indices,
        and splits the data into training, validation, and test sets.
        """
        # Check if the anotation file exists
        if not self.file_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.file_path}")
        
        # Check if the file format is supported
        if self.annotations_file_format not in ['csv', 'xlsx', 'xls']:
            raise ValueError(f"Unsupported file format: {self.annotations_file_format}. Supported formats are: csv, xlsx, xls.")
        if self.annotations_file_format == 'csv':
            self.df = pd.read_csv(self.file_path)
        elif self.annotations_file_format in ['xlsx', 'xls']:
            self.df = pd.read_excel(self.file_path, sheet_name=0)
        if self.drop_empty:
            # print count of empty labels
            empty_count = self.df[self.class_column].isna().sum()
            print(f"Number of empty labels: {empty_count}")
            if empty_count > 0:
                print("Dropping rows with empty labels.")
                # Drop rows with empty labels
                self.df = self.df[self.df[self.class_column].notna()]
        print("Checking filtering status...")
        if len(self.value_to_filter) > 0:
            print(f"Filtering dataset to exclude class(es): {self.value_to_filter}")
            # Filter out unwanted classes
            # check that the values to unique values in the class column minus the values to filter >= 2
            unique_classes = self.df[self.class_column].unique()

            # check if the values to filter are in the dataframe
            if not all(val in unique_classes for val in self.value_to_filter):
                raise ValueError("Some values to filter are not present in the dataframe. Please check the values.")

            if len(unique_classes) - len(self.value_to_filter) < 2:
                raise ValueError("Not enough classes left after filtering. Please reduce the number of classes to filter.")
            self.df = self.df[~self.df[self.class_column].isin(self.value_to_filter)]
        
            print(f"Remaining classes: {self.df[self.class_column].unique()}")
        else:
            print("No filtering applied.")
        
        # Create class mapping
        self.class_to_idx = {
            c: i for i, c in enumerate(self.df[self.class_column].unique())
        }
        self.idx_to_class = {
            i: c for c, i in self.class_to_idx.items()
        }
        
        # Create splits using sklearn's train_test_split
        # First split off the test set
        train_val_df, self.test_df = train_test_split(
            self.df, 
            test_size=self.test_ratio,
            stratify=self.df[self.class_column] if self.stratify else None,
            random_state=self.random_seed
        )
        
        # Then split the remaining data into train and validation
        val_size_adjusted = self.val_ratio / (1 - self.test_ratio)
        self.train_df, self.val_df = train_test_split(
            train_val_df, 
            test_size=val_size_adjusted,
            stratify=train_val_df[self.class_column] if self.stratify else None,
            random_state=self.random_seed
        )
        
        # Print dataset information
        print(f"Loaded dataset with {len(self.df)} total samples")
        print(f"Training: {len(self.train_df)} samples")
        print(f"Validation: {len(self.val_df)} samples")
        print(f"Testing: {len(self.test_df)} samples")
        print(f"Classes: {self.class_to_idx}")
    
    def get_split_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get the split dataframes.
        
        Returns:
            Tuple containing train, validation, and test dataframes
        """
        return self.train_df, self.val_df, self.test_df
    
    def get_class_info(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Get the class mapping information.
        
        Returns:
            Tuple containing:
                - Dictionary mapping class names to indices
                - Dictionary mapping indices to class names
        """
        return self.class_to_idx, self.idx_to_class
    
    def get_num_classes(self) -> int:
        """
        Get the number of classes in the dataset.
        
        Returns:
            Number of unique classes
        """
        return len(self.class_to_idx)
    
    def get_base_dir(self) -> Path:
        """
        Get the base directory for image files.
        
        Returns:
            Path object representing the base directory
        """
        return self.base_dir
    
    def get_config(self) -> Dict:
        """
        Get the configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config


class MonaiDatasetCreator:
    """
    A class responsible for creating MONAI Datasets from dataframes.
    
    This class handles the conversion of dataframes to MONAI datasets.
    """
    
    def __init__(self, dataset_splitter: MedicalImageDatasetSplitter):
        """
        Initialize the dataset creator with a dataset splitter.
        
        Args:
            dataset_splitter: Instance of MedicalImageDatasetSplitter
        """
        self.dataset_splitter = dataset_splitter
        self.base_dir = dataset_splitter.get_base_dir()
        self.class_to_idx, _ = dataset_splitter.get_class_info()
        self.train_df, self.val_df, self.test_df = dataset_splitter.get_split_dataframes()
        self.filename_column = dataset_splitter.filename_column
        self.class_column = dataset_splitter.class_column
        self.config = dataset_splitter.get_config()
        
        # Create transform factory with configs for frame selection
        transform_config = {
            'skip_start_frames': self.config['data'].get('skip_start_frames', 0),
            'skip_end_frames': self.config['data'].get('skip_end_frames', 0),
            'skip_x_start_frames': self.config['data'].get('skip_x_start_frames', 0),
            'skip_x_end_frames': self.config['data'].get('skip_x_end_frames', 0),
            'skip_y_start_frames': self.config['data'].get('skip_y_start_frames', 0),
            'skip_y_end_frames': self.config['data'].get('skip_y_end_frames', 0),
            'skip_z_start_frames': self.config['data'].get('skip_z_start_frames', 0),
            'skip_z_end_frames': self.config['data'].get('skip_z_end_frames', 0),
            'perform_slicing': self.config['data'].get('perform_slicing', True),
            'image_size': self.config['data'].get('image_size', None),
            'normalization': self.config['data'].get('normalization', None),
            'augmentations': self.config['data'].get('augmentations', [])
        }
        self.transform_factory = MedicalImageTransformFactory(transform_config)
    
    def _prepare_data_list(self, df: pd.DataFrame) -> List[Dict[str, Union[str, torch.Tensor]]]:
        """
        Prepare the data list for MONAI Dataset from a DataFrame.
        
        Args:
            df: Pandas DataFrame containing file paths and labels
            
        Returns:
            List of dictionaries with image paths and labels
        """
        return [
            {
                'image': str(self.base_dir / p), 
                'label': torch.tensor(self.class_to_idx[l], dtype=torch.int64)
            }
            for p, l in zip(
                df[self.filename_column], 
                df[self.class_column]
            )
        ]
    
    def create_train_dataset(self) -> Dataset:
        """
        Create a MONAI Dataset for the training set.
        
        Returns:
            MONAI Dataset for training
        """
        data_list = self._prepare_data_list(self.train_df)
        return Dataset(data=data_list, transform=self.transform_factory.get_training_transforms())
    
    def create_val_dataset(self) -> Dataset:
        """
        Create a MONAI Dataset for the validation set.
        
        Returns:
            MONAI Dataset for validation
        """
        data_list = self._prepare_data_list(self.val_df)
        return Dataset(data=data_list, transform=self.transform_factory.get_validation_transforms())
    
    def create_test_dataset(self) -> Dataset:
        """
        Create a MONAI Dataset for the test set.
        
        Returns:
            MONAI Dataset for testing
        """
        data_list = self._prepare_data_list(self.test_df)
        return Dataset(data=data_list, transform=self.transform_factory.get_test_transforms())
    
    def create_datasets(self) -> Dict[str, Dataset]:
        """
        Create all three datasets.
        
        Returns:
            Dictionary containing train, val, and test datasets
        """
        return {
            'train': self.create_train_dataset(),
            'val': self.create_val_dataset(),
            'test': self.create_test_dataset()
        }


class MonaiDataLoaderManager:
    """
    A class responsible for creating and managing MONAI DataLoaders.
    
    This class handles the creation of DataLoaders from MONAI Datasets.
    """
    
    def __init__(self, dataset_creator: MonaiDatasetCreator, config: Dict):
        """
        Initialize the DataLoader manager with a dataset creator and config.
        
        Args:
            dataset_creator: Instance of MonaiDatasetCreator
            config: Configuration dictionary containing DataLoader parameters
        """
        self.dataset_creator = dataset_creator
        
        # Load the configuration
        self.config = config
        
        # DataLoader parameters
        self.batch_size = self.config['data']['batch_size']
        self.num_workers = self.config['data'].get('num_workers', 4)
        self.pin_memory = self.config['data'].get('pin_memory', True)
    
    def create_train_dataloader(self, batch_size: Optional[int] = None, 
                             num_workers: Optional[int] = None) -> DataLoader:
        """
        Create a MONAI DataLoader for the training set.
        
        Args:
            batch_size: Number of samples per batch
            num_workers: Number of subprocesses for data loading
            
        Returns:
            MONAI DataLoader object for training
        """
        batch_size = batch_size or self.batch_size
        num_workers = num_workers or self.num_workers
        
        return DataLoader(
            dataset=self.dataset_creator.create_train_dataset(),
            batch_size=batch_size,
            shuffle=True,  # Always shuffle training data
            num_workers=num_workers,
            pin_memory=self.pin_memory,
        )
    
    def create_val_dataloader(self, batch_size: Optional[int] = None, 
                           num_workers: Optional[int] = None) -> DataLoader:
        """
        Create a MONAI DataLoader for the validation set.
        
        Args:
            batch_size: Number of samples per batch
            num_workers: Number of subprocesses for data loading
            
        Returns:
            MONAI DataLoader object for validation
        """
        batch_size = batch_size or self.batch_size
        num_workers = num_workers or self.num_workers
        
        return DataLoader(
            dataset=self.dataset_creator.create_val_dataset(),
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle validation data
            num_workers=num_workers,
            pin_memory=self.pin_memory,
        )
    
    def create_test_dataloader(self, batch_size: Optional[int] = None, 
                            num_workers: Optional[int] = None) -> DataLoader:
        """
        Create a MONAI DataLoader for the test set.
        
        Args:
            batch_size: Number of samples per batch
            num_workers: Number of subprocesses for data loading
            
        Returns:
            MONAI DataLoader object for testing
        """
        batch_size = batch_size or self.batch_size
        num_workers = num_workers or self.num_workers
        
        return DataLoader(
            dataset=self.dataset_creator.create_test_dataset(),
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle test data
            num_workers=num_workers,
            pin_memory=self.pin_memory,
        )
    
    def get_dataloaders(self, batch_size: Optional[int] = None, 
                      num_workers: Optional[int] = None) -> Dict[str, DataLoader]:
        """
        Create and return all three dataloaders.
        
        Args:
            batch_size: Number of samples per batch
            num_workers: Number of subprocesses for data loading
            
        Returns:
            Dictionary containing train, val, and test dataloaders
        """
        return {
            'train': self.create_train_dataloader(batch_size, num_workers),
            'val': self.create_val_dataloader(batch_size, num_workers),
            'test': self.create_test_dataloader(batch_size, num_workers)
        }


# Example usage
if __name__ == "__main__":
    # Path to config
    pass