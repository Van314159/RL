"""
DataLoader Module

PyTorch/TensorFlow Dataset classes for reading .h5 files
and preparing data for model training.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader


class DQDDataset(Dataset):
    """
    PyTorch Dataset for loading double quantum dot stability diagrams.
    """
    
    def __init__(self, 
                 h5_file_path: str,
                 transform: Optional[callable] = None,
                 normalize: bool = True):
        """
        Initialize the dataset.
        
        Args:
            h5_file_path: Path to the .h5 file
            transform: Optional transform to apply to samples
            normalize: Whether to normalize the data
        """
        self.h5_file_path = h5_file_path
        self.transform = transform
        self.normalize = normalize
        
        # Open file to get metadata
        with h5py.File(h5_file_path, 'r') as f:
            self.num_samples = f.attrs['num_samples']
            self.resolution = f.attrs['resolution']
            
            # Calculate normalization statistics if needed
            if self.normalize:
                data = f['diagrams'][:]
                self.mean = np.mean(data)
                self.std = np.std(data)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (input_tensor, target_tensor)
        """
        with h5py.File(self.h5_file_path, 'r') as f:
            diagram = f['diagrams'][idx]
        
        # Normalize if requested
        if self.normalize:
            diagram = (diagram - self.mean) / (self.std + 1e-8)
        
        # Convert to torch tensor
        diagram = torch.from_numpy(diagram).float()
        
        # Add channel dimension
        diagram = diagram.unsqueeze(0)
        
        # Apply transform if provided
        if self.transform:
            diagram = self.transform(diagram)
        
        # For now, use the same diagram as target (e.g., for autoencoder)
        # Modify this based on your specific task
        target = diagram
        
        return diagram, target


def create_dataloader(h5_file_path: str,
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     **kwargs) -> DataLoader:
    """
    Create a PyTorch DataLoader for the DQD dataset.
    
    Args:
        h5_file_path: Path to the .h5 file
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        **kwargs: Additional arguments passed to DQDDataset
        
    Returns:
        PyTorch DataLoader
    """
    dataset = DQDDataset(h5_file_path, **kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def get_dataset_info(h5_file_path: str) -> dict:
    """
    Get information about a dataset stored in an .h5 file.
    
    Args:
        h5_file_path: Path to the .h5 file
        
    Returns:
        Dictionary containing dataset information
    """
    with h5py.File(h5_file_path, 'r') as f:
        info = {
            'num_samples': f.attrs['num_samples'],
            'resolution': f.attrs['resolution'],
            'shape': f['diagrams'].shape,
            'dtype': f['diagrams'].dtype,
        }
        
        # Add all stored attributes
        for key in f.attrs.keys():
            if key not in info:
                info[key] = f.attrs[key]
    
    return info
