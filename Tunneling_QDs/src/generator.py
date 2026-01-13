"""
Data Generator Module

Functions for generating and saving double quantum dot (DQD) datasets
with pink noise and various parameter configurations.
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Dict, Any
from .config import default_params_5, DATA_DIR
from .physics import simulate_stability_diagram


def generate_pink_noise(shape: tuple, alpha: float = 1.0) -> np.ndarray:
    """
    Generate pink noise (1/f noise) for realistic experimental conditions.
    
    Args:
        shape: Shape of the output array
        alpha: Exponent for the power law (1.0 for pink noise)
        
    Returns:
        Array of pink noise values
    """
    # Generate white noise
    white_noise = np.random.randn(*shape)
    
    # Apply FFT
    fft = np.fft.fftn(white_noise)
    
    # Create frequency grid
    freqs = [np.fft.fftfreq(s) for s in shape]
    freq_grid = np.meshgrid(*freqs, indexing='ij')
    
    # Calculate frequency magnitude
    freq_mag = np.sqrt(sum(f**2 for f in freq_grid))
    freq_mag[freq_mag == 0] = 1  # Avoid division by zero
    
    # Apply 1/f^alpha filter
    fft_filtered = fft / (freq_mag ** (alpha / 2))
    
    # Inverse FFT to get pink noise
    pink_noise = np.fft.ifftn(fft_filtered).real
    
    # Normalize
    pink_noise = (pink_noise - pink_noise.mean()) / pink_noise.std()
    
    return pink_noise


def generate_single_sample(params: Dict[str, Any], 
                          add_noise: bool = True) -> np.ndarray:
    """
    Generate a single stability diagram sample.
    
    Args:
        params: Physical and simulation parameters
        add_noise: Whether to add pink noise to the data
        
    Returns:
        2D array representing a stability diagram
    """
    resolution = params.get('resolution', 128)
    
    # Generate voltage grid
    v_min, v_max = params.get('voltage_range', (-100, 100))
    voltage_grid = np.linspace(v_min, v_max, resolution)
    
    # Simulate the system
    # TODO: Replace with actual JAX simulation
    diagram = np.random.rand(resolution, resolution)
    
    # Add pink noise if requested
    if add_noise:
        noise_level = params.get('noise_level', 0.01)
        pink_noise = generate_pink_noise((resolution, resolution))
        diagram = diagram + noise_level * pink_noise
    
    return diagram


def generate_and_save_dqd_dataset(num_samples: int,
                                  filename: Optional[str] = None,
                                  params: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate and save a dataset of double quantum dot stability diagrams.
    
    Args:
        num_samples: Number of samples to generate
        filename: Output filename (default: 'dqd_dataset.h5')
        params: Physical parameters (default: use default_params_5)
        
    Returns:
        Path to the saved dataset file
    """
    if params is None:
        params = default_params_5.copy()
    
    if filename is None:
        filename = 'dqd_dataset.h5'
    
    # Create data directory if it doesn't exist
    data_path = Path(DATA_DIR)
    data_path.mkdir(exist_ok=True)
    
    output_path = data_path / filename
    
    # Generate samples
    print(f"Generating {num_samples} samples...")
    resolution = params.get('resolution', 128)
    
    with h5py.File(output_path, 'w') as f:
        # Create datasets
        data_dset = f.create_dataset('diagrams', 
                                     shape=(num_samples, resolution, resolution),
                                     dtype='float32')
        
        # Store metadata
        f.attrs['num_samples'] = num_samples
        f.attrs['resolution'] = resolution
        for key, value in params.items():
            if isinstance(value, (int, float, str)):
                f.attrs[key] = value
        
        # Generate and save samples
        for i in range(num_samples):
            diagram = generate_single_sample(params, add_noise=True)
            data_dset[i] = diagram
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} samples")
    
    print(f"Dataset saved to {output_path}")
    return str(output_path)
