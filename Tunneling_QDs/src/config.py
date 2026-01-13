"""
Configuration Module

Single source of truth for default parameters.
Stores 'default_params_5' and other configuration constants.
"""

# Default parameters for the quantum dot simulation
default_params_5 = {
    # Physical parameters
    'temperature': 100,  # mK
    'tunnel_coupling': 0.1,  # meV
    'charging_energy': 1.0,  # meV
    
    # Simulation parameters
    'voltage_range': (-100, 100),  # mV
    'resolution': 128,  # grid points
    'noise_level': 0.01,  # pink noise amplitude
    
    # Training parameters
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    
    # Data generation parameters
    'num_samples': 1000,
    'random_seed': 42,
}

# Additional configuration constants
DATA_DIR = 'data/'
MODEL_DIR = 'models/'
RESULTS_DIR = 'results/'
