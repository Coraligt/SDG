# prepare_data.py

"""
Prepare and validate the ferroelectric dataset.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging
import h5py
import json


def main():
    parser = argparse.ArgumentParser(description='Prepare ferroelectric data')
    parser.add_argument('--input', type=str, required=True, help='Input CSV path')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    df = pd.read_csv(args.input, header=None)
    
    # Validate data shape
    expected_cols = 2008
    if df.shape[1] != expected_cols:
        logger.warning(f"Expected {expected_cols} columns, got {df.shape[1]}")
        
    # Basic statistics
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Number of samples: {df.shape[0]}")
    
    # Check for breakdown patterns
    breakdown_cycles = []
    for idx, row in df.iterrows():
        cycle_data = row[8:].values
        valid_mask = ~pd.isna(cycle_data)
        if valid_mask.any():
            last_valid = np.where(valid_mask)[0][-1]
            breakdown_cycles.append(last_valid + 1)
        else:
            breakdown_cycles.append(0)
            
    logger.info(f"Average breakdown cycle: {np.mean(breakdown_cycles):.1f} ± {np.std(breakdown_cycles):.1f}")
    
    # Save processed data
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as HDF5 for faster loading
    with h5py.File(output_path / 'processed_data.h5', 'w') as f:
        f.create_dataset('trap_parameters', data=df.iloc[:, :4].values)
        f.create_dataset('voltages', data=df.iloc[:, 4:6].values)
        f.create_dataset('device_params', data=df.iloc[:, 6:8].values)
        f.create_dataset('cycles', data=df.iloc[:, 8:].values)
        f.create_dataset('breakdown_cycles', data=breakdown_cycles)
        
    # Save metadata
    metadata = {
        'num_samples': len(df),
        'num_trap_params': 4,
        'max_cycles': 2000,
        'breakdown_threshold': 200,
        'avg_breakdown_cycle': float(np.mean(breakdown_cycles)),
        'std_breakdown_cycle': float(np.std(breakdown_cycles))
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Data prepared and saved to {output_path}")


if __name__ == '__main__':
    main()