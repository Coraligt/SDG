# prepare_data.py

"""
Prepare and validate the ferroelectric dataset with CSV output.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging
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
    
    # Save trap parameters as CSV
    trap_params_df = pd.DataFrame(
        df.iloc[:, :4].values,
        columns=['peak_density', 'thermal_ionization_mean', 'thermal_ionization_spread', 'relaxation_energy']
    )
    trap_params_df.to_csv(output_path / 'trap_parameters.csv', index=False)
    logger.info(f"Saved trap parameters to {output_path / 'trap_parameters.csv'}")
    
    # Save voltages as CSV
    voltages_df = pd.DataFrame(
        df.iloc[:, 4:6].values,
        columns=['voltage1', 'voltage2']
    )
    voltages_df.to_csv(output_path / 'voltages.csv', index=False)
    logger.info(f"Saved voltages to {output_path / 'voltages.csv'}")
    
    # Save device parameters as CSV
    device_params_df = pd.DataFrame(
        df.iloc[:, 6:8].values,
        columns=['pulsewidth', 'thickness']
    )
    device_params_df.to_csv(output_path / 'device_parameters.csv', index=False)
    logger.info(f"Saved device parameters to {output_path / 'device_parameters.csv'}")
    
    # Save cycles data as CSV
    cycles_df = pd.DataFrame(df.iloc[:, 8:].values)
    cycles_df.columns = [f'cycle_{i+1}' for i in range(cycles_df.shape[1])]
    cycles_df.to_csv(output_path / 'cycles_data.csv', index=False)
    logger.info(f"Saved cycles data to {output_path / 'cycles_data.csv'}")
    
    # Save breakdown cycles
    breakdown_df = pd.DataFrame({
        'sample_id': range(len(breakdown_cycles)),
        'breakdown_cycle': breakdown_cycles
    })
    breakdown_df.to_csv(output_path / 'breakdown_cycles.csv', index=False)
    logger.info(f"Saved breakdown cycles to {output_path / 'breakdown_cycles.csv'}")
    
    # Save complete processed data as single CSV (same format as input but cleaned)
    df['breakdown_cycle'] = breakdown_cycles
    df.to_csv(output_path / 'processed_complete.csv', index=False, header=False)
    logger.info(f"Saved complete processed data to {output_path / 'processed_complete.csv'}")
    
    # Save metadata
    metadata = {
        'num_samples': len(df),
        'num_trap_params': 4,
        'max_cycles': 2000,
        'breakdown_threshold': 200,
        'avg_breakdown_cycle': float(np.mean(breakdown_cycles)),
        'std_breakdown_cycle': float(np.std(breakdown_cycles)),
        'data_files': {
            'trap_parameters': 'trap_parameters.csv',
            'voltages': 'voltages.csv',
            'device_parameters': 'device_parameters.csv',
            'cycles_data': 'cycles_data.csv',
            'breakdown_cycles': 'breakdown_cycles.csv',
            'complete_data': 'processed_complete.csv'
        }
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Data prepared and saved to {output_path}")
    logger.info("All data saved in CSV format for easy access")


if __name__ == '__main__':
    main()