# scripts/run_pipeline.py

"""
Main script to run the complete ferroelectric SDG pipeline.
"""

import argparse
import yaml
from pathlib import Path
import subprocess
import sys
import logging


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def run_command(cmd: str, description: str):
    """Run a command and handle errors."""
    logger = logging.getLogger(__name__)
    logger.info(f"{description}...")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"{description} completed successfully")
        if result.stdout:
            logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{description} failed with error:")
        logger.error(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='Run ferroelectric SDG pipeline')
    parser.add_argument('--mode', choices=['all', 'prepare', 'train', 'generate', 'evaluate'],
                       default='all', help='Pipeline mode')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to config file')
    parser.add_argument('--data_path', type=str, default='data/kmc_run1_6nm_80dev_1.csv',
                       help='Path to input CSV data')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for generation')
    parser.add_argument('--num_synthetic', type=int, default=1000,
                       help='Number of synthetic samples to generate')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # Change to project directory
    project_dir = Path('/storage/home/hcoda1/6/cli872/scratch/work/SDG')
    
    # Load config
    config_path = project_dir / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Create necessary directories
    dirs = ['data/processed', 'checkpoints', 'logs', 'synthetic_data', 'results']
    for dir_path in dirs:
        (project_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
    success = True
    
    # Step 1: Data preparation
    if args.mode in ['all', 'prepare']:
        cmd = f"cd {project_dir} && python prepare_data.py --input {args.data_path} --output data/processed/"
        success = run_command(cmd, "Data preparation")
        if not success and args.mode == 'all':
            logger.error("Stopping pipeline due to data preparation failure")
            return 1
            
    # Step 2: Training
    if args.mode in ['all', 'train']:
        cmd = f"cd {project_dir} && python training/train.py"
        success = run_command(cmd, "Model training")
        if not success and args.mode == 'all':
            logger.error("Stopping pipeline due to training failure")
            return 1
            
    # Step 3: Generate synthetic data
    if args.mode in ['all', 'generate']:
        checkpoint = args.checkpoint or "checkpoints/best_model.pt"
        cmd = f"cd {project_dir} && python inference/generate.py --checkpoint {checkpoint} --num_samples {args.num_synthetic} --output_dir synthetic_data --visualize"
        success = run_command(cmd, "Synthetic data generation")
        if not success and args.mode == 'all':
            logger.error("Stopping pipeline due to generation failure")
            return 1
            
    logger.info("Pipeline completed successfully!" if success else "Pipeline completed with errors")
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())