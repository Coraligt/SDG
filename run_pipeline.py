# run_pipeline.py

"""
Main script to run the complete ferroelectric SDG pipeline with enhanced training and analysis.
"""

import argparse
import yaml
from pathlib import Path
import subprocess
import sys
import logging
import time
from datetime import datetime


def setup_logging():
    """Setup logging configuration."""
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create a unique log file for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'pipeline_{timestamp}.log'
    
    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


def run_command(cmd: str, description: str, timeout: int = None):
    """Run a command and handle errors."""
    logger = logging.getLogger(__name__)
    logger.info(f"{description}...")
    logger.info(f"Command: {cmd}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"{description} completed successfully in {elapsed_time:.1f} seconds")
        
        if result.stdout:
            logger.debug("STDOUT:")
            logger.debug(result.stdout)
            
        return True
        
    except subprocess.TimeoutExpired:
        logger.error(f"{description} timed out after {timeout} seconds")
        return False
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        logger.error(f"{description} failed after {elapsed_time:.1f} seconds")
        logger.error(f"Return code: {e.returncode}")
        
        if e.stdout:
            logger.error("STDOUT:")
            logger.error(e.stdout)
            
        if e.stderr:
            logger.error("STDERR:")
            logger.error(e.stderr)
            
        return False


def check_prerequisites(project_dir: Path, logger):
    """Check if all required files and directories exist."""
    logger.info("Checking prerequisites...")
    
    # Check for required files
    required_files = [
        'prepare_data.py',
        'training/train.py',
        'inference/generate.py',
        'analyze_training_results.py',
        'models/fe_surrogate.py',
        'models/losses.py',
        'physics/kmc_physics.py',
        'ferroelectric_dataset.py',
        'config/training_config.yaml'
    ]
    
    all_exist = True
    for file in required_files:
        file_path = project_dir / file
        if not file_path.exists():
            logger.error(f"Required file missing: {file}")
            all_exist = False
        else:
            logger.debug(f"Found: {file}")
            
    if not all_exist:
        logger.error("Some required files are missing. Please ensure all files are in place.")
        return False
        
    logger.info("All prerequisites satisfied")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run ferroelectric SDG pipeline with enhanced features')
    parser.add_argument('--mode', 
                       choices=['all', 'prepare', 'train', 'generate', 'analyze', 'train_analyze'],
                       default='all', 
                       help='Pipeline mode (all runs everything, train_analyze runs training + analysis)')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to config file')
    parser.add_argument('--data_path', type=str, default='data/kmc_run1_6nm_80dev_1.csv',
                       help='Path to input CSV data')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for generation')
    parser.add_argument('--num_synthetic', type=int, default=1000,
                       help='Number of synthetic samples to generate')
    parser.add_argument('--use_enhanced_train', action='store_true', default=True,
                       help='Use enhanced training with detailed metrics tracking')
    parser.add_argument('--skip_analysis', action='store_true', default=False,
                       help='Skip analysis after training')
    parser.add_argument('--timeout', type=int, default=7200,
                       help='Timeout for each step in seconds (default: 2 hours)')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # Log script start
    logger.info("="*60)
    logger.info("FERROELECTRIC SDG PIPELINE - ENHANCED VERSION")
    logger.info("="*60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Enhanced training: {args.use_enhanced_train}")
    
    # Change to project directory
    project_dir = Path('/storage/home/hcoda1/6/cli872/scratch/work/SDG')
    
    # Check prerequisites
    if not check_prerequisites(project_dir, logger):
        return 1
    
    # Load config
    config_path = project_dir / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Training for {config['training']['num_epochs']} epochs")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    
    # Create necessary directories
    dirs = [
        'data/processed', 
        'checkpoints', 
        'checkpoints/metrics',
        'checkpoints/metrics/predictions',
        'logs', 
        'synthetic_data', 
        'results',
        'analysis_results'
    ]
    
    for dir_path in dirs:
        (project_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
    logger.info("Created necessary directories")
    
    # Track overall success
    pipeline_success = True
    step_times = {}
    
    # Step 1: Data preparation
    if args.mode in ['all', 'prepare']:
        step_start = time.time()
        cmd = f"cd {project_dir} && python prepare_data.py --input {args.data_path} --output data/processed/"
        success = run_command(cmd, "Data preparation", timeout=args.timeout)
        step_times['prepare'] = time.time() - step_start
        
        if not success:
            logger.error("Data preparation failed")
            if args.mode == 'all':
                return 1
            pipeline_success = False
            
    # Step 2: Training (enhanced or standard)
    if args.mode in ['all', 'train', 'train_analyze']:
        step_start = time.time()
        
        if args.use_enhanced_train:
            # Use enhanced training with detailed metrics
            train_script = "training/train.py"
            logger.info("Using enhanced training with detailed metrics tracking")
        else:
            # Use standard training
            train_script = "training/train.py"
            logger.info("Using standard training")
            
        cmd = f"cd {project_dir} && python {train_script}"
        success = run_command(cmd, "Model training", timeout=args.timeout)
        step_times['train'] = time.time() - step_start
        
        if not success:
            logger.error("Model training failed")
            if args.mode in ['all', 'train_analyze']:
                return 1
            pipeline_success = False
        else:
            logger.info(f"Training completed in {step_times['train']/60:.1f} minutes")
            
            # Run analysis immediately after training if requested
            if args.mode in ['train_analyze'] and not args.skip_analysis:
                step_start = time.time()
                cmd = f"cd {project_dir} && python analyze_training_results.py --checkpoint_dir checkpoints --output_dir analysis_results"
                success = run_command(cmd, "Training analysis", timeout=600)
                step_times['analyze'] = time.time() - step_start
                
                if success:
                    logger.info("Training analysis completed - check analysis_results directory for plots and metrics")
                else:
                    logger.warning("Training analysis failed but continuing...")
                    
    # Step 3: Generate synthetic data
    if args.mode in ['all', 'generate']:
        step_start = time.time()
        
        # Determine which checkpoint to use
        checkpoint_path = args.checkpoint
        if not checkpoint_path:
            # Try to find best model
            best_model_path = project_dir / "checkpoints/best_model.pt"
            best_breakdown_path = project_dir / "checkpoints/best_breakdown_model.pt"
            
            if best_breakdown_path.exists():
                checkpoint_path = "checkpoints/best_breakdown_model.pt"
                logger.info("Using best breakdown prediction model for generation")
            elif best_model_path.exists():
                checkpoint_path = "checkpoints/best_model.pt"
                logger.info("Using best validation loss model for generation")
            else:
                logger.error("No checkpoint found for generation. Please train a model first.")
                return 1
                
        cmd = f"cd {project_dir} && python inference/generate.py --checkpoint {checkpoint_path} --num_samples {args.num_synthetic} --output_dir synthetic_data --visualize"
        success = run_command(cmd, "Synthetic data generation", timeout=args.timeout)
        step_times['generate'] = time.time() - step_start
        
        if not success:
            logger.error("Synthetic data generation failed")
            if args.mode == 'all':
                return 1
            pipeline_success = False
        else:
            logger.info(f"Generated {args.num_synthetic} synthetic samples")
            
    # Step 4: Analysis (standalone)
    if args.mode == 'analyze':
        step_start = time.time()
        cmd = f"cd {project_dir} && python analyze_training_results.py --checkpoint_dir checkpoints --output_dir analysis_results"
        success = run_command(cmd, "Training analysis", timeout=600)
        step_times['analyze'] = time.time() - step_start
        
        if not success:
            logger.error("Training analysis failed")
            pipeline_success = False
        else:
            logger.info("Analysis results saved to analysis_results directory")
            
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*60)
    
    # Report timing
    if step_times:
        logger.info("\nStep execution times:")
        total_time = 0
        for step, duration in step_times.items():
            logger.info(f"  {step.capitalize()}: {duration/60:.1f} minutes")
            total_time += duration
        logger.info(f"  Total: {total_time/60:.1f} minutes")
        
    # Report outputs
    logger.info("\nGenerated outputs:")
    
    outputs = [
        ("Processed data", "data/processed/"),
        ("Model checkpoints", "checkpoints/"),
        ("Training metrics", "checkpoints/metrics/"),
        ("Loss curves", "checkpoints/metrics/loss_curves.png"),
        ("Validation metrics", "checkpoints/metrics/validation_metrics.png"),
        ("Prediction visualizations", "checkpoints/metrics/predictions/"),
        ("Synthetic data", "synthetic_data/"),
        ("Analysis results", "analysis_results/")
    ]
    
    for name, path in outputs:
        full_path = project_dir / path
        if full_path.exists():
            if full_path.is_file():
                logger.info(f"  ✓ {name}: {path}")
            else:
                # Count files in directory
                num_files = sum(1 for _ in full_path.iterdir() if _.is_file())
                logger.info(f"  ✓ {name}: {path} ({num_files} files)")
        else:
            logger.info(f"  ✗ {name}: Not found")
            
    # Final status
    if pipeline_success:
        logger.info("\n Pipeline completed successfully!")
        
        # Provide next steps
        logger.info("\nNext steps:")
        logger.info("1. Check training metrics: checkpoints/metrics/")
        logger.info("2. Review analysis results: analysis_results/")
        logger.info("3. Inspect synthetic data: synthetic_data/")
        
        if args.use_enhanced_train:
            logger.info("\nTo view detailed training analysis:")
            logger.info("  python analyze_training_results.py")
            
    else:
        logger.error("\n Pipeline completed with errors")
        
    return 0 if pipeline_success else 1


if __name__ == '__main__':
    sys.exit(main())
    