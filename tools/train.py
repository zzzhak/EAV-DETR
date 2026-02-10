import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def main(args, ) -> None:
    '''main training function
    '''
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )

    # Get the task type from config
    task_type = cfg.yaml_cfg['task']
    
    # Print training information
    if task_type == 'codrone_detection':
        print("=" * 80)
        print("CODrone Oriented Object Detection Training")
        print("Using patch-level evaluation (no result merging)")
        print("=" * 80)
    else:
        print("=" * 80) 
        print("Standard Object Detection Training")
        print("=" * 80)
    
    print(f"Task: {task_type}")
    print(f"Config: {args.config}")
    print(f"Resume: {args.resume}")
    print(f"Tuning: {args.tuning}")
    print(f"Test only: {args.test_only}")
    print(f"AMP: {args.amp}")
    print(f"Seed: {args.seed}")
    print("=" * 80)

    solver = TASKS[task_type](cfg)
    
    if args.test_only:
        print("Running validation only...")
        solver.val()
    else:
        print("Starting training...")
        solver.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='RT-DETR Training Script')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--resume', '-r', type=str, 
                        help='Resume training from checkpoint')
    parser.add_argument('--tuning', '-t', type=str,
                        help='Fine-tuning from pretrained model')
    parser.add_argument('--test-only', action='store_true', default=False,
                        help='Only run validation without training')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()

    main(args)
