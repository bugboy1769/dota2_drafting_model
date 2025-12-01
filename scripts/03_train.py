import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import torch
from src.train import Trainer
from src.utils import setup_logging

def main():
    #Load config
    with open('config.yaml', 'r') as f:
        config=yaml.safe_load(f)
    
    #Setup logging
    log_file=Path(config['path']['log_dir'])/'training.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(str(log_file))

    #Check device
    if config['training']['device']=='cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config['training']['device']='cpu'
    
    print(f"Using device: {config['training']['device']}")

    # Determine model name based on git branch
    try:
        import subprocess
        branch_name = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('utf-8').strip()
        if branch_name == 'masking-experiment':
            model_name = 'best_model_masking.pt'
        else:
            model_name = 'best_model.pt'
    except:
        model_name = 'best_model.pt'
    
    config['path']['model_name'] = model_name
    print(f"Current Branch: {branch_name if 'branch_name' in locals() else 'Unknown'}")
    print(f"Target Model File: {model_name}")

    #Train
    trainer=Trainer(config)
    trainer.train()

    print(f"TrainingComplete!")
    print(f"Best model saved to: {config['path']['model_dir']}/{model_name}")

if __name__=='__main__':
    main()
