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
    log_file=Path(config['paths']['log_dir'])/'training.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(str(log_file))

    #Check device
    if config['training']['device']=='cuda' and not torch.cuda.is_available():
        print("CUDA not available, suing CPU")
        config['training']['device']='cpu'
    
    print(f"Using device: {config['training']['device']}")

    #Train
    trainer=Trainer(config)
    trainer.train()

    print(f"TrainingComplete!")
    print(f"Best model saved to: {config['path']['model_dir']}/best_model.pt")

if __name__=='__main__':
    main()
