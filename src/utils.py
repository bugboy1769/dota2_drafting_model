import torch
import logging
from pathlib import Path

def setup_logging(log_file: str=None):
    handlers=[logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger()

def save_checkpoint(model, optimizer, epoch, metrics, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch':epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics':metrics
    }, path)

def load_checkpoint(model, path, optimizer=None):
    checkpoint=torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['optimizer_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('metrics', {})
