import torch
import logging
from pathlib import Path

# Standard Captain's Mode Draft Order (Team 0=Radiant, Team 1=Dire)
# (is_pick, team)
DRAFT_ORDER = [
    (0, 0), (0, 1), (0, 1), (0, 0), (0, 1), (0, 1), (0, 0),# First Ban Phase (7 bans)
    (1, 0), (1, 1), # First Pick Phase (2 picks)
    (0, 0), (0, 0), (0, 1), # Second Ban Phase (3 bans)
    (1, 1), (1, 0), (1, 0), (1, 1), (1, 1), (1, 0), # Second Pick Phase (6 picks)
    (0, 0), (0, 1), (0, 1), (0, 0), # Third Ban Phase (4 bans)
    (1, 0), (1, 1)                  # Third Pick Phase (2 picks)
]

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
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('metrics', {})

def prepare_model_input(history, device):
    """
    Convert a list of hero IDs (history) into model-ready tensors.
    Returns: seq_tensor, type_tensor, team_tensor, valid_tensor
    """
    seq_len = 24
    
    # A. Hero Sequence (Pad with 0)
    sequence = [0] * seq_len
    for i, h_id in enumerate(history):
        if i < seq_len:
            sequence[i] = h_id
    
    # B. Type & Team Sequence (From DRAFT_ORDER)
    type_sequence = [0] * seq_len
    team_sequence = [0] * seq_len
    
    for i in range(min(len(history), seq_len)):
        # Use DRAFT_ORDER to fill past history context
        # Note: For future steps (zeros), we can leave them as 0 or fill them.
        # The model usually masks padding, but filling type/team for the whole sequence 
        # (even future steps) is often better for the Transformer to know "what slot is this".
        pass

    # Actually, better strategy: Fill the ENTIRE sequence with DRAFT_ORDER info
    # This tells the model "Slot 5 is a Radiant Pick", even if we haven't picked it yet.
    # This is crucial for Positional Encoding context.
    for i in range(seq_len):
        if i < len(DRAFT_ORDER):
            is_pick, team = DRAFT_ORDER[i]
            type_sequence[i] = is_pick
            team_sequence[i] = team

    # C. Valid Actions Mask
    valid_actions = [True] * 150
    for h_id in history:
        if 1 <= h_id <= 150:
            valid_actions[h_id-1] = False
    
    # D. Convert to Tensors
    seq_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
    type_tensor = torch.tensor([type_sequence], dtype=torch.long).to(device)
    team_tensor = torch.tensor([team_sequence], dtype=torch.long).to(device)
    valid_tensor = torch.tensor([valid_actions], dtype=torch.bool).to(device)
    
    return seq_tensor, type_tensor, team_tensor, valid_tensor
