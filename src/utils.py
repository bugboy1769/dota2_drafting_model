import torch
import logging
from pathlib import Path

# Standard Captain's Mode Draft Order (Team 0=Radiant, Team 1=Dire)
# (is_pick, team)
# Standard Captain's Mode Draft Structure (FP=First Pick Team, SP=Second Pick Team)
# (is_pick, is_first_pick_team)
# 0=Ban, 1=Pick
# 0=FP, 1=SP
DRAFT_STRUCTURE = [
    (0, 0), (0, 1), (0, 0), (0, 1), (0, 0), (0, 1), (0, 0), # First Ban Phase (7 bans: FP, SP, FP, SP, FP, SP, FP) - Wait, is it alternating?
    # Let's assume the structure in the original DRAFT_ORDER was correct for Radiant First Pick:
    # (0,0), (0,1), (0,1), (0,0), (0,1), (0,1), (0,0) -> FP, SP, SP, FP, SP, SP, FP
    # This matches the "3-2-2" or similar patterns.
    # Let's stick to the pattern observed in the original code which the user said was "current patch".
    (0, 0), (0, 1), (0, 1), (0, 0), (0, 1), (0, 1), (0, 0), # Bans 1
    (1, 0), (1, 1), # Picks 1
    (0, 0), (0, 0), (0, 1), # Bans 2
    (1, 1), (1, 0), (1, 0), (1, 1), (1, 1), (1, 0), # Picks 2
    (0, 0), (0, 1), (0, 1), (0, 0), # Bans 3
    (1, 0), (1, 1) # Picks 3
]

def get_draft_order(first_pick_team=0):
    """
    Generate the draft order based on who has first pick.
    first_pick_team: 0 for Radiant, 1 for Dire.
    Returns: List of (is_pick, team)
    """
    order = []
    for is_pick, is_fp_team in DRAFT_STRUCTURE:
        # If is_fp_team is 0 (FP), use first_pick_team.
        # If is_fp_team is 1 (SP), use 1 - first_pick_team.
        team = first_pick_team if is_fp_team == 0 else (1 - first_pick_team)
        order.append((is_pick, team))
    return order

# Expose a default for backward compatibility (Radiant First Pick)
DRAFT_ORDER = get_draft_order(0)

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

def prepare_model_input(history, device, first_pick_team=0):
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
    
    # B. Type & Team Sequence (From Dynamic DRAFT_ORDER)
    current_draft_order = get_draft_order(first_pick_team)
    
    type_sequence = [0] * seq_len
    team_sequence = [0] * seq_len
    
    for i in range(seq_len):
        if i < len(current_draft_order):
            is_pick, team = current_draft_order[i]
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
