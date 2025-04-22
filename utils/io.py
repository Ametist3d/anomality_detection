import yaml
import torch

def load_config(path='configs/padim.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

def save_model(state, out_path):
    """
    State can be a dict of means, pcas, etc.
    """
    torch.save(state, out_path)

def load_model(path, device='cpu'):
    return torch.load(path, map_location=device)
