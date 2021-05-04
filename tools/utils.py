import torch
import IPython

from pdb import set_trace

def check_nan_inf(loss):
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        print("Loss is NaN or inf")
        set_trace()
