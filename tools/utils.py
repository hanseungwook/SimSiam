import torch
import IPython

def check_nan_inf(loss):
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        print("Loss is NaN or inf")
        IPython.embed()