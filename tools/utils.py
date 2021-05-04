import IPython

def check_nan_inf(loss):
    if torch.isnan(loss) or torch.isinf(loss):
        print("Loss is NaN or inf")
        IPython.embed()