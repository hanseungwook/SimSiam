class ExpScheduler(object):
    def __init__(self, init_val, min_val=0, decay_rate=0.75, decay_int=1, staircase=True):
        """
        Scheduler that performs exponential decay on a lambda (e.g. weight for loss)
        init_lw: initial loss weight
        decay_rate: rate at which to execute exponential decay
        decay_int: interval (in # of epochs) to perform decay
        """

        self.init_val = init_val
        self.min_val = min_val
        self.decay_rate = decay_rate
        self.decay_int = decay_int
        self.cur_step = 0
    
    def step(self):
        self.cur_step += 1

        return self.get_val()
    
    def get_val(self):
        # Only decay down to the min_val (minimum value)
        # Assumes that you step in the beginning of the loop to get the value (So starts at cur_step == 1)
        return max(self.init_val * (self.decay_rate ** ((self.cur_step-1) / self.decay_int)), self.min_val)