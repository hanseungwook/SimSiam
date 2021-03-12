#!/usr/bin/env python3

class LossScheduler(object):
    def __init__(self, init_lw, min_lw=0, decay_rate=0.75, decay_int=1, staircase=True):
        """
        Scheduler that performs exponential decay on a lambda (e.g. weight for loss)

        init_lw: initial loss weight
        decay_rate: rate at which to execute exponential decay
        decay_int: interval (in # of epochs) to perform decay
        """

        self.init_lw = init_lw
        self.min_lw = min_lw
        self.decay_rate = decay_rate
        self.decay_int = decay_int
        self.step = 0
    
    def step(self):
        self.step += 1

        return self.get_lw()
    
    def get_lw(self):
        # Only decay down to the min_lw (minimum loss weight)
        return max(self.init_lw * (self.decay_rate ** (self.step / self.decay_int)), self.min_lw)
    