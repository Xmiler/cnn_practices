import numpy as np


log_interval = 50

class DefaultSchedulerFastAI():
    moms = (0.95, 0.85)
    div_factor = 25.
    pct_start = 0.3
    final_div = div_factor * 1e4

    def __init__(self, optimizer, cyc_len, lr_max, epoch_size, writer):
        self.optimizer = optimizer
        self.cyc_len = cyc_len
        self.lr_max = lr_max
        self.epoch_size = epoch_size
        self.writer = writer

    def __call__(self, engine):
        a1 = int(self.cyc_len * self.pct_start)
        a2 = self.cyc_len-a1

        if 0 < engine.state.iteration <= a1:
            pct = (engine.state.iteration - 1) / a1
            lr_st = self.lr_max / self.div_factor
            lr_en = self.lr_max
            mom_st = self.moms[0]
            mom_en = self.moms[1]
        elif a1 < engine.state.iteration <= self.cyc_len:
            pct = (engine.state.iteration -1 - a1) / a2
            lr_st = self.lr_max
            lr_en = self.lr_max/self.final_div
            mom_st = self.moms[1]
            mom_en = self.moms[0]
        else:
            assert False

        cos_out = 1 - np.cos(np.pi * pct)
        lr = lr_st + ((lr_en - lr_st) / 2) * cos_out
        mom = mom_st + ((mom_en - mom_st) / 2) * cos_out

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['betas'] = (mom, 0.99)

        iteration_on_epoch = (engine.state.iteration - 1) % self.epoch_size + 1
        if iteration_on_epoch % log_interval == 0:
            self.writer.add_scalar('lr', lr, global_step=engine.state.iteration)
            self.writer.add_scalar('mom0', mom, global_step=engine.state.iteration)
