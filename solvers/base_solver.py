import torch
from torch.nn.parallel import DataParallel as DP

class BaseSolver():
    def __init__(self,cfg,dataloader,net,criterion,logger) -> None:
        # basic config
        self.cfg = cfg
        self.dataloader = dataloader
        self.net = net
        self.criterion = criterion
        self.logger = logger
        self.epoch = -1

        if cfg.device == 'cpu' or (cfg.device == 'cuda' and cfg.gpu == -1):
            self.device = cfg.device
        elif cfg.device == 'cuda' and cfg.gpu >= 0:
            self.device = f'cuda:{cfg.gpu}'
            self.cfg.device = f'cuda:{cfg.gpu}'

        # network
        self.net = net.to(self.device)

    def load_network(self):
        # Only load model parameters from checkpoint file
        # map_location keeps this working when the checkpoint was saved on a
        # different GPU index, or when running on CPU.
        ckpt = torch.load(self.cfg.load.ckpt_path, map_location=self.device)
        if 'net' in ckpt.keys():
            net = ckpt['net']
        else:   # legacy checkpoint
            net = ckpt

        if isinstance(self.net,DP):
            self.net.module.load_state_dict(net)
        else:
            self.net.load_state_dict(net)
        print(f"[Load]\tNetwork is loaded from {self.cfg.load.ckpt_path}.")
