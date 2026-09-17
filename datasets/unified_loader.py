from datasets.LSMI import get_loader as get_LSMI_loader

def get_loader(cfg, split):
    return get_LSMI_loader(cfg, split)