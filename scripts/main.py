import sys,os,importlib
from omegaconf import OmegaConf

sys.path.insert(1,os.path.abspath('..'))
sys.path.insert(1,os.path.abspath('../../'))
from utils.common import set_random_seed,init_path
from datasets.unified_loader import get_loader
from helpers.logger import Logger
from criterions.criterion import MasterCriterion

def main(cfg):
    # Set random seed
    # if cfg.mode == 'test' and cfg.random_seed is None:
    #     cfg.random_seed = 0
    if cfg.random_seed is not None:
        set_random_seed(cfg.random_seed)

    # Dataloader
    dataloader = {
        'test' : get_loader(cfg,'test')
    }
    
    # Dynamic Model module import
    network_mod = importlib.import_module(f'models.{cfg.model.name}_{cfg.model.ver}')
    network_class = getattr(network_mod,cfg.model.name)
    network = network_class(cfg)
    
    # Loss
    criterion = MasterCriterion(cfg)

    # Logger, Saver
    logger = Logger(cfg)

    # Dynamic Solver module import
    # solver_mod = importlib.import_module(f'solvers.solver_{cfg.model.name}_{cfg.model.solver}')
    solver_mod = importlib.import_module(f'solvers.solver_cwnet_{cfg.model.solver}')
    # solver_mod = importlib.import_module(f'solvers.solver_fm_{cfg.model.solver}')
    solver_class = getattr(solver_mod,'Solver')
    solver = solver_class(cfg,dataloader,network,criterion,logger)
    
    # Load network parameters from checkpoint
    if cfg.load.ckpt_path is not None:
        solver.load_network()

    # Let's go!
    if cfg.mode != 'test':
        raise ValueError(f'This release only contains the evaluation pipeline; mode must be "test", got "{cfg.mode}".')
    solver.valid(phase='test')

if __name__ == '__main__':
    # import default config file
    cfg = OmegaConf.merge(OmegaConf.load(f'../configs/default.yaml'),OmegaConf.load('../configs/env.yaml'))
    # read from command line
    cfg_cmd = OmegaConf.from_cli()
    # merge model specific config file
    # print(cfg_cmd)
    if 'name' in cfg_cmd.model:
        cfg = OmegaConf.merge(cfg,OmegaConf.load(f'../configs/{cfg_cmd.model.name}.yaml'))
    else:
        cfg = OmegaConf.merge(cfg,OmegaConf.load(f'../configs/{cfg.model.name}.yaml'))
    # merge cfg from command line
    cfg = OmegaConf.merge(cfg,cfg_cmd)

    # Path configuration & generation
    init_path(cfg)

    print(cfg)

    # wandb stays off unless you opt in on the command line: logger.use_wandb=true
    cfg.logger.wandb.run_name = f'{cfg.servername}/{cfg.path.date_time_model}'

    main(cfg)