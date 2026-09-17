import wandb,torch
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from torchvision.utils import save_image
from PIL import Image

class Logger():
    def __init__(self,cfg):
        self.cfg = cfg
        self.accumulate_dict = {}   # Never reset unless you call clear() - always stack values
        self.buffer_dict = {}       # You can choose whether flush this dict or not when using functions
                                    # not use stacking - if key is already exist, overwrite values
        self.count_dict = {}
        self.img_dict = {}          # # Always reset every time you print()

        # init wandb
        if cfg.logger.use_wandb:
            wandb.init(dir=cfg.path.log_root,project=cfg.logger.wandb.project,
                    name=cfg.logger.wandb.run_name)
            wandb.config.update(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    def log(self,key,value,item_len):
        # insert items into accumulate_dict and count_dict
        if key in self.accumulate_dict:
            self.accumulate_dict[key] = np.append(self.accumulate_dict[key],value)
            self.count_dict[key] = np.append(self.count_dict[key],item_len)
        else:
            self.accumulate_dict[key] = np.array([])
            self.accumulate_dict[key] = np.append(self.accumulate_dict[key],value)
            self.count_dict[key] = np.array([])
            self.count_dict[key] = np.append(self.count_dict[key],item_len)

        self.buffer_dict[key] = value

    def log_img(self,key,path,img):
        self.img_dict[key] = [path,img]

    def get_avg_dict(self,include_group=False):
        ret_dict = {}

        for key, value in self.accumulate_dict.items():
            ret_key = key
            if include_group == False:
                ret_key = key.split('/')[-1]

            ret_dict[ret_key] = np.nansum(self.accumulate_dict[key]*self.count_dict[key]) / np.nansum(self.count_dict[key])

        return ret_dict

    def print(self,type='last',prefix='',use_wandb_log=True,use_wandb_img=False,step=None,wandb_commit=True):
        print_dict = {}
        wandb_dict = {}

        # log pre-processing
        if type == 'avg':
            # averaging items in accumulate_dict
            for key, value in self.accumulate_dict.items():
                print_dict[key] = np.nansum(self.accumulate_dict[key]*self.count_dict[key]) / np.nansum(self.count_dict[key])
                if use_wandb_log:
                    wandb_dict[key] = print_dict[key]
        elif type == 'median':
            # median items in accumulate_dict
            for key, value in self.accumulate_dict.items():
                print_dict[key] = np.nanmedian(self.accumulate_dict[key])
                if use_wandb_log:
                    wandb_dict[key] = print_dict[key]
        elif type == 'max':
            # max items in accumulate_dict
            for key, value in self.accumulate_dict.items():
                print_dict[key] = np.nanmax(self.accumulate_dict[key])
                if use_wandb_log:
                    wandb_dict[key] = print_dict[key]
        elif type == 'min':
            # min items in accumulate_dict
            for key, value in self.accumulate_dict.items():
                print_dict[key] = np.nanmin(self.accumulate_dict[key])
                if use_wandb_log:
                    wandb_dict[key] = print_dict[key]
        elif type == 'last':
            # get items from buffer_dict
            for key, value in self.buffer_dict.items():
                print_dict[key] = self.buffer_dict[key]
                if use_wandb_log:
                    wandb_dict[key] = print_dict[key]

        # print shell message
        print_str = prefix + ' '
        for key,value in print_dict.items():
            key_without_group = key.split('/')[-1]
            if key_without_group in ['epoch','step']:
                print_str += f'{key_without_group}:{int(value)} | '
            else:
                print_str += f'{key_without_group}:{value:.5f} | '
        print(print_str)

        # image log
        for key,item in self.img_dict.items():
            path,img = item
            # save image to local directory
            save_image(img,path)
            if use_wandb_img:
                # update wandb dictionary
                pil_img = TF.to_pil_image(img)
                wandb_dict[key] = [wandb.Image(pil_img,caption='result iamges')]

        # print wandb log
        if self.cfg.logger.use_wandb and (use_wandb_log or use_wandb_img):
            wandb.log(wandb_dict,step=step,commit=wandb_commit)

        # flush img_dict
        self.img_dict = {}
        self.buffer_dict = {}

    def clear(self):
        self.accumulate_dict = {}
        self.buffer_dict = {}
        self.count_dict = {}
        self.img_dict = {}