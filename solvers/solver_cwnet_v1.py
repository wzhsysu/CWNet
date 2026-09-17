import torch,os,pickle
import os.path as osp
import numpy as np
from solvers.base_solver import BaseSolver
from torchvision.utils import save_image
from utils.common import smooth_awb, visualize
from utils.error_calc import *
from PIL import Image



class Solver(BaseSolver):
    def valid(self, phase='valid', epoch=None):
        self.net.eval()
        with torch.no_grad():
            total_step = len(self.dataloader[phase])

            for step, item_dict in enumerate(self.dataloader[phase]):
                place = item_dict['place']
                illumcount = item_dict['illum_count']

                print(f'{place[0]}_{illumcount[0]}')

                input_rgb = item_dict['input_rgb'].to(self.device,non_blocking=True)
                gt_rgb = item_dict['gt_rgb'].to(self.device,non_blocking=True)
                gt_illum = item_dict['gt_illum'].to(self.device,non_blocking=True)
                gt_chroma = item_dict['gt_chroma'].to(self.device,non_blocking=True)
                gt_mixmap = item_dict['gt_mixmap'].to(self.device,non_blocking=True)
                mask = item_dict['mask'].to(self.device,non_blocking=True)

                # Inference
                if 'gsn' not in self.cfg.data.root:
                    ret_dict = self.net(x=input_rgb, get_every_ittr=True)
                else:
                    device_id = item_dict['device_id'].to(self.device,non_blocking=True)
                    device = item_dict['device']
                    ret_dict = self.net(x=input_rgb, device_id=device_id, device=device, get_every_ittr=True)

                chroma = ret_dict["chroma"]
                mixmap = ret_dict["mixmap"]
                mixed_illum = ret_dict["mixed_illum"]
                B = chroma.size(0)

                # Loss
                loss_dict = self.criterion(ret_dict, item_dict, ittr_dim_exist=True)
                total_loss = loss_dict['total_loss']
                # the iteration dim holds every intermediate result; score the last one
                mae_illum = get_MAE(pred=mixed_illum[:,-1,:,:,:], gt=gt_illum,tensor_type='illum', mask=mask)

                # [optional] visualize slot decomposition result
                if phase == 'test' and self.cfg.test.visualize_result:

                    # visualize result images
                    for b in range(B):
                        fname_base = f'{place[b]}_{illumcount[b]}'

                        img_rgb = input_rgb[b].detach()

                        ones = torch.ones_like(img_rgb[:1,:,:])
                        mixed_illum_map_rb = mixed_illum[b,-1].detach()
                        mixed_illum_map_rgb = torch.cat((mixed_illum_map_rb[:1,:,:],ones,mixed_illum_map_rb[1:,:,:]),dim=0)

                        # white-balanced image (linear); re-rendered to sRGB below
                        img_wb = img_rgb / mixed_illum_map_rgb
                        img_wb = torch.clamp(img_wb, 0, 1)

                        # visualize sRGB images
                        if self.cfg.camera != 'gsn':
                            if self.cfg.camera == 'galaxy':
                                white_level = 1023.
                            elif self.cfg.camera == 'nikon' or self.cfg.camera == 'sony' or self.cfg.camera == 'Canon1DsMkIII':
                                white_level = 16383.
                            else:
                                white_level = ret_dict['white_level']
                            # get np array of sRGB images
                            input_srgb,output_srgb,gt_srgb = visualize(img_rgb*white_level,img_wb*white_level,gt_rgb[b]*white_level,self.cfg.camera,concat=False)
                            # save sRGB images
                            filename = f'{fname_base}_input_srgb.png'
                            Image.fromarray(input_srgb).save(osp.join(self.cfg.path.result_path, filename))
                            filename = f'{fname_base}_output_srgb.png'
                            Image.fromarray(output_srgb).save(osp.join(self.cfg.path.result_path, filename))
                            filename = f'{fname_base}_gt_srgb.png'
                            Image.fromarray(gt_srgb).save(osp.join(self.cfg.path.result_path, filename))

                # [optional] if test mode, save last itteration's chroma, mixmap into npy file
                if self.cfg.test.save_npy and phase == 'test':
                    # get last itteration's chroma, mixmap
                    chroma_result = chroma[:,-1,:,:]   # [B, N_slot, 2]
                    mixmap_result = mixmap[:,-1,:,:]   # [B, N_slot, H, W]

                    top_dict = self.get_top_N_result(chroma_result,mixmap_result,N=int(len(illumcount[0])))
                    os.makedirs(os.path.join(self.cfg.path.result_path, 'top_N_chroma_mixmap'), exist_ok=True)
                    # save top dict to file
                    with open(os.path.join(self.cfg.path.result_path, 'top_N_chroma_mixmap',f'{place[0]}_{illumcount[0]}.pkl'), 'wb') as f:
                        pickle.dump(top_dict, f)

                    for b in range(B):
                        filename = f'{place[b]}_{illumcount[b]}'
                        np.save(os.path.join(self.cfg.path.result_path, filename+'_chroma.npy'), chroma_result[b].detach().cpu().numpy())
                        np.save(os.path.join(self.cfg.path.result_path, filename+'_mixmap.npy'), mixmap_result[b].detach().cpu().numpy())

                #  [optional] apply smooth AWB using interploate_awb function and save result
                if self.cfg.test.visualize_smooth and phase == 'test':
                    # smoothed_illum_map, smoothed_awb_img : [B, top_N, step, 3, H, W]
                    smoothed_illum_map, smoothed_awb_img = smooth_awb(input_img=input_rgb, chromas=chroma[:,-1], mixmaps=mixmap[:,-1], top_N=3, step=5, colorspace=self.cfg.test.smooth_colorspace, cfg=self.cfg)

                    # apply 2.2 gamma encoding to smoothed_awb_img
                    smoothed_awb_img = torch.pow(smoothed_awb_img, 1/self.cfg.test.smooth_gamma)

                    # save smoothed awb_img & input_img
                    for b in range(B):
                        for i in range(smoothed_awb_img.size(1)):
                            for j in range(smoothed_awb_img.size(2)):
                                filename = f'{place[b]}_{illumcount[b]}_{i}_{j}_{self.cfg.test.smooth_colorspace}.png'
                                save_image(tensor=smoothed_awb_img[b,i,j,:,:,:], fp=os.path.join(self.cfg.path.result_path, filename))

                        filename = f'{place[b]}_{illumcount[b]}_input_gamma.png'
                        gamma_input_rgb = torch.pow(input_rgb[b,:,:,:], 1/self.cfg.test.smooth_gamma)
                        save_image(tensor=gamma_input_rgb, fp=os.path.join(self.cfg.path.result_path, filename))

                        filename = f'{place[b]}_{illumcount[b]}_input.png'
                        save_image(tensor=input_rgb[b,:,:,:], fp=os.path.join(self.cfg.path.result_path, filename))

                    # illuminant chromaticity manipulation
                    H, W = input_rgb.size(2), input_rgb.size(3)
                    
                    for b in range(B):
                        gt_illum_count = len(illumcount[b])
                        mixmap_mean = torch.mean(mixmap[b,-1,:,:,:], dim=(1,2))
                        _, top_N_idx = torch.topk(mixmap_mean, gt_illum_count, dim=0)

                        # selected chroma
                        chroma_top = chroma[b,-1,top_N_idx,:]   # [N, 2]
                        # reverse chroma_top on dim=0
                        reversed_chroma = chroma_top.flip(dims=(0,))

                        chroma_manip = torch.zeros((len(top_N_idx), 2), device=self.device) # [N, 2]
                        for i in range(len(top_N_idx)):
                            chroma_manip[i,:] = chroma[b,-1,top_N_idx[i],:].clone().detach() / (torch.rand(2, device=self.device) * 2.0)

                        # mix chroma with mixmap using einsum
                        illum_map_manip = torch.einsum('nc,nhw->chw', chroma_manip, mixmap[b,-1,top_N_idx,:,:])
                        illum_map_reverse = torch.einsum('nc,nhw->chw', reversed_chroma, mixmap[b,-1,top_N_idx,:,:])
                        
                        # insert ones for G channel, in the middle of the illum_map_manip
                        ones = torch.ones((1, H, W), device=self.device)
                        illum_map_manip = torch.cat((illum_map_manip[:1,:,:], ones, illum_map_manip[1:,:,:]), dim=0)
                        illum_map_reverse = torch.cat((illum_map_reverse[:1,:,:], ones, illum_map_reverse[1:,:,:]), dim=0)
                        mixed_illum = mixed_illum[b,-1]
                        mixed_illum = torch.cat((mixed_illum[:1,:,:], ones, mixed_illum[1:,:,:]), dim=0)

                        # apply AWB to input_rgb using manipulated illumination map
                        awb_img_manip = input_rgb[b,...] / illum_map_manip
                        awb_img_manip = torch.clamp(awb_img_manip, 0, 1)
                        awb_img_manip_gamma = torch.pow(awb_img_manip, 1/self.cfg.test.smooth_gamma)
                        
                        filename = f'{place[b]}_{illumcount[b]}_manip.png'
                        save_image(tensor=awb_img_manip_gamma, fp=os.path.join(self.cfg.path.result_path, filename))

                        # apply AWB to input_rgb using reversed illumination map
                        awb_img_reverse = input_rgb[b] / mixed_illum * illum_map_reverse
                        awb_img_reverse = torch.clamp(awb_img_reverse, 0, 1)
                        awb_img_reverse_gamma = torch.pow(awb_img_reverse, 1/self.cfg.test.smooth_gamma)

                        filename = f'{place[b]}_{illumcount[b]}_reverse.png'
                        save_image(tensor=awb_img_reverse_gamma, fp=os.path.join(self.cfg.path.result_path, filename))

                # Logging
                for key,value in loss_dict.items():
                    self.logger.log(f'{phase}/loss/{key}',value.item(), input_rgb.size(0))
                self.logger.log(f'{phase}/metric/mae_illum', mae_illum.item(), input_rgb.size(0))
                self.logger.print(type='last',prefix=f'[{phase}][{step+1}/{total_step}]',use_wandb_log=False)
            self.logger.print(type='avg',prefix=f'[{phase}][Mean]',use_wandb_log=True,step=self.epoch)
            self.logger.print(type='median',prefix=f'[{phase}][Median]',use_wandb_log=True,step=self.epoch)
            self.logger.print(type='min',prefix=f'[{phase}][Min]',use_wandb_log=True,step=self.epoch)
            self.logger.print(type='max',prefix=f'[{phase}][Max]',use_wandb_log=True,step=self.epoch)
            result_dict = self.logger.get_avg_dict()

            self.logger.clear()
        
        return result_dict
    
    def get_top_N_result(self,chroma_result,mixmap_result,N=2):
        chroma = chroma_result.clone().detach().cpu()[0]
        mixmap_result = mixmap_result.clone().detach().cpu()[0]

        mixmap_sum = torch.sum(mixmap_result,dim=(1,2))
        # get top N index from mixmap_sum
        _, top_N_idx = torch.topk(mixmap_sum, N, dim=0)

        # select chroma and mixmap
        chroma_top = chroma[top_N_idx,:].numpy()
        mixmap_top = mixmap_result[top_N_idx,:,:].numpy()

        # save top chroma & mixmap to dict
        top_result_dict = {}
        top_result_dict['chroma'] = chroma_top
        top_result_dict['mixmap'] = mixmap_top

        return top_result_dict