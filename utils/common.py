import numpy as np
import math
import random,torch,time,os
import matplotlib.pyplot as plt
import cv2,rawpy

##########################
# Generic util functions #
##########################

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_path(cfg):
    # set datetime
    if cfg.load.ckpt_path == None:
        ckpt_filename = 'initial'
        cfg.path.date_time_model = time.strftime(cfg.path.time_format,time.localtime(time.time())) + '_' + \
                                    cfg.model.name + '_' + cfg.model.ver
    else:
        if cfg.mode == 'train':
            ckpt_filename = 'resume_' + os.path.basename(cfg.load.ckpt_path).split('.')[0]
        elif cfg.mode == 'test':
            ckpt_filename = 'test_' + os.path.basename(cfg.load.ckpt_path).split('.')[0]
        cfg.path.date_time_model = os.path.basename(os.path.dirname(cfg.load.ckpt_path))

    if 'finetune' in cfg.model.solver:
        ckpt_filename += f'_finetune_{cfg.camera}'
    
    # set path
    cfg.path.log_path = os.path.join(cfg.path.log_root, cfg.path.date_time_model)
    cfg.path.ckpt_path = os.path.join(cfg.path.ckpt_root, cfg.path.date_time_model,ckpt_filename)
    cfg.path.result_path = os.path.join(cfg.path.result_root,cfg.path.date_time_model,ckpt_filename)
    
    # make directories
    os.makedirs(cfg.path.log_path, exist_ok=True)
    os.makedirs(cfg.path.ckpt_path, exist_ok=True)
    os.makedirs(cfg.path.result_path, exist_ok=True)
    os.makedirs('../configs/archive/', exist_ok=True)

def rgb2uvl(img_rgb):
    epsilon = 1e-8
    img_uvl = np.zeros_like(img_rgb, dtype='float32')
    img_uvl[:,:,2] = np.log(img_rgb[:,:,1] + epsilon)
    img_uvl[:,:,0] = np.log(img_rgb[:,:,0] + epsilon) - img_uvl[:,:,2]
    img_uvl[:,:,1] = np.log(img_rgb[:,:,2] + epsilon) - img_uvl[:,:,2]

    return img_uvl

def mix_chroma(mixmap,chroma_list,illum_count):
    ret = np.stack((np.zeros_like(mixmap[:,:,0],dtype=float),)*3, axis=2)
    for i in range(len(illum_count)):
        illum_idx = int(illum_count[i])-1
        mixmap_3ch = np.stack((mixmap[:,:,i],)*3, axis=2)
        ret += (mixmap_3ch * [[chroma_list[illum_idx]]])
    
    return ret

def visualize_mixture(pred_chromas,pred_mixmap,pred_illum,gt_chromas,gt_mixmap,gt_illum):
    """
    Args:
    pred_chromas    : (B,N_slot,2) - 2 for RB
    pred_mixmap     : (B,N_slot,H,W)
    pred_illum      : (B,2,H,W)
    gt_chromas      : (B,3,3)
    gt_mixmap       : (B,3,H,W)
    gt_illum        : (B,2,H,W)
    """
    b,n_slots,_ = pred_chromas.shape
    _,_,h,w = pred_mixmap.shape
    n_pad = n_slots - 4
    if abs(n_pad) != 0:
        black_patches = torch.zeros(size=(b,3,h,w*abs(n_pad)),device=pred_chromas.device)           # (B,3,H,W*N_slot-4)
    # insert G=1 to RB pred_chromas
    if pred_chromas.size(2) == 2:
        ones = torch.ones_like(pred_chromas[:,:,:1])
        pred_chromas = torch.cat([pred_chromas[:,:,:1],ones,pred_chromas[:,:,1:]],dim=2)          # (B,N_slot,3)

    # insert G=1 to RB illuminations
    ones = torch.ones_like(pred_illum[:,:1,:,:])
    pred_illum = torch.cat([pred_illum[:,:1,:,:],ones,pred_illum[:,1:,:,:]],dim=1)                  # (B,3,H,W)
    gt_illum = torch.cat([gt_illum[:,:1,:,:],ones,gt_illum[:,1:,:,:]],dim=1)                        # (B,3,H,W)

    # Optional: scaling G channel for better visibility
    scale = 0.6
    pred_chromas[:,:,1] *= scale
    gt_chromas[:,:,1] *= scale
    pred_illum[:,1,:,:] *= scale
    gt_illum[:,1,:,:] *= scale

    # generate pred_chroma patches
    pred_chroma_patches = pred_chromas[:,:,:,None,None]                                             # (B,N_slot,3,1,1)
    pred_chroma_patches = pred_chroma_patches.expand(b,n_slots,3,h,w)                               # (B,N_slot,3,H,W)
    pred_chroma_patches = pred_chroma_patches.permute(0,1,4,2,3).reshape(b,-1,3,h).permute(0,2,3,1) # (B,3,H,W*N_slot)
    if n_pad < 0:
        pred_chroma_patches = torch.cat([pred_chroma_patches,black_patches], dim=3)
    # pred_chroma_patches = pred_chroma_patches * 100 / 255

    # generate mixmap patches
    pred_mixmap_patches = pred_mixmap.view(b,n_slots,1,h,w)                                         # (B,N_slot,1,H,W)
    pred_mixmap_patches = pred_mixmap_patches.expand(b,n_slots,3,h,w)                               # (B,N_slot,3,H,W)
    pred_mixmap_patches = pred_mixmap_patches.permute(0,1,4,2,3).reshape(b,-1,3,h).permute(0,2,3,1) # (B,3,H,W*N_slot)
    if n_pad < 0:
        pred_mixmap_patches = torch.cat([pred_mixmap_patches,black_patches], dim=3)
    
    # generate gt_chroma patches
    gt_chroma_patches = gt_chromas[:,:,:,None,None]                                                 # (B,3,3,1,1)
    gt_chroma_patches = gt_chroma_patches.expand(b,3,3,h,w)                                         # (B,3,3,H,W)
    gt_chroma_patches = gt_chroma_patches.permute(0,1,4,2,3).reshape(b,-1,3,h).permute(0,2,3,1)     # (B,3,H,W*3)
    gt_chroma_patches = torch.cat([gt_chroma_patches,pred_illum],dim=3)                             # (B,3,H,W*4)
    if n_pad > 0:
        gt_chroma_patches = torch.cat([gt_chroma_patches,black_patches],dim=3)                      # (B,3,H,W*N_slot)
    # gt_chroma_patches = gt_chroma_patches * 100 / 255

    # generate gt_mixmap patches
    gt_mixmap_patches = gt_mixmap.view(b,3,1,h,w)                                                   # (B,3,1,H,W)
    gt_mixmap_patches = gt_mixmap_patches.expand(b,3,3,h,w)                                         # (B,3,3,H,W)
    gt_mixmap_patches = gt_mixmap_patches.permute(0,1,4,2,3).reshape(b,-1,3,h).permute(0,2,3,1)     # (B,3,H,W*3)
    gt_mixmap_patches = torch.cat([gt_mixmap_patches,gt_illum],dim=3)                               # (B,3,H,W*4)
    if n_pad > 0:
        gt_mixmap_patches = torch.cat([gt_mixmap_patches,black_patches],dim=3)                      # (B,3,H,W*N_slot)
    
    final_visualization = torch.cat([
        pred_chroma_patches,pred_mixmap_patches,
        gt_chroma_patches,gt_mixmap_patches], dim=2)

    return final_visualization

def visualize_illum(pred_illum, gt_illum):
    """
    pred_illum : [B, 2, W, H]
    gt_illum   : [B, 2, W, H]
    """

    # insert G=1 to RB illuminations
    ones = torch.ones_like(pred_illum[:,:1,:,:])
    pred_illum = torch.cat([pred_illum[:,:1,:,:],ones,pred_illum[:,1:,:,:]],dim=1)                  # (B,3,H,W)
    gt_illum = torch.cat([gt_illum[:,:1,:,:],ones,gt_illum[:,1:,:,:]],dim=1)                        # (B,3,H,W)

    scale = 0.6
    pred_illum[:,1,:,:] *= scale
    gt_illum[:,1,:,:] *= scale

    final_visualization = torch.cat([pred_illum, gt_illum], dim=3)

    return final_visualization
    

def draw_AE_map(ae_map):
    fig = plt.figure()

    plt.pcolor(ae_map, vmin=0, vmax=20)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title(f"MAE: {ae_map.mean():.5f}")
    plt.close()

    fig.canvas.draw()

    return np.array(fig.canvas.renderer._renderer)

def merge_similar_solots(chroma, mixmap, verbose=False):
    """
    chroma : [B, N_slot, 2]
    mixmap : [B, N_slot, W, H]
    """
    GALAXY_DAYLIGHT_WHITEBALANCE = torch.tensor([1.9452837705612183, 1.0002996921539307, 1.5499752759933472])[None,:].to(chroma.device)
    CAM2SRGB = torch.tensor([[1.26749312212703,     0.538554442735016,  -0.806047564862046],
                             [-0.310501790822702,   1.72121703773475,   -0.410715246912047],
                             [-0.0835698217586037,  -0.611945832583168, 1.69551565434177]]).to(chroma.device)

    B = chroma.shape[0]
    slot_len = chroma.shape[1]

    pivot_mixmap_threshold = 0.01
    merge_mae_threshold = 12.5

    mixmap_mean = mixmap.mean(dim=(2,3))
    descending_indices = torch.argsort(mixmap_mean, dim=1, descending=True)

    if verbose:
        print("descending_indices", descending_indices)

    for b in range(B):
        debug_msg = []
        debug_string = ""
        merge_flag = np.array([False,] * slot_len)
        for s in range(0,slot_len-1):
            pivot_slot_idx = descending_indices[b,s]
            residual_slot_idx = descending_indices[b,s+1:]

            # if the pivot slot is too dark, skip
            if mixmap_mean[b, pivot_slot_idx] < pivot_mixmap_threshold:
                break
            # if the slot is already merged, skip
            if merge_flag[pivot_slot_idx] == True:
                continue
            
            pivot_chroma = chroma[b, pivot_slot_idx]            # (2)
            residual_chromas = chroma[b, residual_slot_idx]     # (N_slot-1-s, 2)
            pivot_mixmap_mean = mixmap_mean[b, pivot_slot_idx]  # (1)
            residual_mixmap_mean = mixmap_mean[b, residual_slot_idx] # (N_slot-1-s)
            
            # insert G=1 to RB illuminations to pivot_chroma & residual_chromas
            ones = torch.ones_like(pivot_chroma[:1])
            pivot_chroma_rgb = torch.cat([pivot_chroma[:1],ones,pivot_chroma[1:]],dim=0)[None,:]            # (1,3)
            ones = torch.ones_like(residual_chromas[:,:1])
            residual_chromas_rgb = torch.cat([residual_chromas[:,:1],ones,residual_chromas[:,1:]],dim=1)    # (N_slot-1-s, 3)
            
            # apply galaxy_daylight_whitebalance to chromas
            pivot_chroma_wb = pivot_chroma_rgb * GALAXY_DAYLIGHT_WHITEBALANCE
            residual_chromas_wb = residual_chromas_rgb * GALAXY_DAYLIGHT_WHITEBALANCE
            # apply cam2srgb to chromas
            pivot_chroma_srgb = torch.matmul(pivot_chroma_wb, CAM2SRGB.T)
            residual_chromas_srgb = torch.matmul(residual_chromas_wb, CAM2SRGB.T)
            # convert sRGB to Lab, using openCV
            pivot_chroma_lab = torch.tensor(cv2.cvtColor(pivot_chroma_srgb.cpu().numpy()[None,:], cv2.COLOR_RGB2Lab)[0])
            residual_chromas_lab = torch.tensor(cv2.cvtColor(residual_chromas_srgb.cpu().numpy()[None,:], cv2.COLOR_RGB2Lab)[0])

            # compare angular error between pivot_chroma_wb and residual_chromas_wb
            ae = torch.acos(torch.sum(pivot_chroma_wb * residual_chromas_wb, dim=1) / (torch.norm(pivot_chroma_wb, dim=1) * torch.norm(residual_chromas_wb, dim=1))) * 180 / math.pi
            
            # compare angular error between ab components of pivot_chroma_lab & residual_chromas_lab
            # ae = torch.acos(torch.sum(pivot_chroma_lab[:,1:3] * residual_chromas_lab[:,1:3],dim=1) / (torch.norm(pivot_chroma_lab[:,1:3], dim=1) * torch.norm(residual_chromas_lab[:,1:3], dim=1))) * 180 / math.pi 
            # compare angular error between rgb components
            # ae = torch.acos(torch.sum(pivot_chroma_rgb * residual_chromas_rgb, dim=1) / (torch.norm(pivot_chroma_rgb) * torch.norm(residual_chromas_rgb, dim=1))) * 180 / math.pi
            
            # generate mask that has True when ae > merge_mae_threshold
            merge_mask = (ae < merge_mae_threshold)
            merge_slot_idx = residual_slot_idx[merge_mask]
            
            
            # if merge_slot_idx is not empty, merge mixmap and chroma
            if len(merge_slot_idx) > 0:
                # merge mixmap of selected slots into pivot slot mixmap & fill zero to merged slots
                mixmap[b,pivot_slot_idx] += torch.sum(mixmap[b,merge_slot_idx], dim=0)
                mixmap[b,merge_slot_idx] = 0
                # merge chroma of selected slots into pivot chroma, using mixmap_mean as weight
                chroma[b,pivot_slot_idx] = (pivot_chroma * pivot_mixmap_mean + torch.sum(chroma[b,merge_slot_idx] * mixmap_mean[b,merge_slot_idx][:,None], dim=0)) / (pivot_mixmap_mean + torch.sum(mixmap_mean[b,merge_slot_idx], dim=0))

                # toggle merge_flag to True for merged slots
                merge_flag[merge_slot_idx.cpu().numpy()] = True

            if verbose:
                print("======================itter {}========================".format(s))
                print(f"pivot_slot_idx : {pivot_slot_idx}")
                print(f"residual_slot_idx : {residual_slot_idx}")
                print(f"pivot_mixmap_mean : {mixmap_mean[b,pivot_slot_idx]}")
                print(f"ae : {ae}")
                print(f"merge_slot_idx : {merge_slot_idx}")
                print(f"merge_flag : {merge_flag}")
                print("=====================================================")

            debug_string += f"======================itter {s}========================\n"
            debug_string += f"pivot_slot_idx : {pivot_slot_idx}\n"
            debug_string += f"residual_slot_idx : {residual_slot_idx}\n"
            debug_string += f"pivot_mixmap_mean : {mixmap_mean[b,pivot_slot_idx]}\n\n\n"
            debug_string += f"pivot_chroma_rgb : {pivot_chroma_rgb}\n"
            debug_string += f"pivot_chroma_wb : {pivot_chroma_wb}\n"
            debug_string += f"pivot_chroma_srgb : {pivot_chroma_srgb}\n"
            debug_string += f"pivot_chroma_lab : {pivot_chroma_lab}\n"
            debug_string += f"residual_chromas_rgb : {residual_chromas_rgb}\n"
            debug_string += f"residual_chromas_wb : {residual_chromas_wb}\n"
            debug_string += f"residual_chromas_srgb : {residual_chromas_srgb}\n"
            debug_string += f"residual_chromas_lab : {residual_chromas_lab}\n\n\n"
            debug_string += f"ae : {ae}\n"
            debug_string += f"merge_slot_idx : {merge_slot_idx}\n"
            debug_string += f"merge_flag : {merge_flag}\n"
            debug_string += "=========================================================\n"

        debug_msg.append(debug_string)

    mixed_illum = torch.einsum('bki,bkwh->biwh', chroma, mixmap)

    return chroma, mixmap, mixed_illum, debug_msg

def smooth_awb(input_img,chromas,mixmaps,top_N=3,step=5,colorspace='RAW',cfg=None):
    """
    This function applies AWB w.r.t top_N chromas, 

    Args:
        input_img : torch.tensor, shape = (B,C,H,W)
        chromas : torch.tensor, shape = (B,N_slot,2)
        mixmaps : torch.tensor, shape = (B,N_slot,H,W)
        top_N : int, number of top chromas to smooth
        step : int, number of interpolation steps

    Returns:
        smoothed_illum_map: torch.tensor, shape = (B,top_N,step,3,H,W)
        smoothed_awb_img : torch.tensor, shape = (B,top_N,step,C,H,W)
    """
    input_img=input_img.to(cfg.device)
    chromas=chromas.to(cfg.device)
    mixmaps=mixmaps.to(cfg.device)

    B,C,H,W = input_img.shape
    N_slot = chromas.shape[1]

    # generate interpolation chromas (range : [R,B] ~ [1,1])
    interp_chromas = torch.zeros((B,N_slot,step,2)).to(cfg.device)     # [B,N_slot,step,2]
    for b in range(B):
        for s in range(N_slot):
            interp_chromas[b,s,:,0] = torch.linspace(chromas[b,s,0],1,step)
            interp_chromas[b,s,:,1] = torch.linspace(chromas[b,s,1],1,step)

    # get top_N indices of slots, using mixmap_mean
    mixmap_mean = torch.mean(mixmaps, dim=(2,3))            # [B,N_slot]
    _, top_N_idx = torch.topk(mixmap_mean, top_N, dim=1)    # [B,top_N]
    
    # generate smoothed_illum_map
    smoothed_illum_map = torch.zeros((B,top_N,step,2,H,W)).to(cfg.device)    # [B,top_N,step,H,W]
    for b in range(B):
        for n in range(top_N):
            for s in range(step):
                smoothed_chromas = chromas[b].clone().to(cfg.device)  # [N_slot,2]
                
                slot_idx = top_N_idx[b,n]
                # get smoothed chroma w.r.t step s
                step_chroma = interp_chromas[b,slot_idx,s]          # [2]
                
                # insert step_crhomas into  smoothed_chromas
                smoothed_chromas[slot_idx] = step_chroma

                # mix chomas & mixmaps using einsum
                smoothed_illum_map[b,n,s] = torch.einsum('ki,kwh->iwh', smoothed_chromas, mixmaps[b])  # [2,H,W]

    # insert G = 1 in the middle of smoothed_illum_map dim 3, [B,top_N,step,2,H,W] -> [B,top_N,step,3,H,W]
    smoothed_illum_map = torch.cat((smoothed_illum_map[:,:,:,0:1],torch.ones((B,top_N,step,1,H,W)).to(cfg.device),smoothed_illum_map[:,:,:,1:2]),dim=3)

    # generate smoothed_awb_img
    smoothed_awb_img = torch.zeros((B,top_N,step,C,H,W)).to(cfg.device)    # [B,top_N,step,C,H,W]
    for b in range(B):
        for n in range(top_N):
            for s in range(step):
                smoothed_awb_img[b,n,s] = input_img[b] / smoothed_illum_map[b,n,s].unsqueeze(0)
    smoothed_awb_img = torch.clip(smoothed_awb_img,0,1)

    # apply color transform matrix to smoothed_awb_img
    if colorspace == 'sRGB':
        cam2srgb = np.array([[1.26749312212703, 0.538554442735016, -0.806047564862046],
                            [-0.310501790822702, 1.72121703773475, -0.410715246912047],
                            [-0.0835698217586037, -0.611945832583168, 1.69551565434177]])
        cam2srgb = torch.tensor(cam2srgb).to(cfg.device).float()
        smoothed_awb_img = torch.matmul(cam2srgb, smoothed_awb_img.permute(0,1,2,4,5,3).unsqueeze(-1)).squeeze(-1).permute(0,1,2,5,3,4)


    return smoothed_illum_map, smoothed_awb_img

def apply_wb(org_img,pred,pred_type):
    """
    By using pred tensor (illumination map or uv),
    apply wb into original image (3-channel RGB image).
    """
    pred_rgb = torch.zeros_like(org_img) # b,c,h,w

    if pred_type == "illumination":
        pred_rgb[:,1,:,:] = org_img[:,1,:,:]
        pred_rgb[:,0,:,:] = org_img[:,0,:,:] / (pred[:,0,:,:]+1e-8)    # R_wb = R / illum_R
        pred_rgb[:,2,:,:] = org_img[:,2,:,:] / (pred[:,2,:,:]+1e-8)    # B_wb = B / illum_B
    elif pred_type == "uv":
        pred_rgb[:,1,:,:] = org_img[:,1,:,:]
        pred_rgb[:,0,:,:] = org_img[:,1,:,:] * torch.exp(pred[:,0,:,:])   # R = G * (R/G)
        pred_rgb[:,2,:,:] = org_img[:,1,:,:] * torch.exp(pred[:,1,:,:])   # B = G * (B/G)
    
    return pred_rgb

def plot_illum(pred_map=None,gt_map=None,MAE_illum=None,MAE_rgb=None,PSNR=None):
    """
    plot illumination map into R,B 2-D space
    """
    # plot pred first, then gt
    fig = plt.figure()
    if pred_map is not None:
        plt.plot(pred_map[:,0],pred_map[:,1],'bo',alpha=0.03,markersize=5)
    if gt_map is not None:
        plt.plot(gt_map[:,0],gt_map[:,1],'ro',alpha=0.01,markersize=3)
    minx,miny = min(gt_map[:,0]),min(gt_map[:,1])
    maxx,maxy = max(gt_map[:,0]),max(gt_map[:,1])
    lenx = (maxx-minx)/2
    leny = (maxy-miny)/2
    add_len = max(lenx,leny) + 0.3

    center_x = (maxx+minx)/2
    center_y = (maxy+miny)/2

    plt.xlim(center_x-add_len,center_x+add_len)
    plt.ylim(center_y-add_len,center_y+add_len)

    # make square
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.title(f'MAE_illum:{MAE_illum:.4f} / PSNR:{PSNR}')
    plt.close()

    fig.canvas.draw()
    plot_illum = np.array(fig.canvas.renderer._renderer)

    # plot gt first, then pred
    fig = plt.figure()
    if gt_map is not None:
        plt.plot(gt_map[:,0],gt_map[:,1],'ro',alpha=0.01,markersize=3)
    if pred_map is not None:
        plt.plot(pred_map[:,0],pred_map[:,1],'bo',alpha=0.03,markersize=5)
    minx,miny = min(gt_map[:,0]),min(gt_map[:,1])
    maxx,maxy = max(gt_map[:,0]),max(gt_map[:,1])
    lenx = (maxx-minx)/2
    leny = (maxy-miny)/2
    add_len = max(lenx,leny) + 0.3

    center_x = (maxx+minx)/2
    center_y = (maxy+miny)/2

    plt.xlim(center_x-add_len,center_x+add_len)
    plt.ylim(center_y-add_len,center_y+add_len)

    # make square
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.title(f'MAE_illum:{MAE_illum:.4f} / PSNR:{PSNR}')
    plt.close()

    fig.canvas.draw()
    plot_illum_rev = np.array(fig.canvas.renderer._renderer)

    return plot_illum, plot_illum_rev

def visualize(input_patch, pred_patch, gt_patch, templete, concat=True):
    """
    Visualize model inference result.
    1. Re-bayerize RGB image by duplicating G pixels.
    2. Copy bayer pattern image into rawpy templete instance
    3. Use user_wb to render RGB image
    4. Crop proper size of patch from rendered RGB image
    """
    # move all tensors to cpu
    input_patch = input_patch.cpu()
    pred_patch = pred_patch.cpu()
    gt_patch = gt_patch.cpu()

    input_patch = input_patch.permute((1,2,0))
    pred_patch = pred_patch.permute((1,2,0))
    gt_patch = gt_patch.permute((1,2,0))

    height, width, _ = input_patch.shape

    dng_path = "../datasets/" + templete + ".dng"
    if not os.path.exists(dng_path):
        raise FileNotFoundError(
            f"RAW template '{dng_path}' not found.\n"
            f"Visualization re-renders the result through a camera RAW template, which is\n"
            f"NOT distributed with this release. Either place your own '{templete}.dng'\n"
            f"under datasets/, or turn it off with test.visualize_result=false."
        )
    raw = rawpy.imread(dng_path)

    white_level = raw.white_level

    if templete == 'sony':
        black_level = 512
        white_level = raw.white_level / 4
    else:
        black_level = min(raw.black_level_per_channel)
        white_level = raw.white_level
        
    input_rgb = input_patch.numpy().astype('uint16')
    output_rgb = np.clip(pred_patch.cpu().numpy(), 0, white_level).astype('uint16')
    gt_rgb = gt_patch.numpy().astype('uint16')

    input_bayer = bayerize(input_rgb, templete, black_level)
    output_bayer = bayerize(output_rgb, templete, black_level)
    gt_bayer = bayerize(gt_rgb, templete, black_level)

    input_rendered = render(raw, white_level, input_bayer, height, width, "daylight_wb")
    output_rendered = render(raw, white_level, output_bayer, height, width, "maintain")
    gt_rendered = render(raw, white_level, gt_bayer, height, width, "maintain")

    if concat:
        return np.hstack([input_rendered, output_rendered, gt_rendered])
    else:
        return input_rendered, output_rendered, gt_rendered

def bayerize(img_rgb, camera, black_level):
    h,w,c = img_rgb.shape

    bayer_pattern = np.zeros((h*2,w*2))
    
    if camera == "galaxy":
        bayer_pattern[0::2,1::2] = img_rgb[:,:,0] # R
        bayer_pattern[0::2,0::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,1::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,0::2] = img_rgb[:,:,2] # B
    elif camera == "sony" or camera == 'nikon':
        bayer_pattern[0::2,0::2] = img_rgb[:,:,0] # R
        bayer_pattern[0::2,1::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,0::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,1::2] = img_rgb[:,:,2] # B

    return bayer_pattern + black_level

def render(raw, white_level, bayer, height, width, wb_method):
    raw_mat = raw.raw_image
    for h in range(height*2):
        for w in range(width*2):
            raw_mat[h,w] = bayer[h,w]

    if wb_method == "maintain":
        user_wb = [1.,1.,1.,1.]
    elif wb_method == "daylight_wb":
        user_wb = raw.daylight_whitebalance

    rgb = raw.postprocess(user_sat=white_level, user_wb=user_wb, half_size=True, no_auto_bright=True)
    rgb_croped = rgb[0:height,0:width,:]
    
    return rgb_croped