import os,cv2,json,colorsys,sys
import time
import filelock
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
sys.path.insert(1,os.path.abspath('..'))
from utils.common import *
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import RandomResizedCrop



def safe_file_access(path, operation="read", max_retries=5, wait_base=0.1):
    """
    File access helper that tolerates filesystem races.
    operation: 'read', 'load' (numpy), or 'image'
    """
    for i in range(max_retries):
        try:
            # make sure the file exists
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

            # skip files that are still being written
            if os.path.getmtime(path) > time.time() - 2:  # modified within the last 2s
                print(f"WARNING: file may still be being written: {path}, waiting...")
                time.sleep(wait_base * (2 ** i))
                continue

            # take a file lock
            lock_path = path + ".lock"
            with filelock.FileLock(lock_path, timeout=0.5):
                if operation == "read":
                    with open(path, "rb") as f:
                        return f.read()
                elif operation == "load":
                    with open(path, "rb") as f:
                        return np.load(f)
                elif operation == "image":
                    # more robust image decoding
                    with open(path, "rb") as f:
                        img_data = np.frombuffer(f.read(), dtype=np.uint8)
                        return cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)
        except (OSError, FileNotFoundError, filelock.Timeout) as e:
            print(f"File access error ({i + 1}/{max_retries}): {e}")
            time.sleep(wait_base * (2 ** i))

    # all retries exhausted
    print(f"ERROR: could not access file: {path}")
    return None


class LSMI(data.Dataset):
    def __init__(self,cfg,root,split,img_pool,normalize=True,maxval=None,uncalculable=-1,
                 mask_uncalculable=None,mask_highlight=None,mask_black=None,
                 illum_aug=None, chroma_sampling_method=None, transform=None):
        self.cfg = cfg
        self.root = root                        # dataset root
        self.split = split                      # train / val / test
        self.img_pool = img_pool                # 1 / 2 / 3
        self.normalize = normalize              # normalize rgb image to 0~1
        self.maxval = maxval                    # max value w.r.t bit-depth of camera   [1023, 16383, -1]
                                                # -1 value means it differs from camera to camera (for gsn_merged dataset)
        self.uncalculable = uncalculable        # Masked value for uncalculable mixture
        self.mask_uncalculable = mask_uncalculable  # None or Masking value for uncalculable mixture
        self.mask_highlight = mask_highlight    # None or Saturation value
        self.mask_black = mask_black            # masking value for G=0 pixels
        self.random_color = illum_aug
        self.chroma_sampling_method = chroma_sampling_method
        self.transform = transform

        self.img_list = sorted([f for f in os.listdir(os.path.join(root,split))
                                 if f.endswith(".tiff")
                                 and len(os.path.splitext(f)[0].split("_")[-1]) in img_pool
                                 and 'gt' not in f])
        
        meta_file = os.path.join(self.root,'meta.json')
        with open(meta_file, 'r') as meta_json:
            self.meta_data = json.load(meta_json)

        self.chromaticity_pool = []
        for p in self.meta_data:
            for i in self.meta_data[p]:
                if "Light" in i and "NumOfLights" not in i:
                    if self.meta_data[p][i][0] > 10 or self.meta_data[p][i][2] > 4:
                        continue   # skip if chromaticity is too extreme
                    self.chromaticity_pool.append(self.meta_data[p][i])
        self.chromaticity_pool = np.array(self.chromaticity_pool)

        print("[Data]\t"+str(self.__len__())+" "+split+" images are loaded from "+root)

    def __getitem__(self, idx):
        """
        Returns
        metadata        : meta information
        input_***       : input image (uvl or rgb)
        gt_***          : GT (None or illumination or chromaticity)
        mask            : mask for undetermined illuminations (black pixels) or saturated pixels
        """

        # parse fname
        fname = os.path.splitext(self.img_list[idx])[0]
        img_file = fname+".tiff"
        mixmap_file = fname+".npy"
        place, illum_count = fname.split('_')
        maxval = self.maxval
        if maxval == -1:
            maxval = self.meta_data[place]["maxval"]
        #print(place,illum_count)

        # 1. prepare meta information
        ret_dict = {}
        ret_dict["gt_chroma"] = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]],dtype='float32')
        for illum_no in illum_count:
            gt_chroma = self.meta_data[place]["Light"+illum_no]
            ret_dict["gt_chroma"][int(illum_no)-1] = gt_chroma
        ret_dict["img_file"] = img_file
        ret_dict["place"] = place
        ret_dict["illum_count"] = illum_count

        # 2. prepare input & output GT
        # load mixture map & 3 channel RGB tiff image
        input_path = os.path.join(self.root,self.split,img_file)
        input_bgr = cv2.imread(input_path, cv2.IMREAD_UNCHANGED).astype('float32')
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
        if len(illum_count) != 1:
            # guard against file read errors
            # mixmap = np.load(os.path.join(self.root,self.split,mixmap_file)).astype('float32')
            mixmap_path = os.path.join(self.root, self.split, mixmap_file)
            mixmap_data = safe_file_access(mixmap_path, operation="load")
            mixmap = mixmap_data.astype('float32') if mixmap_data is not None else np.zeros((256, 256),dtype=np.float32)
        else:
            mixmap = np.ones_like(input_rgb[:,:,0:1])
        # mixmap contains -1 for ZERO_MASK, which means uncalculable pixels with LSMI's G channel approximation.
        # So we must replace negative values to 0 if we use pixel level augmentation.
        uncalculable_masked_mixmap = np.where(mixmap==self.uncalculable,0,mixmap)
        
        # make mixmap always to be 3 channel by adding dummy mixmap which is filled with 0
        # get ommitted illuminant index among 1,2,3 in illum_count (e.g. 1,3 -> 2)
        omitted_illum_idx = [i for i in range(3) if str(i+1) not in illum_count]
        dummy = np.zeros_like(mixmap[:,:,0])
        # insert dummy mixmap to omitted illuminant index
        padded_mixmap = uncalculable_masked_mixmap
        for i in omitted_illum_idx:
            padded_mixmap = np.insert(padded_mixmap,i,dummy,axis=2)
        ret_dict["gt_mixmap"] = padded_mixmap

        # random data augmentation
        # if self.random_color and self.split in ['train', 'test']:
        #     if self.chroma_sampling_method == "pool":
        #         augment_chroma = self.random_color(illum_count, chroma_pool=self.chromaticity_pool)
        #     elif self.chroma_sampling_method == "random":
        #         augment_chroma = self.random_color(illum_count)
        #     ret_dict["gt_chroma"] = augment_chroma
        #     tint_map = mix_chroma(uncalculable_masked_mixmap,augment_chroma,illum_count)
        #     # apply augmentation to input image
        #     gt_bgr = cv2.imread(os.path.join(self.root,self.split,fname+"_gt.tiff"), cv2.IMREAD_UNCHANGED).astype('float32')
        #     gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        #     input_rgb = (gt_rgb * tint_map).astype('float32')
        #     # clip input_rgb to 0~maxval
        #     input_rgb = np.clip(input_rgb,0,maxval)

        # prepare input tensor
        ret_dict["input_rgb"] = input_rgb/maxval if self.normalize else input_rgb
        ret_dict["input_uvl"] = rgb2uvl(input_rgb)
        
        # prepare output tensor
        illum_map = mix_chroma(uncalculable_masked_mixmap,ret_dict["gt_chroma"],illum_count)
        ret_dict["gt_illum"] = np.delete(illum_map, 1, axis=2)

        # guard against file read errors
        # output_bgr = cv2.imread(os.path.join(self.root,self.split,fname+"_gt.tiff"), cv2.IMREAD_UNCHANGED).astype('float32')
        img_path = os.path.join(self.root, self.split, fname + "_gt.tiff")
        img_data = safe_file_access(img_path, operation="image")
        output_bgr = img_data.astype('float32') if img_data is not None else np.zeros((512, 512, 3), dtype=np.float32)

        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        ret_dict["gt_rgb"] = output_rgb/maxval if self.normalize else output_rgb
        output_uvl = rgb2uvl(output_rgb)
        ret_dict["gt_uv"] = np.delete(output_uvl, 2, axis=2)

        # prepare one-hot device identifier
        if 'gsn' in self.root:          # return device_id only if dataset is Galaxy,Sony,Nikon merged dataset
            device_id = np.zeros(3, dtype='float32')
            if self.meta_data[place]["device"] == "galaxy":
                device_id[0] = 1
            elif self.meta_data[place]["device"] == "nikon":
                device_id[1] = 1
            elif self.meta_data[place]["device"] == "sony":
                device_id[2] = 1
            ret_dict["device_id"] = device_id
            ret_dict["device"] = self.meta_data[place]["device"]
        
        # 3. prepare mask
        if self.split == 'train':
            mask = cv2.imread(os.path.join(self.root,self.split,place+'_mask.png'), cv2.IMREAD_GRAYSCALE)
            mask = mask[:,:,None].astype('float32')
        else:
            mask = np.ones_like(input_rgb[:,:,0:1], dtype='float32')
        if self.mask_uncalculable != None:
            mask[mixmap[:,:,0]==self.uncalculable] = self.mask_uncalculable
        if self.mask_highlight != None:
            raise NotImplementedError("Implement highlight masking!")
        if self.mask_black != None:
            mask[input_rgb[:,:,1:2]==0] = self.mask_black
        ret_dict["mask"] = mask

        # 4. apply transform
        if self.transform != None:
            ret_dict = self.transform(ret_dict)

        if 'FM' in self.cfg.model.name:
            white_level = 1023
            input_srgb, output_srgb, gt_srgb = visualize(input_rgb * white_level, gt_rgb * white_level,
                                                         gt_rgb * white_level, self.cfg.camera, concat=False)
            ret_dict['input_srgb'] = input_srgb
            ret_dict['gt_srgb'] = gt_srgb

        return ret_dict

    def __len__(self):
        return len(self.img_list)

class PairedRandomResizedCrop():
    def __init__(self,size=(256,256),scale=(0.3,1.0),ratio=(1.,1.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio
    def __call__(self,ret_dict):
        i,j,h,w = RandomResizedCrop.get_params(img=ret_dict['input_rgb'],scale=self.scale,ratio=self.ratio)
        ret_dict['input_rgb'] = TF.resized_crop(ret_dict['input_rgb'],i,j,h,w,self.size)
        ret_dict['input_uvl'] = TF.resized_crop(ret_dict['input_uvl'],i,j,h,w,self.size)
        ret_dict['gt_illum'] = TF.resized_crop(ret_dict['gt_illum'],i,j,h,w,self.size)
        ret_dict['gt_mixmap'] = TF.resized_crop(ret_dict['gt_mixmap'],i,j,h,w,self.size)
        ret_dict['gt_rgb'] = TF.resized_crop(ret_dict['gt_rgb'],i,j,h,w,self.size)
        ret_dict['gt_uv'] = TF.resized_crop(ret_dict['gt_uv'],i,j,h,w,self.size)
        ret_dict['mask'] = TF.resized_crop(ret_dict['mask'],i,j,h,w,self.size)
        
        return ret_dict

class Resize():
    def __init__(self,size=(256,256)):
        self.size = size
    def __call__(self,ret_dict):
        ret_dict['input_rgb'] = TF.resize(ret_dict['input_rgb'],self.size)
        ret_dict['input_uvl'] = TF.resize(ret_dict['input_uvl'],self.size)
        ret_dict['gt_illum'] = TF.resize(ret_dict['gt_illum'],self.size)
        ret_dict['gt_mixmap'] = TF.resize(ret_dict['gt_mixmap'],self.size)
        ret_dict['gt_rgb'] = TF.resize(ret_dict['gt_rgb'],self.size)
        ret_dict['gt_uv'] = TF.resize(ret_dict['gt_uv'],self.size)
        ret_dict['mask'] = TF.resize(ret_dict['mask'],self.size)

        return ret_dict

class ToTensor():
    def __call__(self, ret_dict):
        ret_dict['input_rgb'] = torch.from_numpy(ret_dict['input_rgb'].transpose((2,0,1)))
        ret_dict['input_uvl'] = torch.from_numpy(ret_dict['input_uvl'].transpose((2,0,1)))
        ret_dict['gt_illum'] = torch.from_numpy(ret_dict['gt_illum'].transpose((2,0,1)))
        ret_dict['gt_mixmap'] = torch.from_numpy(ret_dict['gt_mixmap'].transpose((2,0,1)))
        ret_dict['gt_rgb'] = torch.from_numpy(ret_dict['gt_rgb'].transpose((2,0,1)))
        ret_dict['gt_uv'] = torch.from_numpy(ret_dict['gt_uv'].transpose((2,0,1)))
        ret_dict['mask'] = torch.from_numpy(ret_dict['mask'].transpose((2,0,1)))
        
        return ret_dict

class RandomColor():
    def __init__(self,sat_min,sat_max,val_min,val_max,hue_threshold):
        self.sat_min = sat_min
        self.sat_max = sat_max
        self.val_min = val_min
        self.val_max = val_max
        self.hue_threshold = hue_threshold
        self.chroma_threshold = np.pi/60 # 3 degree

    def hsv2rgb(self,h,s,v):
        return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))
    
    def hue_threshold_test(self,hue_list,hue):
        if len(hue_list) == 0:
            return True
        for h in hue_list:
            if abs(h - hue) < self.hue_threshold:
                return False
        return True
    
    def pool_threshold_test(self,chroma_list,chroma_rgb):
        if len(chroma_list) == 0:
            return True
        for c in chroma_list:
            # if angular distance is less than self.chroma_threshold, return False
            if np.arccos(np.dot(c,chroma_rgb)/(np.linalg.norm(c)*np.linalg.norm(chroma_rgb))) < self.chroma_threshold:
                return False
        return True

    def __call__(self, illum_count, chroma_pool=None):
        hue_list = []
        chroma_list = []
        ret_chroma = [[0,0,0],[0,0,0],[0,0,0]]

        for i in illum_count:
            while(True):
                hue = np.random.uniform(0,1)
                saturation = np.random.uniform(self.sat_min,self.sat_max)
                value = np.random.uniform(self.val_min,self.val_max)
                
                if type(chroma_pool) == np.ndarray:
                    chroma_rgb = chroma_pool[np.random.randint(0,len(chroma_pool))]

                    if self.pool_threshold_test(chroma_list,chroma_rgb):
                        chroma_list.append(chroma_rgb)
                        ret_chroma[int(i)-1] = chroma_rgb
                        break
                else:
                    chroma_rgb = np.array(self.hsv2rgb(hue,saturation,value), dtype='float32')
                    chroma_rgb /= chroma_rgb[1]

                    if self.hue_threshold_test(hue_list,hue):
                        hue_list.append(hue)
                        ret_chroma[int(i)-1] = chroma_rgb
                        break
        return np.array(ret_chroma).astype('float32')


def get_loader(cfg, split):
    random_color = None
    if cfg.data.illum_aug and split in ['train','test'] :
        random_color = RandomColor(0.2,0.8,
                                   1.0,1.0,
                                   0.2)

    if cfg.data.random_crop and split=='train':
        # train mode & random crop
        tsfm = transforms.Compose([ToTensor(),
                                   PairedRandomResizedCrop(size=(cfg.data.img_size,cfg.data.img_size),scale=(0.3,1.0),ratio=(1.,1.))])
    elif cfg.data.img_size != None:
        # validation & test mode or train mode without random crop / square resizing
        tsfm = transforms.Compose([ToTensor(),
                                   Resize(size=(cfg.data.img_size,cfg.data.img_size))])
    else :
        # validation & test mode or train mode without random crop / original image size
        tsfm = transforms.Compose([ToTensor()])

    if 'gsn' in cfg.data.root:
        maxval = -1
    elif 'galaxy' in cfg.data.root:
        maxval = 1023.
    elif 'nikon' in cfg.data.root:
        maxval = 16383.
    elif 'sony' in cfg.data.root:
        maxval = 16383.
    else:
        maxval = cfg.data.maxval

    dataset = LSMI(cfg=cfg,
                   root=cfg.data.root,
                   split=split,
                   normalize=cfg.data.normalize,
                   maxval=maxval,
                   img_pool=cfg.data.img_pool,
                   uncalculable=cfg.data.uncalculable,
                   mask_uncalculable=cfg.data.mask_uncalculable,
                   mask_black=cfg.data.mask_black,
                   mask_highlight=cfg.data.mask_highlight,
                   illum_aug=random_color,
                   chroma_sampling_method=cfg.data.chroma_sampling_method,
                   transform=tsfm)
    
    dataloader = data.DataLoader(dataset,batch_size=cfg.test.batch_size,shuffle=False,
                                 pin_memory=True,num_workers=cfg.test.num_workers)

    return dataloader
