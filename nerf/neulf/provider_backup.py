import os
import cv2
import glob
import json
from cv2 import transform
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader

from ..utils import get_rays , get_rays2
from ..provider import NeRFDataset


class NeuLFDataset(NeRFDataset):

    def __init__(self, opt, device, type='train', downscale=1, n_test=10):
        if type == 'train_neulf':
            super_type = "train"
        else:
            super_type =type
        super().__init__(opt, device, super_type, downscale, n_test)
        
        
        if type == 'train_neulf':
            self.type  = 'train_neulf'
            self.batch_size = 8192
            temp = []
            for pose in self.poses:
                pose_input = pose.unsqueeze(0)
                LF = self.get_LF(pose_input,self.intrinsics,self.H,self.W)
                temp.append(LF)

            self.poses = torch.cat(temp,dim=0)
            self.poses = self.poses.view(-1, self.poses.shape[-1])
            self.whole_size = self.poses.shape[0]
            self.images = self.images.view(-1, self.images.shape[-1])


            perm = torch.randperm(self.whole_size)

            self.poses = self.poses[perm]
            self.images = self.images[perm]

            
            self.start , self.end = [],[]
            s = 0
            self.batch_count = 0
            while s < self.whole_size:
                self.start.append(s)
                s += self.batch_size
                self.end.append(min(s,self.whole_size))
            
            self.train_loader = [{'lightfield': self.poses[s:e], 
                                    'images': self.images[s:e]} for s, e in zip(self.start, self.end)]

            print("finish shuffle")

    def get_LF(self, pose, intrinsics , H, W):
            rays = get_rays2(pose,intrinsics,H,W)
            rays_o = rays["rays_o"]
            rays_d = rays["rays_d"]
            xyzxyz = torch.cat((rays_d,rays_o),dim=3)
            return xyzxyz

    def collate_neulf_train(self,index):

        index =  index[0]
        
        data = self.train_loader[index]

        LF = data['lightfield'].to(self.device)
        images = data['images'].to(self.device)
        LF = LF.unsqueeze(0)
        images = images.unsqueeze(0)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': LF[0,:,:3].unsqueeze(0),
            'rays_d': LF[0,:,3:].unsqueeze(0),
            'images': images
        }
        #breakpoint()
        return results
    
    def collate(self, index):


        B = len(index) # a list of length 1

        # random pose without gt images.
       
        poses = self.poses[index].to(self.device) # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]
        
        rays = get_rays(poses, self.intrinsics, self.H, self.W, -1, error_map, self.opt.patch_size)
        
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }

        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            # if self.training:
            #     C = images.shape[-1]
            #     images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            if self.training:
                B, H, W, C = images.shape
                images = images.view(B, H*W, C)
            results['images'] = images
        
        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']
            
        return results
    
    

    def dataloader(self):
        if(self.type == "train_neulf"):
            size = len(self.train_loader)
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate_neulf_train, shuffle=False, num_workers=0)
            loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
            loader.has_gt = self.images is not None
        else:
            size = len(self.poses)
            if self.training and self.rand_pose > 0:
                size += size // self.rand_pose # index >= size means we use random pose.
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
            loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
            loader.has_gt = self.images is not None
       
        if self.type == 'render':
            loader.is_render = True
        return loader
    
    