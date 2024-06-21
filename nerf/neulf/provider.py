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

from ..utils import get_rays , get_rays2  , get_uvst
from ..provider import NeRFDataset


class NeuLFDataset(NeRFDataset):

    def __init__(self, opt, device, type='train', downscale=1, n_test=10 ):
        if type == 'train_neulf':
            super_type = "train"
        else:
            super_type =type
        super().__init__(opt, device, super_type, downscale, n_test)

        self.dim = 6
        self.split_dim = 3

        print(self.poses[:,0,3].min())

        print(self.poses[:,0,3].max())

        print(self.poses[:,1,3].min())

        print(self.poses[:,1,3].max())

        print(self.poses[:,2,3].min())
 
        print(self.poses[:,2,3].max())

        self.x_min = self.poses[:,0,3].min()

        self.x_max = self.poses[:,0,3].max()

        self.y_min = self.poses[:,1,3].min()

        self.y_max = self.poses[:,1,3].max()

        self.z_min = self.poses[:,2,3].min()

        self.z_max = self.poses[:,2,3].max()

  
        #self.set_top_two_dimensions()
        #self.normalize_poses()
        print(f"self.x_index : {self.x_index}  self.y_index : {self.y_index}")
       #breakpoint()
        if type != 'train_neulf':

            temp = []
            for pose in self.poses:
                pose_input = pose.unsqueeze(0)
                LF = self.get_LF(pose_input,self.intrinsics,self.H,self.W)
                temp.append(LF)

            self.poses = torch.cat(temp,dim=0)
            N  = self.poses.shape[0]
            self.poses = self.poses.view(N,self.H*self.W,self.poses.shape[-1])
            
          
            #self.images = self.images.view(N,self.H*self.W,self.poses.shape[-1])
            if self.images is None:
                self.train_loader = [{'lightfield': self.poses[i, :, :]} for i in range(self.poses.shape[0])]
            else:
                
                #self.images = self.images.view(N,self.H*self.W,self.images.shape[-1])
                self.train_loader = [{'lightfield': self.poses[i, :, :],
                                      'images': self.images[i,:,:]} for i in range(self.poses.shape[0])]
            print("test")

        
        
        if type == 'train_neulf':
            #breakpoint()
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

    def collate(self,index):

        index =  index[0]
        
        data = self.train_loader[index]

        LF = data['lightfield'].to(self.device)
        LF = LF.unsqueeze(0)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': LF[0,:,:self.split_dim].unsqueeze(0),
            'rays_d': LF[0,:,self.split_dim:].unsqueeze(0),
        }
        if self.images is not None:
            images = data['images'].to(self.device)
            images = images.unsqueeze(0)
            results['images'] = images
        #breakpoint()
        return results


    def dataloader(self):
        
        size = len(self.train_loader)
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=False, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
       
        if self.type == 'render':
            loader.is_render = True
        return loader
    
    def set_top_two_dimensions(self):
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min
        self.z_range = self.z_max - self.z_min

        self.x_center = (self.x_max + self.x_min) / 2
        self.y_center = (self.y_max + self.y_min) / 2
        self.z_center = (self.z_max + self.z_min) / 2

        ranges = [(self.x_range, 0), (self.y_range, 1), (self.z_range, 2)]

        # 차이가 큰 순서대로 정렬
        sorted_ranges = sorted(ranges, key=lambda x: x[0], reverse=True)

        # 상위 두 개 차원 인덱스를 설정
        self.length_max = sorted_ranges[0][0]

        self.x_index = sorted_ranges[0][1]
        self.y_index = sorted_ranges[1][1]
    
    def normalize_poses(self):
        # 정규화 함수
        def normalize(values, center, length):
            # 중심점 기준으로 정규화 실행
            return 2 * (values - center) / length

        # 각 축에 대해 정규화 실행
        self.poses[:, 0, 3] = normalize(self.poses[:, 0, 3], self.x_center, self.length_max)
        self.poses[:, 1, 3] = normalize(self.poses[:, 1, 3], self.y_center, self.length_max)
        self.poses[:, 2, 3] = normalize(self.poses[:, 2, 3], self.z_center, self.length_max)

        print("adjust poses")
        print(self.poses[:,0,3].min())

        print(self.poses[:,0,3].max())

        print(self.poses[:,1,3].min())

        print(self.poses[:,1,3].max())

        print(self.poses[:,2,3].min())
 
        print(self.poses[:,2,3].max())

        self.x_min = self.poses[:,0,3].min()

        self.x_max = self.poses[:,0,3].max()

        self.y_min = self.poses[:,1,3].min()

        self.y_max = self.poses[:,1,3].max()

        self.z_min = self.poses[:,2,3].min()

        self.z_max = self.poses[:,2,3].max()

    
class UVXYDataset(NeuLFDataset):
    def __init__(self, opt, device, type='train', downscale=1, n_test=10):
        super().__init__(opt, device, type, downscale, n_test)
        self.dim = 4  # UVXYDataset 클래스에 특화된 차원 수로 변경
        self.split_dim = 2


    def get_LF(self, pose, intrinsics , H, W):
            uvst = self.get_uvxy(pose,intrinsics,H,W)
            return uvst
    

    
    @torch.cuda.amp.autocast(enabled=False)
    def get_uvxy(self,poses, intrinsics, H, W):
        device = poses.device
        B = poses.shape[0]

        aspect = W / H
        u = torch.linspace(-1, 1, H, dtype=torch.float32, device=device)
        v = torch.linspace(1, -1, W, dtype=torch.float32, device=device) / aspect

        uv = torch.meshgrid(u, v, indexing='ij')  # ij-indexing for consistent dimension order

        u = uv[0].unsqueeze(0).expand(B, -1, -1)  # Directly expand to (B, H, W)
        v = uv[1].unsqueeze(0).expand(B, -1, -1)

        # Ensure poses is correctly broadcasted
        s = torch.ones_like(u) * poses[:, self.x_index, 3].view(B, 1, 1)  # Use view for safe broadcasting
        t = torch.ones_like(v) * poses[:, self.y_index, 3].view(B, 1, 1)

        uvst = torch.stack((u, v, s ,t), dim=-1)

        return uvst


class UVSTDataset(NeuLFDataset):
    def __init__(self, opt, device, type='train', downscale=1, n_test=10):
        super().__init__(opt, device, type, downscale, n_test)
        self.dim = 4  # UVXYDataset 클래스에 특화된 차원 수로 변경
        self.split_dim = 2

    def get_LF(self, pose, intrinsics , H, W):
            
            uvst = get_uvst(pose,intrinsics,H,W , self.focal_depth)
            return uvst

