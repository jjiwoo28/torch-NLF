import math
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching
from ..utils import custom_meshgrid
from encoding import get_encoder


    
class CLinear(nn.Linear):
    
    def forward(self, input):
        
        if self.weight.is_complex() and not input.is_complex():
            input = input*1j
            
        return super().forward(input)

class FreqAndSHNeuLFNetwork(nn.Module):
    def __init__(self,
                 encoding="None",
                 input_dim = 6,
                 num_layers=8,
                 hidden_dim=256,
                 sigma=40,
                 omega=40,
                 act = 'wire',
                 cuda_ray=False,

                 ):
        super().__init__()
        self.in_dim = input_dim
        self.split_dim = self.in_dim// 2

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rgb_dim = 3

        self.sigma = sigma
        self.omega = omega
        self.nonlin = lambda x : torch.exp(self.omega*x - self.sigma*x.abs().square())

        self.encoder_o, self.o_dim = get_encoder("sphere_harmonics" , degree=5)
        self.encoder_d, self.d_dim = get_encoder("frequency" ,input_dim=self.split_dim ,multires=4)


        self.cuda_ray = cuda_ray

        self.act = act

        mlp_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.o_dim + self.d_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = self.rgb_dim
            else:
                out_dim = hidden_dim
            
            mlp_net.append(nn.Linear(in_dim, out_dim, bias=True))

        self.mlp_net = nn.ModuleList(mlp_net)
        self.init_weights()


    def forward(self,x):

        if self.act == "wire":
            d = x[...,:self.split_dim]
            o = x[...,self.split_dim:]

            o = self.encoder_o(o)
            d = self.encoder_d(d)

            x = torch.cat((d, o), dim=-1)

            for l in range(self.num_layers):
                x = self.mlp_net[l](x)
                if l == 0:
                    x = self.nonlin(x)
                elif l != self.num_layers -1:
                    x = F.relu(x, inplace=True)
                    

            color = torch.sigmoid(x.real)
            return color
        
        if self.act == "relu":
            d = x[...,:self.split_dim]
            o = x[...,self.split_dim:]

            o = self.encoder_o(o)
            d = self.encoder_d(d)

            x = torch.cat((d, o), dim=-1)

            for l in range(self.num_layers):
                x = self.mlp_net[l](x)
                # if l == 0:
                #     x = self.nonlin(x)
                if l != self.num_layers -1:
                    x = F.relu(x, inplace=True)
                    

            color = torch.sigmoid(x.real)
            return color
            
   

            
    @torch.no_grad() 
    def init_weights(self):
        # Initialize weights for Gabor and SIREN
        denom = max(1e-3, self.omega)
        
        for idx, m_mod in enumerate(self.mlp_net):
            if idx == 0:
                const = 1/(self.in_dim)
            else:
                const = np.sqrt(6/self.hidden_dim)/denom
                
                #if self.nonlin == 'complexgabor':
                const *= 1
            m_mod.weight.uniform_(-const, const)

    
    def render(self, rays_o, rays_d, staged=False, max_ray_batch=4096, **kwargs):
        
        device = rays_o.device
        xyzxyz = torch.cat((rays_o, rays_d), dim=2)

        image = self.forward(xyzxyz)

        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(1, 1, 3)
        # 가중치 적용하여 그레이스케일로 변환
        gray = torch.sum(image * weights, dim=2, keepdim=True)

        return {
            'depth': gray,
            'image': image,
        }

    def get_params(self, lr):

        params = [
            {'params': self.mlp_net.parameters(), 'lr': lr}
          
        ]
       
        return params
    


class WireAndSHNeuLFNetwork(nn.Module):

    def __init__(self,
                 encoding="None",
                 input_dim = 6,
                 num_layers=4,
                 hidden_dim=64,
                 sigma=40,
                 omega=40,
                 act = 'all',
                 cuda_ray=False,

                 ):
        super().__init__()
        self.in_dim = input_dim
        self.split_dim = self.in_dim//2
        

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rgb_dim = 3

        self.wire_num_layer = 3
        self.wire_hidden_dim = 64
        self.wire_feat_dim = 24

        self.sigma = sigma
        self.omega = omega
        self.nonlin = lambda x : torch.exp(self.omega*x - self.sigma*x.abs().square())

        self.encoder_o, self.o_dim = get_encoder("sphere_harmonics" , degree=5)



        self.cuda_ray = cuda_ray

        self.test = act

        wire_net = []

        for l in range(self.wire_num_layer):
            if l==0:
                in_dim = self.split_dim
            else:
                in_dim = self.wire_hidden_dim
            
            if l == self.wire_num_layer - 1:    
                out_dim = self.wire_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            wire_net.append(nn.Linear(in_dim, out_dim, bias=True))
        self.wire_net = nn.ModuleList(wire_net)

        mlp_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.o_dim + self.wire_feat_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = self.rgb_dim
            else:
                out_dim = hidden_dim
            
            mlp_net.append(nn.Linear(in_dim, out_dim, bias=True))

        self.mlp_net = nn.ModuleList(mlp_net)
        self.init_weights()


    def forward(self,x):
        d = x[...,:self.split_dim]
        o = x[...,self.split_dim:]

        o = self.encoder_o(o)

        for l in range(self.wire_num_layer):
            d = self.wire_net[l](d)
            if l == 0:
                d  = self.nonlin(d)
            
            elif l != self.wire_num_layer -1:
                d = F.relu(d, inplace=True)
        d = d.real

        x = torch.cat((d, o), dim=-1)

        for l in range(self.num_layers):
            x = self.mlp_net[l](x)
            # if l == 0:
            #     x = self.nonlin(x)
            if l != self.num_layers -1:
                x = F.relu(x, inplace=True)
                

        color = torch.sigmoid(x.real)
        return color
        
   

            
    @torch.no_grad() 
    def init_weights(self):
        # Initialize weights for Gabor and SIREN
        denom = max(1e-3, self.omega)
        
        for idx, m_mod in enumerate(self.mlp_net):
            if idx == 0:
                const = 1/(self.in_dim)
            else:
                const = np.sqrt(6/self.hidden_dim)/denom
                
                #if self.nonlin == 'complexgabor':
                const *= 1
            m_mod.weight.uniform_(-const, const)

        for idx, m_mod in enumerate(self.wire_net):
            if idx == 0:
                const = 1/(self.in_dim)
            else:
                const = np.sqrt(6/self.hidden_dim)/denom
                
                #if self.nonlin == 'complexgabor':
                const *= 1
            m_mod.weight.uniform_(-const, const)


    
    def render(self, rays_o, rays_d, staged=False, max_ray_batch=4096, **kwargs):
        
        device = rays_o.device
        xyzxyz = torch.cat((rays_o, rays_d), dim=2)

        image = self.forward(xyzxyz)

        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(1, 1, 3)
        # 가중치 적용하여 그레이스케일로 변환
        gray = torch.sum(image * weights, dim=2, keepdim=True)

        return {
            'depth': gray,
            'image': image,
        }

    def get_params(self, lr):

        params = [
            {'params': self.mlp_net.parameters(), 'lr': lr}
          
        ]
       
        return params
    

class FreqHNeuLFNetwork(nn.Module):
    def __init__(self,
                 encoding="None",
                 input_dim = 6,
                 num_layers=8,
                 hidden_dim=256,
                 sigma=40,
                 omega=40,
                 act = 'wire',
                 freq_degree = 4,
                 cuda_ray=False,

                 ):
        super().__init__()
        self.in_dim = input_dim
        self.split_dim = self.in_dim//2

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rgb_dim = 3

        self.sigma = sigma
        self.omega = omega
        self.nonlin = lambda x : torch.exp(self.omega*x - self.sigma*x.abs().square())

        #self.encoder_o, self.o_dim = get_encoder("sphere_harmonics" , degree=5)
        self.encoder_d, self.d_dim = get_encoder("frequency" ,input_dim=self.in_dim, multires=freq_degree)


        self.cuda_ray = cuda_ray

        self.act = act

        mlp_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim =self.d_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = self.rgb_dim
            else:
                out_dim = hidden_dim
            
            mlp_net.append(nn.Linear(in_dim, out_dim, bias=True))

        self.mlp_net = nn.ModuleList(mlp_net)
        self.init_weights()


    def forward(self,x):

        if self.act == "wire":
            
            
            x = self.encoder_d(x)

            for l in range(self.num_layers):
                x = self.mlp_net[l](x)
                if l == 0:
                    x = self.nonlin(x)
                elif l != self.num_layers -1:
                    x = F.relu(x, inplace=True)
                    

            color = torch.sigmoid(x.real)
            return color
        
        if self.act == "relu":
            x = self.encoder_d(x)


            for l in range(self.num_layers):
                x = self.mlp_net[l](x)
                # if l == 0:
                #     x = self.nonlin(x)
                if l != self.num_layers -1:
                    x = F.relu(x, inplace=True)
                    

            color = torch.sigmoid(x.real)
            return color
            
   

            
    @torch.no_grad() 
    def init_weights(self):
        # Initialize weights for Gabor and SIREN
        denom = max(1e-3, self.omega)
        
        for idx, m_mod in enumerate(self.mlp_net):
            if idx == 0:
                const = 1/(self.in_dim)
            else:
                const = np.sqrt(6/self.hidden_dim)/denom
                
                #if self.nonlin == 'complexgabor':
                const *= 1
            m_mod.weight.uniform_(-const, const)

    
    def render(self, rays_o, rays_d, staged=False, max_ray_batch=4096, **kwargs):
        

        device = rays_o.device
        xyzxyz = torch.cat((rays_o, rays_d), dim=2)

        image = self.forward(xyzxyz)

        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(1, 1, 3)
        # 가중치 적용하여 그레이스케일로 변환
        gray = torch.sum(image * weights, dim=2, keepdim=True)

        return {
            'depth': gray,
            'image': image,
        }

    def get_params(self, lr):

        params = [
            {'params': self.mlp_net.parameters(), 'lr': lr}
          
        ]
       
        return params
    
