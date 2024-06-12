import math
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching
from ..utils import custom_meshgrid



class NeuLFNetwork(nn.Module):
    def __init__(self,
                 encoding="None",
                 input_dim = 6,
                 num_layers=8,
                 hidden_dim=256,

                 cuda_ray=False,

                 ):
        super().__init__()
        self.in_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rgb_dim = 3

        self.cuda_ray = cuda_ray

        mlp_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = self.rgb_dim
            else:
                out_dim = hidden_dim
            
            mlp_net.append(nn.Linear(in_dim, out_dim, bias=True))

        self.mlp_net = nn.ModuleList(mlp_net)


    def forward(self,x):

        for l in range(self.num_layers):
            x = self.mlp_net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)

        color = torch.sigmoid(x)

        return color


    
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
    
class CLinear(nn.Linear):
    
    def forward(self, input):
        
        if self.weight.is_complex() and not input.is_complex():
            input = input*1j
            
        return super().forward(input)

class NeuLFWireNetwork(nn.Module):
    def __init__(self,
                 encoding="None",
                 input_dim = 6,
                 num_layers=8,
                 hidden_dim=256,
                 sigma=40,
                 omega=40,
                 test = 'in',
                 cuda_ray=False,

                 ):
        super().__init__()
        self.in_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rgb_dim = 3

        self.sigma = sigma
        self.omega = omega
        self.nonlin = lambda x : torch.exp(self.omega*x - self.sigma*x.abs().square())


        self.cuda_ray = cuda_ray

        self.test = test

        mlp_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
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

        if self.test == 'full':
            for l in range(self.num_layers):
                x = self.mlp_net[l](x)
                if l != self.num_layers - 1:
                    x = self.nonlin(x)
                    

            color = torch.sigmoid(x.real)

            return color
        elif self.test == 'in':
            for l in range(self.num_layers):
                x = self.mlp_net[l](x)
                if l == 0:
                    x = self.nonlin(x)
                elif l != self.num_layers -1:
                    x = F.relu(x, inplace=True)
                    

            color = torch.sigmoid(x.real)
            return color
        
        elif self.test == 'out':
            for l in range(self.num_layers):
                x = self.mlp_net[l](x)
                if l == self.num_layers -2:
                    x = self.nonlin(x)
                elif l != self.num_layers -1:
                    x = F.relu(x, inplace=True)
                    

            color = torch.sigmoid(x.real)
            return color
        
        elif self.test == 'in2':
            for l in range(self.num_layers):
                x = self.mlp_net[l](x)
                if l == 0:
                    x = self.nonlin(x)
                elif l == 1:
                    x = self.nonlin(x)    
                elif l != self.num_layers -1:
                    x = F.relu(x, inplace=True)
                    

            color = torch.sigmoid(x.real)
            return color
        
        elif self.test == 'even':
            for l in range(self.num_layers):
                x = self.mlp_net[l](x)
                if l in [0,2,4,6,8] :
                    x = self.nonlin(x)

                elif l != self.num_layers -1:
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
    


class NeuLFWireNetwork2(nn.Module):
    def __init__(self,
                 encoding="None",
                 input_dim = 6,
                 num_layers=8,
                 hidden_dim=256,
                 sigma=40,
                 omega=40,
                 test = 'all',
                 cuda_ray=False,

                 ):
        super().__init__()
        self.in_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rgb_dim = 3

        self.sigma = sigma
        self.omega = omega
        self.nonlin = lambda x : torch.exp(self.omega*x - self.sigma*x.abs().square())


        self.cuda_ray = cuda_ray

        self.test = test

        mlp_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
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

        if self.test == 'full':
            for l in range(self.num_layers):
                x = self.mlp_net[l](x)
                if l != self.num_layers - 1:
                    x = self.nonlin(x)
                    

            color = torch.sigmoid(x.real)

            return color
        elif self.test == 'in':
            for l in range(self.num_layers):
                x = self.mlp_net[l](x)
                if l == 0:
                    x = self.nonlin(x)
                elif l != self.num_layers -1:
                    x = F.relu(x, inplace=True)
                    

            color = torch.sigmoid(x.real)
            return color
        
        elif self.test == 'out':
            for l in range(self.num_layers):
                x = self.mlp_net[l](x)
                if l == self.num_layers -2:
                    x = self.nonlin(x)
                elif l != self.num_layers -1:
                    x = F.relu(x, inplace=True)
                    

            color = torch.sigmoid(x.real)
            return color
        
        elif self.test == 'in2':
            for l in range(self.num_layers):
                x = self.mlp_net[l](x)
                if l == 0:
                    x = self.nonlin(x)
                elif l == 2:
                    x = self.nonlin(x)    
                elif l != self.num_layers -1:
                    x = F.relu(x, inplace=True)
                    

            color = torch.sigmoid(x.real)
            return color
        
        elif self.test == 'even':
            for l in range(self.num_layers):
                x = self.mlp_net[l](x)
                if l in [0,2,4,6,8] :
                    x = self.nonlin(x)

                elif l != self.num_layers -1:
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