import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer


class CLinear(nn.Linear):
    
    def forward(self, input):
        
        if self.weight.is_complex() and not input.is_complex():
            input = input*1j
            
        return super().forward(input)


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=4,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 bound=1,
                 nonlin='relu',
                 sigma=0.5,
                 omega=1.0,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding,
                                                desired_resolution=2048 * bound)
        
        self.encoding = encoding
        self.encoding_dir = encoding_dir
        self.encoding_bg = encoding_bg
        
        self.sigma = sigma
        self.omega = omega

        dtype = None
        
        # Set non linearity here
        if nonlin == 'relu':
            self.nonlin = torch.nn.functional.relu
        elif nonlin == 'gauss':
            self.nonlin = lambda x : torch.exp(-self.sigma*x*x)
        elif nonlin == 'sine':
            self.nonlin = lambda x : torch.sin(self.omega*x)
        elif nonlin == 'gabor':
            self.nonlin = lambda x : torch.sin(self.omega*x)*torch.exp(-self.sigma*x*x)
        elif nonlin == 'complexgabor':
            self.nonlin = lambda x : torch.exp(self.omega*x - self.sigma*x.abs().square())
            dtype = torch.cfloat
        elif nonlin == 'mexicanhat':
            self.nonlin = lambda x : (1 - self.sigma*x*x)*torch.exp(-self.sigma*x*x)
        else:
            raise ValueError('Non linearity %s not implemented' % nonlin)
        
        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(CLinear(in_dim, out_dim, bias=True, dtype=dtype))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim_color
            
            color_net.append(CLinear(in_dim, out_dim, bias=True, dtype=dtype))

        self.color_net = nn.ModuleList(color_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg,
                                                          input_dim=2,
                                                          num_levels=4,
                                                          log2_hashmap_size=19,
                                                          desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(CLinear(in_dim, out_dim, bias=True, dtype=dtype))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None
            
        if nonlin == 'sine' or nonlin == 'gabor' or nonlin == 'complexgabor':
            self.init_weights()
        
    @torch.no_grad()    
    def init_weights(self):
        # Initialize weights for Gabor and SIREN
        denom = max(1e-3, self.omega)
        
        for idx, m_mod in enumerate(self.sigma_net):
            if idx == 0:
                const = 1/(self.in_dim)
            else:
                const = np.sqrt(6/self.hidden_dim)/denom
                
                if self.nonlin == 'complexgabor':
                    const *= 1
            m_mod.weight.uniform_(-const, const)

        for idx, m_mod in enumerate(self.color_net):
            if idx == 0:
                const = 1/(self.in_dim_dir + self.geo_feat_dim)
            else:
                const = np.sqrt(6/self.hidden_dim_color)/denom
                if self.nonlin == 'complexgabor':
                    const *= 1
            m_mod.weight.uniform_(-const, const)
            
        if self.bg_net is not None:
            for idx, m_mod in enumerate(self.bg_net):
                if idx == 0:
                    const = 1/(self.in_dim_bg + self.in_dim_dir)
                else:
                    const = np.sqrt(6/self.hidden_dim_bg)/denom
                    if self.nonlin == 'complexgabor':
                        const *= 2
                m_mod.weight.uniform_(-const, const)
                
    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)
        
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = self.nonlin(h)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0].real)
        geo_feat = h[..., 1:].real

        # color
        
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = self.nonlin(h)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h.real)

        return sigma, color

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = self.nonlin(h)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0].real)
        geo_feat = h[..., 1:].real

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = self.nonlin(h)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h.real)

        return rgbs

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = self.nonlin(h)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h.real)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
        ]
        if self.encoding != 'None':
            params.append({'params': self.encoder.parameters(), 'lr': lr})
        if self.encoding_dir != 'None':
            params.append({'params': self.encoder_dir.parameters(), 'lr': lr})
        if self.bg_radius > 0:
            if self.encoder_bg != 'None':
                params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params
