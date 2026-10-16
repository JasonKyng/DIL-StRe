import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

from einops import rearrange
from models.modules.galerkin import simple_attn

class SplineBasis_2nd(torch.autograd.Function):    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        t = torch.where(x <= -1.5, 0.0 * x, 0.5 * (1.5 + x)**2)
        t = torch.where(x <= -0.5, t, 0.75 - x**2)
        t = torch.where(x <= 0.5, t, 0.5 * (1.5 - x)**2)
        w = torch.where(x > 1.5, 0.0 * x, t)
        return w
        
    @staticmethod
    def backward(ctx, grad_in):
        x, = ctx.saved_tensors
        t = torch.where(x <= -1.5, 0.0 * x, 1.5 + x)
        t = torch.where(x <= -0.5, t, -2 * x)
        t = torch.where(x <= 0.5, t, -(1.5 - x))
        w = torch.where(x > 1.5, 0.0 * x, t)
        return w * grad_in

class SplineBasis_3rd(torch.autograd.Function):    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        t = torch.where(x <= -2.0, 0.0 * x, ((2.0 + x)**3) / 6.0)
        t = torch.where(x <= -1.0, t, (4.0 - 6.0 * (x**2) - 3.0 * (x**3)) / 6.0)
        t = torch.where(x <= 0.0, t, (4.0 - 6.0 * (x**2) + 3.0 * (x**3)) / 6.0)
        t = torch.where(x <= 1.0, t, ((2.0 - x)**3) / 6.0)
        w = torch.where(x > 2.0, 0.0 * x, t)
        return w
    
    @staticmethod
    def backward(ctx, grad_in):
        x, = ctx.saved_tensors
        t = torch.where(x <= -2.0, 0.0 * x, 0.5 * ((2.0 + x)**2))
        t = torch.where(x <= -1.0, t, -2.0 * x - 1.5 * (x**2))
        t = torch.where(x <= 0.0, t, -2.0 * x + 1.5 * (x**2))
        t = torch.where(x <= 1.0, t, -0.5 * ((2.0 - x)**2))
        w = torch.where(x > 2.0, 0.0 * x, t)
        return w * grad_in

class SplineBasis_4th(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        t = torch.where(x <= -2.5, 0.0 * x, ((x**4) + 10 * (x**3) + 37.5 * (x**2) + 62.5 * x + 39.0625)/24.)
        t = torch.where(x <= -1.5, t, (-4 * (x**4) - 20 * (x**3) - 30 * (x**2) - 5 * x + 13.75)/24.)
        t = torch.where(x <= -0.5, t, (6 * (x**4) - 15 * (x**2) + 14.375)/24.)
        t = torch.where(x <= 0.5, t, (-4 * (x**4) + 20 * (x**3) - 30 * (x**2) + 5 * x + 13.75)/24.)
        t = torch.where(x <= 1.5, t, ((x**4) - 10 * (x**3) + 37.5 * (x**2) - 62.5 * x + 39.0625)/24.)
        w = torch.where(x > 2.5, 0.0 * x, t)
        return w
    
    @staticmethod
    def backward(ctx, grad_in):
        x, = ctx.saved_tensors
        t = torch.where(x <= -2.5, 0.0 * x, (4 * (x**3) + 30 * (x**2) + 75 * x + 62.5)/24.)
        t = torch.where(x <= -1.5, t, (-16 * (x**3) - 60 * (x**2) - 60 * x - 5)/24.)
        t = torch.where(x <= -0.5, t, (24 * (x**3) - 30 * x)/24.)
        t = torch.where(x <= 0.5, t, (-16 * (x**3) + 60 * (x**2) - 60 * x + 5)/24.)
        t = torch.where(x <= 1.5, t, (4 * (x**3) - 30 * (x**2) + 75 * x - 62.5)/24.)
        w = torch.where(x > 2.5, 0.0 * x, t)
        return w * grad_in
    
def spline_basis(x, basis_ord=3):
    if basis_ord==2:
        x.clamp_(-1.5 + 1e-6, 1.5 - 1e-6)
        return SplineBasis_2nd.apply(x)
    elif basis_ord==3:
        x.clamp_(-2.0 + 1e-6, 2.0 - 1e-6)
        return SplineBasis_3rd.apply(x)
    elif basis_ord==4:
        x.clamp_(-2.5 + 1e-6, 2.5 - 1e-6)
        return SplineBasis_4th.apply(x)
    else:
        raise ValueError

def Sinkhorn(K, u, v, thresh):
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    # thresh = args.ot_threshold
    for i in range(50):
        r0 = r
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break
    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

    return T

@register('btc')
class BTC(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, hidden_dim=256, basis_ord=3):
        super().__init__()
        self.encoder = models.make(encoder_spec)
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.knot = nn.Conv2d(self.encoder.out_dim, int(2 * math.sqrt(hidden_dim)), 3, padding=1)
        self.dilation = nn.Linear(1, int(math.sqrt(hidden_dim)), bias=False)
        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})
        self.basis_ord = basis_ord
        
    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        self.coeff = self.coef(self.feat)
        self.knott = self.knot(self.feat)
        return self.feat
    
    def query_rgb(self, coord, cell=None):
        feat = self.feat
        coef = self.coeff
        knot = self.knott
        
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1).unsqueeze(0)\
            .expand(feat.shape[0], 2, *feat.shape[-2:])
        
        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                
                q_coef = F.grid_sample(
                    coef, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_knot = F.grid_sample(
                    knot, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)                  

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                
                bs, q = coord.shape[:2]
                q_knot = torch.stack(torch.split(q_knot, 2, dim=-1), dim=-1) # bs, q, 2, h'//2
                rel_coord = torch.stack(torch.split(rel_coord, 2, dim=-1), dim=-1) # bs, q, 2, 1
                dilation = self.dilation(rel_cell[:, :, 0:1].view((bs * q, -1))).view(bs, q, -1).unsqueeze(dim=-2) # bs, q, 1, h'//2
                
                q_knot = rel_coord - q_knot
                q_knot = q_knot * dilation
                q_knot = spline_basis(q_knot, self.basis_ord) # bs, q, 2, h'//2
                y_basis = q_knot[:,:,0,:].unsqueeze(-1) # bs, q, h'//2, 1
                x_basis = q_knot[:,:,1,:].unsqueeze(-2) # bs, q, 1, h'//2
                inp = torch.matmul(y_basis, x_basis).view(bs, q, -1) # bs, q, (h'//2)^2
                inp = q_coef * inp

                # pred = self.imnet(inp.contiguous().view(bs * q, -1)).view(bs, q, -1)
                # preds.append(pred)
                # if len(inp.shape) == 3:
                #     inp = inp.unsqueeze(0)
                pred = self.imnet(inp)
                preds.append(pred)
                # preds.append(rearrange(pred, 'b c h w -> b (h w) c', w=self.inp.shape[-1]))


                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)
                
        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area)
        if self.training:
            # add nosie to the input
            noise = torch.randn_like(self.inp) * 0.1
            res_inp = self.inp + noise

            # res_inp = torch.zeros_like(self.inp)
        else:
            res_inp = self.inp
        ret += F.grid_sample(res_inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return ret
        # res_inp = self.inp
        # lr_bi = F.grid_sample(res_inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
        #               padding_mode='border', align_corners=False)[:, :, 0, :] \
        #               .permute(0, 2, 1)

        # return out_ot
    
    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
    
@register('btc-attn')
class BTCAttn(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, hidden_dim=256, basis_ord=3):
        super().__init__()
        self.encoder = models.make(encoder_spec)
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.knot = nn.Conv2d(self.encoder.out_dim, int(2 * math.sqrt(hidden_dim)), 3, padding=1)
        self.dilation = nn.Linear(1, int(math.sqrt(hidden_dim)), bias=False)
        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})
        self.sim_attn = simple_attn(64, 8)
        # self.sim_attn2 = simple_attn(64, 8)
        self.basis_ord = basis_ord
        
    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        
        h,w = self.feat.shape[-2:]

        ########## DIL ##########
        self.feat = self.sim_attn(rearrange(self.feat, 'b c h w -> b (h w) c'))
        self.feat = rearrange(self.feat, 'b (h w) c -> b c h w', h=h, w=w)

        # self.feat = self.sim_attn2(rearrange(self.feat, 'b c h w -> b (h w) c'))
        # self.feat = rearrange(self.feat, 'b (h w) c -> b c h w', h=h, w=w)

        # if self.training:
        #     # add nosie to the input
        #     noise = torch.randn_like(self.feat) * 1
        #     self.feat += noise
        
        self.coeff = self.coef(self.feat)
        self.knott = self.knot(self.feat)
        return self.feat
    
    def query_rgb(self, coord, cell=None, pre_loss=None):
        feat = self.feat
        coef = self.coeff
        knot = self.knott
        
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1).unsqueeze(0)\
            .expand(feat.shape[0], 2, *feat.shape[-2:])
        
        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                
                q_coef = F.grid_sample(
                    coef, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_knot = F.grid_sample(
                    knot, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)                  

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                
                bs, q = coord.shape[:2]
                q_knot = torch.stack(torch.split(q_knot, 2, dim=-1), dim=-1) # bs, q, 2, h'//2
                rel_coord = torch.stack(torch.split(rel_coord, 2, dim=-1), dim=-1) # bs, q, 2, 1
                dilation = self.dilation(rel_cell[:, :, 0:1].view((bs * q, -1))).view(bs, q, -1).unsqueeze(dim=-2) # bs, q, 1, h'//2
                
                q_knot = rel_coord - q_knot
                q_knot = q_knot * dilation
                q_knot = spline_basis(q_knot, self.basis_ord) # bs, q, 2, h'//2
                y_basis = q_knot[:,:,0,:].unsqueeze(-1) # bs, q, h'//2, 1
                x_basis = q_knot[:,:,1,:].unsqueeze(-2) # bs, q, 1, h'//2
                inp = torch.matmul(y_basis, x_basis).view(bs, q, -1) # bs, q, (h'//2)^2
                inp = q_coef * inp

                pred = self.imnet(inp.contiguous().view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)
                # if len(inp.shape) == 3:
                #     inp = inp.unsqueeze(0)
                # pred = self.imnet(inp)
                # preds.append(pred)
                # preds.append(rearrange(pred, 'b c h w -> b (h w) c', w=self.inp.shape[-1]))


                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)
                
        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area)
        
        # ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
        #               padding_mode='border', align_corners=False)[:, :, 0, :] \
        #               .permute(0, 2, 1)
        
        ########## StRe ##########
        if self.training:
            # add nosie to the input
            # n_w = pre_loss * 5
            # n_w = torch.abs(gt - pred).mean()*0.8
            # n_w = torch.abs(gt - pred) * 0.1
            noise = torch.randn_like(self.inp) * 0.1
            res_inp = self.inp + noise
            # ret += noise
            # res_inp = torch.zeros_like(self.inp)
        else:
            res_inp = self.inp
        ret += F.grid_sample(res_inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return ret
        # res_inp = self.inp
        # lr_bi = F.grid_sample(res_inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
        #               padding_mode='border', align_corners=False)[:, :, 0, :] \
        #               .permute(0, 2, 1)

        # return out_ot
    
    def forward(self, inp, coord, cell, pre_loss=None):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell, pre_loss)