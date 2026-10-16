import os
import time
import random
import shutil

import torch
import torch.nn.functional as F
import numpy as np

from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # sets the seed for cpu
    torch.cuda.manual_seed(seed) # Sets the seed for the current GPU.
    torch.cuda.manual_seed_all(seed) #  Sets the seed for the all GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=False
    torch.set_deterministic(True)


def numpy_random_init(worker_id):
    process_seed = torch.initial_seed()
    base_seed    = process_seed - worker_id
    ss  = np.random.SeedSequence([worker_id, base_seed])
    np.random.seed(ss.generate_state(4))


def numpy_fix_init(worker_id):
    np.random.seed(2<<16 + worker_id)
    

numpy_init_dict = {
    "train": numpy_random_init,
    "val"  : numpy_fix_init,
    "test" : numpy_fix_init
}


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
        coord_x = -1+(2*i+1)/W
        coord_y = -1+(2*i+1)/H
        normalize to (-1, 1)
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img, flatten=True):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:], flatten=flatten)
    if flatten:
        rgb = img.view(3, -1).permute(1, 0)
    else:
        rgb = img
    return coord, rgb

def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1, calc_ssim=True):
    # import torch
    
    diff = (sr - hr) / rgb_range
    
    if dataset is not None:
        if dataset in ['sci1k', 'scid', 'siqad']:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
            shave = 1
        elif dataset in ['set5', 'set14', 'urban100', 'b100']:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
            shave = scale
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        
        valid = diff[..., shave:-shave, shave:-shave]
        sr_valid = sr[..., shave:-shave, shave:-shave]
        hr_valid = hr[..., shave:-shave, shave:-shave]
    else:
        valid = diff
        sr_valid = sr
        hr_valid = hr
    
    # Calculate PSNR
    mse = valid.pow(2).mean()
    psnr = -10 * torch.log10(mse)
    
    # Calculate SSIM if requested
    if calc_ssim:
        ssim = calculate_ssim(sr_valid, hr_valid, rgb_range)
        return psnr, ssim
    
    return psnr


def calculate_ssim(img1, img2, rgb_range=1, window_size=11, channel=None):
    """Calculate SSIM (Structural Similarity Index)"""
    # import torch
    # import torch.nn.functional as F
    
    # Normalize to [0, 1] range
    img1 = img1 / rgb_range
    img2 = img2 / rgb_range
    
    if channel is None:
        channel = img1.size(1)
    
    # Create Gaussian window
    def create_window(window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.exp(torch.tensor([-(x - window_size//2)**2 / (2*sigma**2) 
                                           for x in range(window_size)], dtype=torch.float32))
            return gauss / gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    window = create_window(window_size, channel).to(img1.device).type_as(img1)
    
    # SSIM constants
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    # Calculate means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


# def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
#     diff = (sr - hr) / rgb_range
#     if dataset is not None:
#         if dataset in ['sci1k', 'scid', 'siqad']:
#             gray_coeffs = [65.738, 129.057, 25.064]
#             convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
#             diff = diff.mul(convert).sum(dim=1)
#             shave = 1
#         elif dataset in ['set5', 'set14', 'urban100', 'b100']:
#             gray_coeffs = [65.738, 129.057, 25.064]
#             convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
#             diff = diff.mul(convert).sum(dim=1)
#             shave = scale
#         elif dataset == 'div2k':
#             shave = scale + 6
#         else:
#             raise NotImplementedError
#         valid = diff[..., shave:-shave, shave:-shave]
        
#     else:
#         valid = diff
#     mse = valid.pow(2).mean()
#     return -10 * torch.log10(mse)

# import functools
import torch.nn.functional as F

class FPL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_sp, self.gamma_sp = 1, 0.5
        self.alpha_lp, self.gamma_lp = 1, 1
        # self.upscale_func = functools.partial(
        #     F.interpolate, mode='bicubic', align_corners=False)
        self.weig_func = lambda x, y, z: torch.exp((x-x.min()) / (x.max()-x.min()) * y) * z

    def forward(self, x, hr, lr, coord):
        # f_BI_x = self.upscale_func(lr, size=hr.size()[2:])
        f_BI_x = F.grid_sample(lr, coord.flip(-1).unsqueeze(1), mode='bicubic',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)

        # y_sp = torch.abs(hr - x)
        # w_y_sp = self.weig_func(y_sp, self.alpha_sp, self.gamma_sp).detach()

        y_sp = torch.abs(hr - f_BI_x)
        w_y_sp = self.weig_func(y_sp, self.alpha_sp, self.gamma_sp).detach()

        y_lp = torch.abs(hr - x)
        w_y_lp = self.weig_func(y_lp, self.alpha_lp, self.gamma_lp).detach()

        # y_hat = hr
        loss = torch.mean(w_y_sp * w_y_lp * torch.abs(hr - x))
        # loss = torch.mean(w_y_sp * y_sp)

        return loss