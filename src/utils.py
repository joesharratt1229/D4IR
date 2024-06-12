import torch
#import torch_radon

import numpy as np

import random


def sample_gaussian(m, v):
    epsilon = torch.randn_like(m)
    std = torch.sqrt(v)
    z = m + std * epsilon
    return z


class PoissonModel: # for phase retrieval 
    def __init__(self, alphas):
        super().__init__()
        self.alphas = alphas
        
    def __call__(self, z, alpha):
        
        z2 = z ** 2
        intensity_noise = alpha/255 * torch.abs(z) * torch.randn_like(z)
                
        y2 = torch.clamp(z2 + intensity_noise, min=0)
        y = torch.sqrt(y2)

        rr =  y - torch.abs(z)
        sigma = rr.std()

        return y, sigma
    
    

class GaussianModelD:  # discrete noise levels
    def __init__(self, sigmas):
        super().__init__()
        self.sigmas = sigmas
        
    def __call__(self, x, sigma):
        sigma = sigma / 255.
        y = x + torch.randn(*x.shape) * sigma
              
        return y, sigma
    

def fft(img: torch.Tensor
        ) -> torch.Tensor:
    img_new = torch.fft.ifftshift(img, dim = (-2, -1))
    img_new = torch.fft.fftn(img_new, dim = (-2, -1), norm = 'ortho')
    img_new = torch.fft.fftshift(img_new, dim = (-2, -1))

    return img_new

def ifft(img: torch.Tensor
         ) -> torch.Tensor:
    img_new = torch.fft.ifftshift(img, dim = (-2, -1))
    img_new = torch.fft.ifftn(img_new, dim = (-2, -1), norm = 'ortho')
    img_new = torch.fft.fftshift(img_new, dim = (-2, -1))
    return img_new


def kron(a, b):
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]

    return res.reshape(siz0 + siz1)

def spi_forward(x, K, alpha, q):
    ones = torch.ones(1, 1, K, K).to(x.device)
    theta = alpha * torch.kron(x, ones) / (K**2)
    y = torch.poisson(theta)
    ob = (y >= torch.ones_like(y) * q).float()
    return ob

def real2complex(x):
    return x.dtype(torch.complex64)

def cdp_forward(data, mask):
    sampling_rate = mask.shape[1]
    x = data.expand(-1, sampling_rate, -1, -1)
    masked_data = x * mask
    forward_data = torch.fft.fftn(masked_data, dim = (-2, -1), norm = 'ortho')
    return forward_data

def cdp_backward(data, mask):
    Ifft_data = torch.fft.ifftn(data, dim = (-2, -1), norm = 'ortho')
    backward_data = Ifft_data * mask.conj()
    return backward_data.mean(1, keepdim=True)


def spi_inverse(ztilde, K1, K, mu):
    """
    Proximal operator "Prox\_{\frac{1}{\mu} D}" for single photon imaging
    assert alpha == K and q == 1
    """
    z = torch.zeros_like(ztilde)

    K0 = K**2 - K1
    indices_0 = (K1 == 0)

    z[indices_0] = ztilde[indices_0] - (K0 / mu)[indices_0]

    func = lambda y: K1 / (torch.exp(y) - 1) - mu * y - K0 + mu * ztilde

    indices_1 = torch.logical_not(indices_0)

    # differentiable binary search
    bmin = 1e-5 * torch.ones_like(ztilde)
    bmax = 1.1 * torch.ones_like(ztilde)

    bave = (bmin + bmax) / 2.0

    for i in range(10):
        tmp = func(bave)
        indices_pos = torch.logical_and(tmp > 0, indices_1)
        indices_neg = torch.logical_and(tmp < 0, indices_1)
        indices_zero = torch.logical_and(tmp == 0, indices_1)
        indices_0 = torch.logical_or(indices_0, indices_zero)
        indices_1 = torch.logical_not(indices_0)

        bmin[indices_pos] = bave[indices_pos]
        bmax[indices_neg] = bave[indices_neg]
        bave[indices_1] = (bmin[indices_1] + bmax[indices_1]) / 2.0

    z[K1 != 0] = bave[K1 != 0]
    return torch.clamp(z, 0.0, 1.0)



def power_method_opnorm(normal_op, x, n_iter=10):
    def _normalize(x):
        size = x.size()
        x = x.view(size[0], -1)
        norm = torch.norm(x, dim=1)
        x /= norm.view(-1, 1)
        return x.view(*size), torch.max(norm).item()
    
    with torch.no_grad():
        x, _ = _normalize(x)

        for i in range(n_iter):
            next_x = normal_op(x)
            x, v = _normalize(next_x)

    return v**0.5


#class Radon_norm(Radon):
#    def __init__(self, resolution, angles, det_count=-1, det_spacing=1.0, clip_to_circle=False, opnorm=None):
#        if opnorm is None:
#        super(Radon_norm, self).__init__(resolution, angles, det_count, det_spacing, clip_to_circle)#
#            normal_op = lambda x: super(Radon_norm, self).backward(super(Radon_norm, self).forward(x))
#            x = torch.randn(1, 1, resolution, resolution).cuda()
#            opnorm = power_method_opnorm(normal_op, x, n_iter=10)
#        self.opnorm = opnorm
#        self.resolution = resolution
#        self.view = angles.shape[0]

#    def backprojection_norm(self, sinogram):
#        return self.backprojection(sinogram) / self.opnorm**2

#    def filter_backprojection(self, sinogram):
#        return self.backprojection(sinogram)
#        sinogram = self.filter_sinogram(sinogram, filter_name='ramp')

#    def normal_operator(self, x):
#        return self.backprojection_norm(self.forward(x))


#def create_radon(resolution, view, opnorm):
#    angles = torch.linspace(0, 179/180*np.pi, view)
#    det_count = int(np.ceil(np.sqrt(2) * resolution))
#    radon = Radon_norm(resolution, angles, det_count, opnorm=opnorm)
#    return radon


#class RadonGenerator:
#    def __init__(self):
#        self.opnorms = {}

#    def __call__(self, resolution, view):
#        key = (resolution, view)
#        if key in self.opnorms:
#            opnorm = self.opnorms[key]
#            radon = create_radon(resolution, view, opnorm)
#        else:
#            radon = create_radon(resolution, view, opnorm=None)
#
#            opnorm = radon.opnorm
#            self.opnorms[key] = opnorm
#        return radon