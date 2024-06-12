import torch

import utils
from pnp.denoiser import UNetDenoiser2D


class CsmriSolverMixin:
    def _forward_csmri(self, env_ob, parameters):
        mu, sigma_d = parameters['mu'], parameters['sigma_d']
        state = env_ob['output']
        mask = env_ob['mask']
        y0 = env_ob['y0']
        
        B = state.shape[0]
        
        x, z, u = torch.chunk(state, chunks = 3, dim = 1)
        #mask = mask.unsqueeze(1) 
        
        
        for i in range(self.iter_num):
            _sigma_d = sigma_d[:, i]
            _mu = mu[:, i] 
            temp_var = (z - u)
            x = self.denoiser(temp_var.real, _sigma_d)
            z = utils.fft(x + u)
            _mu = _mu.view(B, 1, 1, 1)
            temp = ((_mu * z.clone()) + y0)/(1+ _mu)
            z[mask] = temp[mask]
            z = utils.ifft(z)
            
            u = u + x - z
        
        return torch.cat((x, z, u), dim = 1)
            
        
class CTSolverMixin:
    pass



class PrSolverMixin:
    def _forward_pr(self, env_ob, parameters):
        sigma_d, mu, tau = parameters['sigma_d'], parameters['mu'], parameters['tau']
        state = env_ob['output']
        y0 = env_ob['y0']
        mask = env_ob['mask']
        
        x, z, u = torch.chunk(state, chunks = 3, dim = 1)
        B = x.shape[0]
        
        for i in range(self.iter_num):
            _sigma_d = sigma_d[:, i]
            _mu = mu[:, i]
            _tau = tau[:, i]
            temp_var = (z - u)
            x = self.denoiser(temp_var.real, _sigma_d)
            
            _tau = _tau.view(B, 1, 1, 1)
            _mu = _mu.view(B, 1, 1, 1)
            
            Az = utils.cdp_forward(z, mask)  
            y_hat = Az.abs()
            meas_err = y_hat - y0
            gradient_forward = torch.stack((meas_err/y_hat*Az[...,0], meas_err/y_hat*Az[...,1]), -1)
            gradient = utils.cdp_backward(gradient_forward, mask)
            z = z - _tau * (gradient + _mu * (z - (x + u)))
            
        return torch.cat((x, z, u), dim = 1)
    

class SpiSolverMixin:
    def _forward_spi(self, env_ob, parameters):
        mu, sigma_d = parameters['mu'], parameters['sigma_d']
        state = env_ob['output']
        K = env_ob['K']
        
        x, z, u = torch.chunk(state, chunks = 3, dim = 1)
        
        B = state.shape[0]
        
        K = K[:, 0, 0, 0].view(B, 1, 1, 1) * 10 
        K1 = env_ob['x0'] * (K ** 2)
        
        for i in range(self.iter_num):
            _sigma_d = sigma_d[:, i]
            _mu = mu[:, i]        
            _mu = _mu.view(B, 1, 1, 1)

            # z step (x + u)
            z = utils.spi_inverse(x + u, K1, K, _mu)

            # u step
            u = u + x - z

            # x step
            x = self.denoiser((z - u), _sigma_d) 
        
        return torch.cat((x, z, u), dim = 1)  


class PnPSolver(PrSolverMixin, CsmriSolverMixin, SpiSolverMixin):
    def __init__(self) -> None:
        super().__init__()
        self.denoiser = UNetDenoiser2D()
        self.iter_num = 6
        
    def __call__(self, task, env_ob, parameters):
        task, _, _ = task.split('_')
        
        if task == 'csmri':
            return self._forward_csmri(env_ob, parameters)
        elif task == 'spi':
            return self._forward_spi(env_ob, parameters)
        elif task == 'pr':
            return self._forward_pr(env_ob, parameters)
        else:
            raise NotImplementedError