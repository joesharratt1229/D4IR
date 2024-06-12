import os

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

import numpy as np

import random

from pnp.solver import PnPSolver
import utils as utils


class CsmriEnvMixin:
    #May omit some of these -> can form a task 
    def build_csmri_observation(self, target, noise_val):
        acc = random.choice(self.csmri_mask)
        acc = torch.from_numpy(acc.astype(np.bool))
        y0 = utils.fft(target)
        
        y0 = self.noise_model(y0, noise_val)
        y0[:, ~acc, :] = 0
        Aty0 = utils.ifft(y0)
        x0 = Aty0.clone().detach()
        output = Aty0.clone().detach().real
        dic = {'y0': y0, 
               'x0': x0, 
               'ATy0': Aty0, 
               'gt': target, 
               'output': output, 
               'mask': acc}
        return dic
    
    
class CTEnvMixin:
    def build_ct_observation(self, target):
        noise_lev = random.choice(self.noise_level)
        resolution = target.shape[-1]
        radon = self.radon_generator(resolution, self.view)
        y0 = radon.forward(target)
        y0 = self.noise_model(y0, noise_lev)
        
        ATy0 = radon.backprojection_norm(y0)
        x0 = radon.filter_backprojection(y0)
        output = ATy0.clone().detach()
        view = torch.ones_like(target) * view / 120
        T = torch.zeros_like(x0)
        
        dic = {'y0': y0, 'ATy0': ATy0, 'output': output, 'x0': x0, 'gt': target, 'view': view, 'T': T}
        return dic
    
    
class SPIEnvMixin:
    def build_spi_observation(self, target, K):
        with torch.no_grad():
            y0 = utils.spi_forward(target, K, K**2, 1)
            x0 = F.avg_pool2d(y0, K)      
        x0 = x0.clone().detach()
        K = torch.ones_like(target) * K / 10
        dic = {'x0': x0, 'gt': target, 'K': K, 'y0': y0, 'output': x0.clone().detach()}
        return dic
            

class PREnvMixin:
    def build_pr_observation(self, target, alpha_val):
        mask = random.choice(self.pr_mask)
        C = mask.shape[0]
        mask = torch.from_numpy(mask).reshape(1, C, 128, 128)

        y0 = utils.cdp_forward(torch.complex(target, torch.zeros_like(target)),
                                mask).abs()[0]
        y0 = self.noise_model(y0, alpha_val)
        x0 = torch.ones_like(target)
        #sigma_n = x0 * noise_lev
        dic = {'y0': y0, 'x0': x0, 'output': x0, 'gt': target, 'mask': mask}
        return dic
         
         
         
class PnpEnv(CsmriEnvMixin, CTEnvMixin, SPIEnvMixin, PREnvMixin):
    tasks = {
        'csmri_5_noise': 0,
        'csmri_10_noise': 1,
        'spi_4_k': 2,
        'spi_8_k': 3,
        'pr_27_alpha': 4,
        'pr_81_alpha': 5
    }
    
    csmri_mask = None
    noise_level = [5, 10, 15]

    
    def __init__(self, 
                 dataloader: DataLoader,
                 solver: PnPSolver,
                 stoch_encoder,
                 determ_encoder,
                 mri_masks,
                 pr_masks,
                 mri_noise_model,
                 pr_noise_model,
                 #view: float,
                 device: torch.device) -> None:
        
        self.dataloader = iter(dataloader)
        self.solver = solver
        self.device = device
        #self.radon_generator = utils.RadonGenerator()
        self.stoch_encoder = stoch_encoder
        self.det_encoder = determ_encoder
        #self.view = view
        self.max_episode_step = 6
        self.cur_episode_step = 0
        self.csmri_mask = mri_masks
        self.pr_masks = pr_masks
        self.mri_noise = mri_noise_model
        self.pr_noise_model = pr_noise_model
        
    
    def _build_init_ob(self, task, ob):
        task, value, _ = task.split('_')
        if task == 'csmri':
            return self.build_csmri_observation(ob, float(value))
        elif task == 'spi':
            return self.build_pr_observation(ob, float(value))
        elif task == 'pr':
            return self.build_spi_observation(ob, float(value))
        else:
            raise NotImplementedError
        
            
    def sample_task(self):
        return random.choice(list(self.tasks.items()))
        
        
    def _build_init_env_ob(self, data):
        x = data['output'].to(self.device)
        z = x.clone().detach()
        u = torch.zeros_like(x)
        data['T'] = torch.zeros_like(x)
        data['output'] = torch.cat([x, z, u], dim = 1)
        return data
        
    def build_policy_ob(self, data):
        state = data['output']
        B, _, W, H = state.shape
        T = data['T'] + self.cur_episode_step/self.max_episode_step
        return torch.cat([state, T], dim = 1).to(self.device)
    
    
    def build_traj_ob(self, data):
        return data['x'].to(self.device)
    
    @staticmethod
    def _compute_exploit_reward(output, gt):
        N = output.shape[0]
        output = torch.clamp(output, 0, 1)
        mse = torch.mean(F.mse_loss(output.view(N, -1), gt.view(N, -1), reduction='none'), dim=1)
        psnr = 10 * torch.log10((1 ** 2) / mse)
        return psnr.unsqueeze(1)
        
        
    def reset(self, task, data = None):
        """
        reset environment with appropriate
        task based on configuration. If data is None
        obtain from data loader else from data
        """
        
        #Need some type of temporary list to know when done and state
        if data is None:
            try:
                data = next(self.dataloader)
            except StopIteration:
                data = iter(self.dataloader)
            
            env_ob = self._build_init_ob(task, data)
            env_ob = self._build_init_env_ob(env_ob)
            policy_ob = self.build_policy_ob(env_ob)
            traj_ob = self.build_traj_ob(env_ob)
            return env_ob, policy_ob, traj_ob
        
        #env_ob = self._build_init_env_ob(env_ob)
        traj_ob = self.build_traj_ob(data)
        policy_ob = self.build_policy_ob(data)
        return data, policy_ob, traj_ob    
    
    
    def _compute_reward(self, next_state, env_ob, z, is_explore):
        #### at thsi point decided to only use deterministic encoder to calculate
        if is_explore:
            prev_traj = env_ob['prev_output']
            g_w1 = self.det_encoder(prev_traj)
            
            #f_v = self.stoch_encoder(task_id)
            g_w = self.det_encoder(z)
            reward = g_w1 - g_w
            return reward
        else:
            gt = env_ob['gt']
            x, _, _ = torch.chunk(next_state, chunks = 3, dim = 1)
            return self._compute_exploit_reward(x, gt)
    
    def build_env_ob(self, env_ob, next_state):
        env_ob['prev_output'] = env_ob['output']
        env_ob['output'] = next_state.clone().detach()
        env_ob['T'] = env_ob['T'] * self.cur_step/self.max_episode_step
        return env_ob
    
    
    def step(self, task, env_ob, parameters):
        """ 
        Take a step based on task using forward 
        method of solver and compute reward 
        based on whether reward is explore or exploit
        store in replay buffer
        """

        next_state = self.solver(task, env_ob, parameters)
        env_ob = self.build_env_ob(env_ob, next_state)
        traj_ob = self.build_traj_ob(env_ob)
        
        #reward = self._compute_reward(env_ob, traj_ob, is_explore)
        policy_ob = self.build_policy_ob(env_ob)
        done = parameters['idx_stop']
        return policy_ob, env_ob, traj_ob, next_state, done
        
           
    def forward(self, task, env_ob, z, action, explore = False):
        next_state = self.solver(task, env_ob, action)
        reward = self._compute_reward(next_state, env_ob, z, explore)
        return reward, next_state, action['idx_stop']
        
        
           
    
    