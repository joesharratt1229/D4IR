import torch
from torch.utils.data import DataLoader
from scipy.io import loadmat

from contextlib import nullcontext
import os
import logging

from conf import Config
from trainer import TrainingAgent
from rl.critic import ResNet_wobn
from rl.actor import ResNetActorMeta
from rl.encoder import DetermEncoder, StochasticEncoder
from pnp.env import PnpEnv
from pnp.solver import PnPSolver
from data.datasets import TrainingDataset
from experience.replay import Buffer, ElementBuffer

from utils import GaussianModelD, PoissonModel


device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type = device_type, dtype = ptdtype)


DATA_DIR = os.path.join(os.getcwd(), 'data/datadir/')
CSMRI_MASK_DIR = os.path.join(DATA_DIR, 'masks/csmri_masks')
PR_MASK_DIR = os.path.join(DATA_DIR, 'masks/pr_masks')
TRAINING_DIR = os.path.join(DATA_DIR, 'Images_128/')



def load_masks(dir):
    mask_pths = [os.path.join(dir, mask_name) for mask_name in os.listdir(dir)]
    masks = [loadmat(mask_pth)['mask'] for mask_pth in mask_pths]
    return masks


def prepare_dataset(cfg):
    dataset = TrainingDataset(TRAINING_DIR)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, pin_memory = True)
    return dataloader

def create_buffer():
    element_buffer = ElementBuffer()
    buffer = Buffer(element_buffer, device_type)
    return buffer
    

def main():
    cfg = Config()
    
    alphas = [9, 27, 81]
    csmri_masks = load_masks(CSMRI_MASK_DIR)
    pr_masks = load_masks(PR_MASK_DIR)
    
    mri_noise = GaussianModelD(sigmas = [5, 10, 15])
    pr_noise = PoissonModel(alphas = alphas)
    
    stoch_encoder = StochasticEncoder(cfg)
    det_encoder = DetermEncoder(cfg)
    
    explore_buffer = create_buffer()
    exploit_buffer = create_buffer()
    
    
    data_loader = prepare_dataset(cfg)
    solver = PnPSolver()
    env = PnpEnv(data_loader, solver, stoch_encoder, det_encoder, csmri_masks, pr_masks, mri_noise, pr_noise, device_type)
    
    
    explore_policy = ResNetActorMeta(num_inputs=4, action_bundle=5, num_actions=3)
    task_policy = ResNetActorMeta(num_inputs=4, action_bundle=5, num_actions=3)
    
    explore_critic = ResNet_wobn(4, 34, 1)
    task_critic = ResNet_wobn(4, 34, 1)
    
    trainer = TrainingAgent(cfg, stoch_encoder, det_encoder, explore_policy, task_policy, explore_critic, task_critic, data_loader, explore_buffer, exploit_buffer, env)
    trainer.train()
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    



