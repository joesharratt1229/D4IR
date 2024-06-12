from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import torch
import numpy as np


class ElementBuffer:
    csmri_5_noise: List = []
    csmri_10_noise: List = []
    spi_4_k: List = []
    spi_8_k: List = []
    pr_27_alpha: List = []
    pr_81_alpha: List = []


class Buffer:
    max_size = 100
    def __init__(self,
                 buffer: ElementBuffer,
                 device: torch.device
                 ) -> None:

        self.element_buffer = buffer
        self.device = device 
    
      
    def store(self, 
              task: str, 
              env_ob: Dict
              ) -> None:
        
        buf_size = len(self.element_buffer[task])
        task_buffer = getattr(self.element_buffer, task)
        
        if buf_size >= self.max_size:
            task_buffer.pop(0)
        
        self.element_buffer[task].append(env_ob)
    
    
    def _sample_batch(self,
                     task: str,
                     env_batch: int
                     ) -> Tuple:
        
        buf_size = len(self.element_buffer[task])
        task_buffer = getattr(self.element_buffer, task)
        
        if buf_size < env_batch:
            return task_buffer
        else:
            indexes = np.random.choice(buf_size, env_batch, replace = False)    
            return task_buffer[indexes]
        
        
    def sample(self, task, env_batch):
        states = self._sample_batch(task, env_batch)
        
        states = torch.from_numpy(states).to(self.device)
        
        for state in states:
            for key, value in state.items():
                state[key] = value.to(self.device)
        
         
        size = states.shape[0]
        if size < env_batch:
            pad_len = env_batch - size
            
            states = torch.cat([states, torch.zeros(([pad_len] + list(states.shape[1:])), dtype = states.dtype)], dim = 0)
            
        
        return states
        
        
    
        
        
            

            
            
            
            
        
        
        
    
    
    

