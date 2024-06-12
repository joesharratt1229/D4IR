import torch
import torch.nn as nn

class StochasticEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.num_tasks = cfg.num_training_tasks
        self.embed_dim = cfg.embed_dim
        self.layers = nn.Embedding(self.num_tasks, self.embed_dim)
        
    def forward(self, task_id):
        out = self.layers(task_id)
        return out
    
    
class DetermEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(4, 8, 5, stride = 4, padding = 0), nn.ReLU(),
                                    nn.Conv2d(8, 16, 5, stride = 2, padding = 0), nn.ReLU(),
                                    nn.Conv2d(16, 16, 4, stride = 1, padding = 0), nn.ReLU(),
                                    nn.Flatten(), nn.Linear(1936, cfg.embed_dim))
        
    def forward(self, obs):
        out = self.layers(obs)
        return out
        
                            