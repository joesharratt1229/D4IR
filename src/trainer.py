import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

from conf import Config
from rl.actor import ResNetActorMeta
from rl.critic import ResNet_wobn
from experience.replay import Buffer
from pnp.env import PnpEnv



def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

class TrainingAgent:
    def __init__(self,
                 cfg: Config,
                 stoc_encoder: nn.Module,
                 det_encoder: nn.Module,
                 actor_exp: ResNetActorMeta, 
                 actor_task: ResNetActorMeta,
                 critic_exp: ResNet_wobn,
                 critic_task: ResNet_wobn,
                 train_loader: DataLoader,
                 expl_buffer: Buffer,
                 task_buffer: Buffer,
                 env: PnpEnv):
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cfg = cfg
        self.stoc_encoder = stoc_encoder.to(self.device)
        self.det_encoder = det_encoder.to(self.device)
        self.actor_exp = actor_exp.to(self.device)
        self.critic_exp = critic_exp.to(self.device)
        self.actor_task = actor_task.to(self.device)
        self.critic_task = critic_task.to(self.device)
        self.explore_buffer = expl_buffer
        self.exploit_buffer = task_buffer
        self.env = env
        self.data_loader = train_loader
        self.batch_size = 12
        self.decoder_loss = nn.MSELoss()
        self.criterion = nn.MSELoss()
        self._create_optimzers()
        
        
    def _create_optimzers(self):
        self.actor_exp_optim = torch.optim.Adam(self.actor_exp.parameters(), lr = self.cfg.actor_lr)
        self.actor_task_optim = torch.optim.Adam(self.actor_task.parameters(), lr = self.cfg.actor_lr)
        self.critic_exp_optim = torch.optim.Adam(self.critic_exp.parameters(), lr = self.cfg.critic_lr)
        self.critic_task_optim = torch.optim.Adam(self.critic_task.parameters(), lr = self.cfg.critic_lr)
        self.stoc_enc_optim = torch.optim.Adam(self.stoc_encoder.parameters(), lr = self.cfg.stc_lr)
        self.det_enc_optim = torch.optim.Adam(self.det_encoder.parameters(), lr = self.cfg.det_lr)
        
        
    def _roll_exp_traj(self, task_id):
        """
        Executes pnp algorithm for a given 
        task -> samples data, action and value (maximum of
        six iterations). Store in replay buffer
        """
        is_explore = True
        data = self.env.reset(task_id)
        policy_obs, env_obs, _ = data
        
        for _ in self.cfg.max_episodes:
            ##take an action
            with torch.no_grad():
                actions, _, _ = self.actor_exp(policy_obs, None, True)
            policy_obs, env_obs, _, _, done= self.env.step(task_id, env_obs, actions, is_explore)
            self.save_experience(task_id, env_obs)
            
            if (done == 1).all().item():
                return
            
        return
    
    
    def _roll_task_traj(self, task, task_id, trial):
        """
        Compute embedding, roll out exploitation policy and store resulting
        experience in task replay buffer
        """
        #is_explore = False
        policy_obs, env_obs, traj_ob = self.env.reset(task)
        
        for _ in self.cfg.max_episodes:
            if trial%2 == 0:
                z  = self.stoc_encoder(task_id)
                #z = sample_gaussian(sample_mean, self.cfg.var)
            else:
                z = self.det_encoder(traj_ob)
                
            policy_obs = torch.cat([policy_obs, env_obs], dim = 1)
            
            with torch.no_grad():   
                actions, _, _= self.actor_task(policy_obs, None, True, z)
            policy_obs, env_obs, traj_ob, done = self.env.step(task_id, env_obs, actions)
            self.save_experience(task_id, env_obs)
            
            if (done == 1).all().item():
                return
    
    
    def save_experience(self, task_id, env_obs, is_explore):
        for k, v in env_obs.items():
            if isinstance(v, torch.Tensor):
                env_obs[k] = env_obs[k].clone().detach().cpu()
        
        for i in range(self.batch_size):
            temp_dict = {key: value[i] for key, value in env_obs.items()}
            
            
            if is_explore:
                self.explore_buffer.store(task_id, temp_dict)
            else:
                self.exploit_buffer.store(task_id, temp_dict)

            
    def _update_exploration_pol(self, task, critic_target):
        """
        run policy for six iterations and obtain resulting state. observe reward at 
        each step and compute loxx of policy -> apply slower updates to targets
        """
        
        for episode in self.cfg.episode_train_times:
            expl_experience = self.explore_buffer.sample(task, self.batch_size)
            obs = self.env.reset(task, expl_experience)
            policy_obs, env_obs, traj_ob = obs
            
            action, logprob, dist_entropy = self.actor_exp(policy_obs, None, True)
            reward, obs2 = self.env.forward(task, env_obs, action, explore = False)
            
            eval_ob = self.env.get_eval_ob(obs)
            #get last state
            eval_ob2 = self.env.get_eval_ob(obs2)
            
            V_cur = self.critic_exp(eval_ob)
            
            with torch.no_grad():
                V_next_target = critic_target(eval_ob2)
                V_next_target = (
                self.opt.discount * (1 - action['idx_stop'].float())).unsqueeze(-1) * V_next_target
                #compute q value as immediate reward plus sum of future rewards
                Q_target = V_next_target + reward
                
            advantage = (Q_target - V_cur).clone().detach()
            #discrete computation
            a2c_loss = logprob * advantage

            # compute ddpg loss for continuous actions 
            V_next = self.critic_exp(eval_ob2)
            V_next = (self.opt.discount * (1 - action['idx_stop'].float())).unsqueeze(-1) * V_next
            #compute dpg loss _> continous denoising strength and penalty parameters
            ddpg_loss = V_next + reward

            # compute entroy regularization
            entroy_regularization = dist_entropy

            #compute policy and value network's losss
            policy_loss = - (a2c_loss + ddpg_loss + self.opt.lambda_e * entroy_regularization).mean()
            value_loss = self.criterion(Q_target, V_cur)

            # zero out gradients to prevent them accumulating
            self.actor_exp.zero_grad()
            
            # computes gradient of the loss for the policy network, creates computational graph 
            policy_loss.backward(retain_graph=True)

            
            #clips graidents to prevent from becoming too large
            actor_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 50)
            #update step of the actor network
            self.actor_exp_optim.step()

            #same for the critic network
            self.critic_exp.zero_grad()
            value_loss.backward(retain_graph=True)
            critic_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), 50)
            self.critic_exp_optim.step()

            # soft update target network -> exponentially weighted moving average of previous policies
            soft_update(self.critic_target, self.critic_exp, self.opt.tau)
            
            #self._compute_decoder_loss(task, traj_ob)
            
            wandb.log({'Policy Explore Loss Loss': -policy_loss.item(), 
                       'Value Explore Loss': value_loss.item(), 
                       'Entropy explore regularisation': entroy_regularization.mean().item(), 
                       'Actor explore norm': actor_norm.item(), 
                       'Critic explore norm': critic_norm.item()})
            
            env_obs = self.env.build_env_ob(env_obs, obs2)
            policy_obs = self.env.build_policy_ob(env_obs)
            #traj_ob = self.env.build_traj_ob(env_obs)


    def _compute_decoder_loss(self, task, traj):
         stoc_output = self.stoc_encoder(task) 
         #stoc_output = sample_gaussian(sample_mean, self.cfg.var)
         determ_output = self.det_encoder(traj)
         stoc_output.detach()
         loss = self.decoder_loss(stoc_output, determ_output)
         
         self.det_encoder.zero_grad()
         loss.backward()
         self.det_enc_optim.step()
    
    
    
    def _update_task_pol(self, task, task_id, critic_target, trial):
        """
        Run policy for six iterations compute loss 
        of task policy, optimize with stochastic encoder
        on trial mod 0 else deterministic -> optimize stochastic
        or deterministic dependencing on mod -> apply slower update to targets
        """
        
        
        for _ in self.cfg.episode_train_times:
            task_experience = self.exploit_buffer.sample(task, self.batch_size)
            obs = self.env.reset(task, task_experience)
            policy_obs, env_obs, traj_ob = obs
            
            if trial%2 == 0:
                z = self.stoc_encoder(task_id)
                #z = sample_gaussian(sample_mean, self.cfg.var)
            else:
                z = self.det_encoder(traj_ob)
                
            policy_obs = torch.cat([policy_obs, env_obs], dim = 1)
                
            action, logprob, dist_entropy = self.actor_task(policy_obs, None, True, z)
            reward, obs2 = self.env.forward(task, env_obs, action, explore = False)
            reward -= self.opt.loop_penalty
            
            eval_ob = self.env.get_eval_ob(obs)
            #get last state
            eval_ob2 = self.env.get_eval_ob(obs2)
            
            V_cur = self.critic_task(eval_ob)
            
            with torch.no_grad():
                V_next_target = critic_target(eval_ob2)
                V_next_target = (
                self.opt.discount * (1 - action['idx_stop'].float())).unsqueeze(-1) * V_next_target
                #compute q value as immediate reward plus sum of future rewards
                Q_target = V_next_target + reward
                
        
            advantage = (Q_target - V_cur).clone().detach()
            #discrete computation
            a2c_loss = logprob * advantage

            # compute ddpg loss for continuous actions 
            V_next = self.critic_task(eval_ob2)
            V_next = (self.opt.discount * (1 - action['idx_stop'].float())).unsqueeze(-1) * V_next
            #compute dpg loss _> continous denoising strength and penalty parameters
            ddpg_loss = V_next + reward

            # compute entroy regularization
            entroy_regularization = dist_entropy

            #compute policy and value network's losss
            policy_loss = - (a2c_loss + ddpg_loss + self.opt.lambda_e * entroy_regularization).mean()
            value_loss = self.criterion(Q_target, V_cur)

            # zero out gradients to prevent them accumulating
            self.actor_task.zero_grad()
            
            if trial%2 == 0:
                self.stoc_encoder.zero_grad()
            else:
                self.det_encoder.zero_grad()
            
            # computes gradient of the loss for the policy network, creates computational graph 
            policy_loss.backward(retain_graph=True)

            
            #clips graidents to prevent from becoming too large
            actor_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 50)
            #update step of the actor network
            self.actor_exp_optim.step()

            if trial%2 == 0:
                self.stoc_enc_optim.step()
            else:
                self.det_enc_optim.step()
                
            
            self.critic.zero_grad()
            value_loss.backward(retain_graph=True)
            critic_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), 50)
            self.critic_exp_optim.step()          

            # soft update target network -> exponentially weighted moving average of previous policies
            soft_update(self.critic_target, self.critic_task, self.opt.tau)
            
            ## update weights of stochastic and deterministic encoders
            #### TODO stochastic first
            
            self._compute_decoder_loss(task_id, traj_ob)
            ###store somewhere
            wandb.log({'Policy Task Loss': -policy_loss.item(), 
                        'Critic task Loss': value_loss.item(), 
                        'Entropy task regularisation': entroy_regularization.mean().item(), 
                        'Actor task norm': actor_norm.item(), 
                        'Critic task norm': critic_norm.item()})
            
            
            env_obs = self.env.build_env_ob(env_obs, obs2)
            policy_obs = self.env.build_policy_ob(env_obs)
            traj_ob = self.env.build_traj_ob(env_obs)
            
    
    def save_to_checkpoint(self, model, model_name):
        ckpt = model.state_dict()
        PATH = f"checkpoints/{model_name}.pt"
        torch.save(ckpt, PATH)  
            
    #TODO -> how to handle batch input during training for explore and exploit      
    def train(self):
        critic_target_explore = copy.deepcopy(self.critic_exp)
        critic_task_task = copy.deepcopy(self.actor_exp)
        
        wandb.login(key='d26ee755e0ba08a9aff87c98d0cedbe8b060484b')
        wandb.init(project='MetaRL for inverse imaging', entity='joesharratt1229')
        wandb.watch(self.actor_exp)
        wandb.watch(self.actor_task)
        wandb.watch(self.critic_exp)
        wandb.watch(self.critic_task)
        
        task, task_id = self.env.sample_task()
        # TODO deepcopy of policy and value to create target networks
        
        for trial in self.cfg.num_trials:
            if trial%10 == 0:
                self.save_to_checkpoint(self.actor_exp, 'policy_exp_model')  
                self.save_to_checkpoint(self.actor_task, 'policy_task_model')   
                self.save_to_checkpoint(self.critic_exp, 'critic_exp_model')  
                self.save_to_checkpoint(self.critic_task, 'critic_task_model')                   
            
            
            self._roll_exp_traj(self, task)
            self._roll_task_traj(self, task, task_id, trial)
            
            if trial > self.cfg.warmup:
                env_task, task_id = self.env.sample_task()
                self._update_exploration_pol(env_task, critic_target_explore)
                self._update_task_pol(env_task, task_id, critic_task_task, trial)
                
            task, task_id = self.env.sample_task()
                
                
                
                
                
                
                    
                    
                
            
            
            
            
            
            
        
        