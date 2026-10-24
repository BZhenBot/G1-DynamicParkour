"""Two genuine frozen PPO controllers at their declared rates; causal current geometry switch.

Jump25Hz actual29 commands held for two50Hz manager steps; recovery50Hz direct29.
No privileged contact identity/FSM/verdict and no future platform trajectory.
"""
import math
import torch
from tensordict import TensorDict

class DualRateJumpRecovery:
    def __init__(self, jump_policy, jump_actor, recovery_actor, geometry, num_envs, device,
                 control_dt=.02, jump_period=.04, maximum_tilt_deg=45., minimum_clock=.8):
        ratio=round(jump_period/control_dt)
        if ratio<1 or abs(ratio*control_dt-jump_period)>1e-8:
            raise ValueError('Jump policy period must be an integer multiple of the physical manager control period')
        self.jump_policy=jump_policy
        self.jump_actor=jump_actor
        self.recovery_actor=recovery_actor
        self.ratio=ratio
        self.bounds=torch.tensor([geometry.target_half_length-.02,geometry.target_half_width-.02],device=device)
        self.maximum_tilt_deg=maximum_tilt_deg
        self.minimum_clock=minimum_clock
        self.active=torch.zeros(num_envs,device=device,dtype=torch.bool)
        self.steps=torch.zeros(num_envs,device=device,dtype=torch.long)
        self.held_joints=torch.zeros(num_envs,29,device=device)
        self.last_ready=torch.zeros_like(self.active)
        self.last_new_handoff=torch.zeros_like(self.active)
        self.last_source_updated=torch.zeros_like(self.active)
        self.alpha=torch.zeros(num_envs,device=device)
        self.last_public_observation=None

    def reset(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        for tensor in (self.active,self.steps,self.held_joints,self.last_ready,self.last_new_handoff,self.last_source_updated,self.alpha):
            tensor[ids]=0

    def ready(self,actor114,relative_feet6):
        if actor114.ndim!=2 or actor114.shape[1]!=114 or relative_feet6.shape!=(actor114.shape[0],6):
            raise ValueError('Explicit current120 observation layout required')
        feet=relative_feet6.reshape(-1,2,3)
        return (torch.isfinite(actor114).all(-1) & torch.isfinite(relative_feet6).all(-1)
            & actor114[:,106:108].gt(.5).all(-1)
            & (feet[:,:,:2].abs()<=self.bounds).all(-1).all(-1)
            & ((feet[:,:,2]>=-.01)&(feet[:,:,2]<=.04)).all(-1)
            & (-actor114[:,8]>=math.cos(math.radians(self.maximum_tilt_deg)))
            & (-actor114[:,101]>=.5) & (actor114[:,105]>=self.minimum_clock))

    def __call__(self,obs):
        only_actor=TensorDict({'actor':obs['actor']},batch_size=obs.batch_size)
        self.last_public_observation=torch.cat((obs['actor'],obs['supervisor_current']),-1).detach().clone()
        self.last_ready=self.ready(obs['actor'],obs['supervisor_current'])
        self.last_new_handoff=self.last_ready & ~self.active
        self.active |= self.last_ready
        self.alpha=self.active.float()
        # Fresh source forward always binds the action adapter to the current
        # observation. Its direction is irrelevant when coordinate30 is zero.
        source=self.jump_policy(only_actor)
        mapped=(source[:,:29].clamp(-1,1)+source[:,29:30].clamp(-1,1)*self.jump_actor.last_direction).clamp(-1,1)
        self.last_source_updated=(self.steps%self.ratio==0)&~self.active
        self.held_joints[self.last_source_updated]=mapped[self.last_source_updated]
        result=torch.zeros_like(source)
        result[:,:29]=self.held_joints
        if self.active.any():
            recovering=self.recovery_actor(only_actor).clamp(-1,1)
            result[self.active,:29]=recovering[self.active]
        self.steps+=1
        return result
