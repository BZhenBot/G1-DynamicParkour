"""Dedicated landing recovery PPO: frozen genuine standing99 plus direct-joint residual114.

Separate recovery-only curriculum; never claims a full jump from a landing reset.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.utils import unpad_trajectories
from g1_dynamicparkour.residual_policy import ResidualStandingModelCfg
from g1_dynamicparkour.checkpoint_transfer import expand_model_state

@dataclass
class RecoveryStandingModelCfg(ResidualStandingModelCfg):
    class_name: str = 'g1_dynamicparkour.recovery_standing_policy:RecoveryStandingMLP'

class RecoveryStandingMLP(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, prior_checkpoint, prior_sha256,
                 action_names, physical_units, **kwargs):
        params={key:deepcopy(value) for key,value in kwargs.items() if key in inspect.signature(MLPModel.__init__).parameters}
        super().__init__(obs,obs_groups,obs_set,output_dim,**deepcopy(params))
        if obs_set!='actor' or self.obs_dim!=114 or output_dim!=29 or len(action_names)!=29:
            raise ValueError('Dedicated recovery requires actor114 and original29 direct physical joint coordinates')
        path=Path(prior_checkpoint)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=prior_sha256:
            raise ValueError('Frozen real standing PPO hash mismatch')
        source=torch.load(path,map_location='cpu',weights_only=False)['actor_state_dict']
        proprio=TensorDict({'actor':obs['actor'][:,:99]},batch_size=obs.batch_size)
        prior_params={key:value for key,value in params.items() if key!='distribution_cfg'}
        self.standing_prior=MLPModel(proprio,{'actor':['actor']},'actor',29,distribution_cfg=None,**prior_params)
        self.standing_prior.load_state_dict({key:value for key,value in source.items() if not key.startswith('distribution.')},strict=True)
        self.standing_prior.requires_grad_(False)
        self.standing_prior.eval()
        self.register_buffer('physical_units',torch.tensor(physical_units))
        self.initialize_residual(source)

    def initialize_residual(self, source):
        expanded=expand_model_state(source,114)
        expanded['mlp.4.weight'].zero_()
        expanded['mlp.4.bias'].zero_()
        state=self.state_dict()
        for key,value in expanded.items():
            state[key]=value
        self.load_state_dict(state,strict=True)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self,'standing_prior'):
            self.standing_prior.eval()
        return self

    def prior_joint_commands(self, obs):
        with torch.no_grad():
            proprio=TensorDict({'actor':obs['actor'][:,:99]},batch_size=obs.batch_size)
            return self.standing_prior(proprio).clamp(-1,1)/self.physical_units

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        obs=unpad_trajectories(obs,masks) if masks is not None else obs
        mean=self.prior_joint_commands(obs)+self.mlp(self.get_latent(obs,None,hidden_state))
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mean)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mean)
        return mean

    def as_jit(self,*args,**kwargs):
        raise NotImplementedError('Combined standing prior and residual export unverified; Python inference only')

    def as_onnx(self,*args,**kwargs):
        raise NotImplementedError('Combined standing prior and residual export unverified; Python inference only')
