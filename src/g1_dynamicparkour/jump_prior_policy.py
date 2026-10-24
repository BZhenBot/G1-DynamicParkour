"""Frozen real jumping PPO mapped to physical joint commands plus trainable direct residual."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.utils import unpad_trajectories
from mjlab.rl.config import RslRlModelCfg
from g1_dynamicparkour.action_basis import bilateral_basis
from g1_dynamicparkour.residual_policy import ResidualStandingMLP

@dataclass
class ResidualJumpModelCfg(RslRlModelCfg):
    class_name: str='g1_dynamicparkour.jump_prior_policy:ResidualJumpMLP'
    prior_checkpoint: str=''
    prior_sha256: str=''
    jump_checkpoint: str=''
    jump_sha256: str=''
    action_names: tuple[str,...]=()
    physical_units: tuple[float,...]=()
    recovery_blend: bool=False

class ResidualJumpMLP(MLPModel):
    def __init__(self,obs,obs_groups,obs_set,output_dim,prior_checkpoint,prior_sha256,jump_checkpoint,jump_sha256,action_names,physical_units,recovery_blend=False,**kwargs):
        params={key:deepcopy(value) for key,value in kwargs.items() if key in inspect.signature(MLPModel.__init__).parameters}
        super().__init__(obs,obs_groups,obs_set,output_dim,**deepcopy(params))
        if obs_set!='actor' or self.obs_dim!=106 or output_dim!=29:
            raise ValueError('Direct residual requires actual actor106/action29')
        path=Path(jump_checkpoint)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=jump_sha256:
            raise ValueError('Frozen jump PPO checkpoint hash mismatch')
        source=torch.load(path,map_location='cpu',weights_only=False)['actor_state_dict']
        self.jump_prior=ResidualStandingMLP(obs,obs_groups,obs_set,output_dim,
            prior_checkpoint,prior_sha256,action_names,physical_units,**deepcopy(params))
        self.jump_prior.load_state_dict(source,strict=True)
        self.jump_prior.requires_grad_(False)
        self.jump_prior.eval()
        basis=bilateral_basis(action_names,'cpu')
        if not torch.allclose(self.jump_prior.basis_inverse,torch.linalg.inv(basis),atol=1e-6,rtol=1e-6):
            raise ValueError('Jump source does not match the verified bilateral basis')
        if not torch.equal(self.jump_prior.physical_units,torch.tensor(physical_units)):
            raise ValueError('Changing physical units is not supported by exact jump transfer')
        self.register_buffer('jump_basis',basis)
        # Preserve parent features/normalization; fresh zero mean head and declared new Gaussian std.
        with torch.no_grad():
            current=self.state_dict()
            for key,value in source.items():
                if key.startswith(('mlp.','obs_normalizer.')):
                    current[key]=value.clone()
            self.load_state_dict(current,strict=True)
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()
        self.recovery_blend_head=torch.nn.Linear(self.mlp[-1].in_features,1) if recovery_blend else None
        if self.recovery_blend_head is not None:
            torch.nn.init.zeros_(self.recovery_blend_head.weight)
            torch.nn.init.zeros_(self.recovery_blend_head.bias)

    def train(self,mode=True):
        super().train(mode)
        if hasattr(self,'jump_prior'):
            self.jump_prior.eval()
        return self

    def prior_joint_commands(self,obs):
        with torch.no_grad():
            latent=self.jump_prior(obs).clamp(-1,1)
            return (latent@self.jump_basis.T).clamp(-1,1)

    def forward(self,obs,masks=None,hidden_state=None,stochastic_output=False):
        obs=unpad_trajectories(obs,masks) if masks is not None else obs
        prior=self.prior_joint_commands(obs)
        latent=self.get_latent(obs,None,hidden_state)
        if self.recovery_blend_head is None:
            mean=prior+self.mlp(latent)
        else:
            features=latent
            for layer in list(self.mlp.children())[:-1]:
                features=layer(features)
            gate=self.recovery_blend_head(features).tanh()
            with torch.no_grad():
                proprio=TensorDict({'actor':obs['actor'][:,:99]},batch_size=obs.batch_size)
                recovery=self.jump_prior.standing_prior(proprio).clamp(-1,1)/self.jump_prior.physical_units
            mean=prior+gate*(recovery-prior)+self.mlp[-1](features)
            self.last_recovery_blend=gate.detach()
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mean)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mean)
        return mean

    def as_jit(self,*args,**kwargs):
        raise NotImplementedError('Combined jumping prior export unverified; use recorded Python inference')

    def as_onnx(self,*args,**kwargs):
        raise NotImplementedError('Combined jumping prior export unverified; use recorded Python inference')