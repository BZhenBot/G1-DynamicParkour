"""PPO samples 29 joint coordinates plus one current-observation recovery direction coordinate."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.utils import unpad_trajectories
from g1_dynamicparkour.jump_prior_policy import ResidualJumpModelCfg
from g1_dynamicparkour.residual_policy import ResidualStandingMLP
from g1_dynamicparkour.action_basis import bilateral_basis
from g1_dynamicparkour.checkpoint_transfer import expand_model_state

@dataclass
class RecoveryCoordinateModelCfg(ResidualJumpModelCfg):
    class_name: str = 'g1_dynamicparkour.recovery_coordinate:RecoveryCoordinateMLP'
    contact_extension: bool = False
    bounded_coordinate_mean: bool = False

class RecoveryCoordinateMLP(MLPModel):
    def __init__(self,obs,obs_groups,obs_set,output_dim,prior_checkpoint,prior_sha256,
                 jump_checkpoint,jump_sha256,action_names,physical_units,recovery_blend=False,contact_extension=False,bounded_coordinate_mean=False,**kwargs):
        params={key:deepcopy(value) for key,value in kwargs.items()
                if key in inspect.signature(MLPModel.__init__).parameters}
        super().__init__(obs,obs_groups,obs_set,output_dim,**deepcopy(params))
        self.contact_extension=contact_extension
        self.bounded_coordinate_mean=bounded_coordinate_mean
        if obs_set!='actor' or self.obs_dim!=(114 if contact_extension else 106) or output_dim!=30 or recovery_blend:
            raise ValueError('Recovery coordinate requires actor106, sampled action30 and no mean-only blend')
        path=Path(jump_checkpoint)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=jump_sha256:
            raise ValueError('Frozen jump source hash mismatch')
        source=torch.load(path,map_location='cpu',weights_only=False)['actor_state_dict']
        self.jump_prior=ResidualStandingMLP(self.prior_observations(obs),obs_groups,obs_set,29,
            prior_checkpoint,prior_sha256,action_names,physical_units,**deepcopy(params))
        self.jump_prior.load_state_dict(source,strict=True)
        self.jump_prior.requires_grad_(False)
        self.jump_prior.eval()
        basis=bilateral_basis(action_names,'cpu')
        if not torch.allclose(self.jump_prior.basis_inverse,torch.linalg.inv(basis),atol=1e-6,rtol=1e-6):
            raise ValueError('Frozen source action basis mismatch')
        if not torch.equal(self.jump_prior.physical_units,torch.tensor(physical_units)):
            raise ValueError('Frozen source physical units mismatch')
        self.register_buffer('jump_basis',basis)
        # Reuse actual hidden features and normalization. Output30 is fresh and zero.
        with torch.no_grad():
            state=self.state_dict()
            expanded=expand_model_state(source,self.obs_dim)
            for key,value in expanded.items():
                if key.startswith(('mlp.','obs_normalizer.')) and key in state and state[key].shape==value.shape:
                    state[key]=value.clone()
            self.load_state_dict(state,strict=True)
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()
        self.direction_generation=0
        self.last_direction=None

    def train(self,mode=True):
        super().train(mode)
        if hasattr(self,'jump_prior'):
            self.jump_prior.eval()
        return self

    def prior_observations(self,obs):
        # Original frozen jump observes its unchanged106 prefix; standing prior keeps99.
        return TensorDict({'actor':obs['actor'][:,:106]},batch_size=obs.batch_size)

    def prior_joint_commands(self,obs):
        with torch.no_grad():
            return (self.jump_prior(self.prior_observations(obs)).clamp(-1,1)@self.jump_basis.T).clamp(-1,1)

    def forward(self,obs,masks=None,hidden_state=None,stochastic_output=False):
        obs=unpad_trajectories(obs,masks) if masks is not None else obs
        prior=self.prior_joint_commands(obs)
        with torch.no_grad():
            proprio=TensorDict({'actor':obs['actor'][:,:99]},batch_size=obs.batch_size)
            standing=self.jump_prior.standing_prior(proprio).clamp(-1,1)/self.jump_prior.physical_units
            self.last_direction=(standing-prior).detach()
            self.direction_generation+=1
        head=self.mlp(self.get_latent(obs,None,hidden_state))
        coordinate_mean=head[:,29:30].tanh() if self.bounded_coordinate_mean else head[:,29:30]
        # Only Gaussian MEAN is bounded; samples/log-probabilities remain latent30 Gaussian.
        # This is not a tanh-transformed action distribution and needs no Jacobian correction.
        mean=torch.cat((prior+head[:,:29],coordinate_mean),dim=-1)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mean)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mean)
        return mean

    def as_jit(self,*args,**kwargs):
        raise NotImplementedError('Current-observation action adapter must be exported with actor; Python path verified only')

    def as_onnx(self,*args,**kwargs):
        raise NotImplementedError('Current-observation action adapter must be exported with actor; Python path verified only')

def bind_recovery_coordinate(env,actor):
    term=env.action_manager.get_term('joint_pos')
    if hasattr(term,'direction_provider'):
        if not isinstance(actor,RecoveryCoordinateMLP):
            raise TypeError('Recovery coordinate action term requires its matched actor')
        term.direction_provider=lambda:(actor.last_direction,actor.direction_generation)

def load_contact_expansion(runner,path):
    actor=runner.alg.actor
    if not isinstance(actor,RecoveryCoordinateMLP) or not actor.contact_extension or actor.obs_dim!=114:
        raise ValueError('Contact expansion requires the explicit114 observation actor')
    source=torch.load(path,map_location='cpu',weights_only=False)
    state=source['actor_state_dict']
    if state['mlp.0.weight'].shape[1]!=106 or state['mlp.4.weight'].shape[0]!=30:
        raise ValueError('Contact transfer requires a trained106/30 recovery-coordinate source')
    actor.load_state_dict(expand_model_state(state,114),strict=True)
    runner.alg.critic.load_state_dict(source['critic_state_dict'],strict=True)
    if actor.jump_prior.obs_dim!=106 or actor.jump_prior.standing_prior.obs_dim!=99:
        raise AssertionError('Frozen priors must not receive extra contact information')
    if torch.count_nonzero(actor.mlp[0].weight[:,106:]):
        raise AssertionError('New contact columns must initialize exactly zero')
    return {'actor':'unchanged trained106 prefix +8 current unlabelled foot-contact channels with zero incoming weights',
            'critic':'unchanged130 weights and normalization','optimizer':'fresh for changed actor input shape',
            'distribution_std':'copied without reset','iteration':'fresh0','source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def load_bounded_coordinate_transfer(runner, path):
    actor=runner.alg.actor
    if not isinstance(actor,RecoveryCoordinateMLP) or not actor.bounded_coordinate_mean:
        raise ValueError('Explicit bounded-coordinate model is required for transfer')
    source=torch.load(path,map_location='cpu',weights_only=False)
    if source['actor_state_dict']['mlp.0.weight'].shape[1]!=actor.obs_dim or source['actor_state_dict']['mlp.4.weight'].shape[0]!=30:
        raise ValueError('Bounded mean transfer preserves trained observation/action dimensions')
    actor.load_state_dict(source['actor_state_dict'],strict=True)
    runner.alg.critic.load_state_dict(source['critic_state_dict'],strict=True)
    return {'actor':'all trained weights and normalization copied; only auxiliary coordinate Gaussian mean becomes tanh(raw_head)',
            'critic':'all trained weights and normalization copied','distribution_std':'copied unchanged',
            'optimizer':'fresh for explicitly changed mean parameterization','iteration':'fresh0',
            'probability_space':'unmodified latent30 Gaussian, physical29 commands remain clipped action-adapter outputs',
            'source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
