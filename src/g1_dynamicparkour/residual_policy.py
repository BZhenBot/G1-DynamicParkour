"""Frozen real standing PPO prior plus trainable current-observation residual; no scripted trajectory."""
from dataclasses import dataclass
import hashlib,inspect
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.utils import unpad_trajectories
from mjlab.rl.config import RslRlModelCfg
from g1_dynamicparkour.action_basis import bilateral_basis
from g1_dynamicparkour.checkpoint_transfer import expand_model_state

@dataclass
class ResidualStandingModelCfg(RslRlModelCfg):
    class_name: str="g1_dynamicparkour.residual_policy:ResidualStandingMLP"
    prior_checkpoint: str=""
    prior_sha256: str=""
    action_names: tuple[str,...]=()
    physical_units: tuple[float,...]=()

class ResidualStandingMLP(MLPModel):
    def __init__(self,obs,obs_groups,obs_set,output_dim,prior_checkpoint,prior_sha256,action_names,physical_units,**kwargs):
        params={k:v for k,v in kwargs.items() if k in inspect.signature(MLPModel.__init__).parameters}
        super().__init__(obs,obs_groups,obs_set,output_dim,**params)
        if obs_set!="actor" or self.obs_dim!=106 or output_dim!=29:
            raise ValueError("Residual standing actor requires actual106/29 interface")
        path=Path(prior_checkpoint)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=prior_sha256:
            raise ValueError("Frozen standing PPO source hash mismatch")
        source=torch.load(path,map_location="cpu",weights_only=False)["actor_state_dict"]
        prior_obs=TensorDict({"actor":obs["actor"][:,:99]},batch_size=obs.batch_size)
        prior_params={k:v for k,v in params.items() if k!="distribution_cfg"}
        self.standing_prior=MLPModel(prior_obs,{"actor":["actor"]},"actor",29,distribution_cfg=None,**prior_params)
        self.standing_prior.load_state_dict({k:v for k,v in source.items() if not k.startswith("distribution.")},strict=True)
        self.standing_prior.requires_grad_(False)
        self.register_buffer("basis_inverse",torch.linalg.inv(bilateral_basis(action_names,"cpu")))
        self.register_buffer("physical_units",torch.tensor(physical_units))
        self.initialize_residual(source)

    def initialize_residual(self,source):
        expanded=expand_model_state(source,106)
        # Reuse feature extraction/normalization, initialize residual mean exactly zero.
        expanded["mlp.4.weight"].zero_()
        expanded["mlp.4.bias"].zero_()
        current=self.state_dict()
        for key,value in expanded.items():
            current[key]=value
        self.load_state_dict(current,strict=True)

    def prior_coordinates(self,obs):
        prior_obs=TensorDict({"actor":obs["actor"][:,:99]},batch_size=obs.batch_size)
        with torch.no_grad():
            physical=self.standing_prior(prior_obs).clamp(-1,1)/self.physical_units
            return physical@self.basis_inverse.T

    def forward(self,obs,masks=None,hidden_state=None,stochastic_output=False):
        obs=unpad_trajectories(obs,masks) if masks is not None else obs
        mean=self.prior_coordinates(obs)+self.mlp(self.get_latent(obs,None,hidden_state))
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mean)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mean)
        return mean

    def as_jit(self,*args,**kwargs):
        raise NotImplementedError("Residual prior export requires a combined graph; use validated Python inference")

    def as_onnx(self,*args,**kwargs):
        raise NotImplementedError("Residual prior export requires a combined graph; use validated Python inference")

def configure_runner_model(agent,env):
    from g1_dynamicparkour.jump_prior_policy import ResidualJumpModelCfg
    if isinstance(agent.actor,(ResidualStandingModelCfg,ResidualJumpModelCfg)):
        from g1_dynamicparkour.jump import action_units
        agent.actor.action_names=tuple(env.action_manager.get_term("joint_pos").target_names)
        agent.actor.physical_units=tuple(action_units(env)[0].cpu().tolist())