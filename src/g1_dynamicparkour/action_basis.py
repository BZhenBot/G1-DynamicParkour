"""Static invertible action coordinates for bilateral sagittal exploration; no motion timeline/state writes."""
from dataclasses import dataclass
import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction,JointPositionActionCfg


def bilateral_basis(names,device):
    """All 29 coordinates retained; two common leg coordinates plus independent residuals."""
    if len(names)!=29 or len(set(names))!=29:
        raise ValueError("Basis is defined only for actual unique29 G1 actuator-joint names")
    index={name:i for i,name in enumerate(names)}
    s=index["left_knee_joint"]
    toe=index["left_ankle_pitch_joint"]
    matrix=torch.eye(29,device=device)
    # Common bilateral knee coordinate. Hip/ankle residual coordinates still independent.
    for side in ("left","right"):
        matrix[index[f"{side}_hip_pitch_joint"],s]=-1.
        matrix[index[f"{side}_ankle_pitch_joint"],s]=-.5
    matrix[index["right_knee_joint"],s]=1.
    matrix[index["right_ankle_pitch_joint"],toe]=1.
    assert int(torch.linalg.matrix_rank(matrix))==29
    return matrix


@dataclass(kw_only=True)
class BilateralJointPositionActionCfg(JointPositionActionCfg):
    action_basis: str="bilateral_sagittal"

    def build(self,env):
        return BilateralJointPositionAction(self,env)


class BilateralJointPositionAction(JointPositionAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        if cfg.action_basis!="bilateral_sagittal":
            raise ValueError("Unknown static action basis")
        self.basis_matrix=bilateral_basis(self.target_names,self.device)

    def map_actions(self,actions):
        # PPO wrapper clips the latent coordinates too. This additional joint-space
        # clip retains the same validated normalized physical target bounds [-1,1].
        return (actions@self.basis_matrix.T).clamp(-1,1)

    def process_actions(self,actions):
        super().process_actions(self.map_actions(actions))


def matrix(env):
    term=env.action_manager.get_term("joint_pos")
    return getattr(term,"basis_matrix",torch.eye(29,device=env.device))


def map_actions(env,actions):
    term=env.action_manager.get_term("joint_pos")
    return term.map_actions(actions) if hasattr(term,"map_actions") else actions
@dataclass(kw_only=True)
class RecoveryCoordinateActionCfg(JointPositionActionCfg):
    """One auxiliary PPO sampling coordinate; still exactly29 physical G1 joint targets."""
    def build(self,env):
        return RecoveryCoordinateAction(self,env)

class RecoveryCoordinateAction(JointPositionAction):
    def __init__(self,cfg,env):
        # Allocate the original29 physical scale/offset/raw buffers before changing policy dimension.
        super().__init__(cfg,env)
        if self._action_dim!=29:
            raise ValueError('Recovery adapter requires actual29 G1 joints')
        self._action_dim=30
        self.previous_joint_actions=torch.zeros_like(self._raw_actions)
        self.last_recovery_coordinate=torch.zeros((env.num_envs,1),device=self.device)
        self.direction_provider=None
        self.last_consumed_generation=-1

    def map_actions(self,actions):
        if actions.shape[-1]!=30:
            raise ValueError('Expected29 normalized physical coordinates plus one recovery coordinate')
        if self.direction_provider is None:
            if torch.any(actions!=0):
                raise RuntimeError('Recovery coordinate actor has not been bound')
            direction=torch.zeros_like(actions[:,:29])
        else:
            direction,_=self.direction_provider()
            if direction is None or direction.shape!=actions[:,:29].shape:
                raise RuntimeError('Missing or stale current Actor observation direction')
        return (actions[:,:29].clamp(-1,1)+actions[:,29:30].clamp(-1,1)*direction).clamp(-1,1)

    def process_actions(self,actions):
        if self.direction_provider is not None:
            _,generation=self.direction_provider()
            if generation<=self.last_consumed_generation:
                raise RuntimeError('Each physical step requires a fresh Actor observation forward')
            self.last_consumed_generation=generation
        joint=self.map_actions(actions)
        self.previous_joint_actions.copy_(self._raw_actions)
        self.last_recovery_coordinate.copy_(actions[:,29:30].clamp(-1,1))
        super().process_actions(joint)

    def reset(self,env_ids=None):
        super().reset(env_ids)
        ids=slice(None) if env_ids is None else env_ids
        self.previous_joint_actions[ids]=0
        self.last_recovery_coordinate[ids]=0
