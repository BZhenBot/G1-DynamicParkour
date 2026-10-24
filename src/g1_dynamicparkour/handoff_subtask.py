"""Training-only jump-to-public-handoff subtask; never claims complete stable landing."""
from types import SimpleNamespace
import torch
from g1_dynamicparkour import jump
from g1_dynamicparkour.public_handoff import current_relative_feet
from g1_dynamicparkour.composed_controller import DualRateJumpRecovery

def public_ready(actor114,feet6,geometry):
    observer=SimpleNamespace(bounds=torch.tensor([geometry.target_half_length-.02,geometry.target_half_width-.02],device=actor114.device),maximum_tilt_deg=45.,minimum_clock=.8)
    return DualRateJumpRecovery.ready(observer,actor114,feet6)

def qualified_ready(actor114,feet6,geometry,phase,airborne_count):
    return public_ready(actor114,feet6,geometry)&(phase==4)&(airborne_count>0)

class SubtaskState:
    def __init__(self,env):
        self.last_step=-1
        self.qualified=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
        self.raw_ready=torch.zeros_like(self.qualified)
        self.actor=None;self.feet=None

def state(env):
    if not hasattr(env,'handoff_subtask_state'):env.handoff_subtask_state=SubtaskState(env)
    return env.handoff_subtask_state

def reset(env,env_ids):
    s=state(env);s.qualified[env_ids]=False;s.raw_ready[env_ids]=False

def update(env):
    s=state(env)
    if s.last_step==env.common_step_counter:return torch.zeros_like(s.qualified)
    s.last_step=env.common_step_counter
    jump.update_state(env)
    actor=env.observation_manager.compute_group('handoff_teacher_current',update_history=False)
    feet=current_relative_feet(env)
    assert actor.shape==(env.num_envs,114)
    s.raw_ready=public_ready(actor,feet,env.geometry)
    js=jump.state(env)
    s.qualified=qualified_ready(actor,feet,env.geometry,js.detector.phase,js.event_counts[:,3])
    s.actor=actor.detach().clone();s.feet=feet.detach().clone()
    return torch.zeros_like(s.qualified)

def completed(env):
    update(env)
    return state(env).qualified

def bonus(env):
    # Terminal pulse: manager integrates bonus/dt once on the completing step, then reset clears it.
    return state(env).qualified.float()