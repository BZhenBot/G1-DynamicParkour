"""原跳跃PPO到低yaw接管的定向微调；Actor仍114，当前teacher仅reward/termination。"""
import torch
from . import handoff_subtask as hs,jump

def update_yaw_gate(env):
    hs.update(env)
    state=hs.state(env)
    state.qualified &= env.scene['robot'].data.heading_w.abs().le(.15)
    state.qualified &= env.scene['robot'].data.root_link_ang_vel_w[:,2].abs().le(.5)
    return torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)

def yaw_orientation_cost(env):
    phase=jump.state(env).detector.phase
    return (1-env.scene['robot'].data.heading_w.cos())*(phase<=4)

def yaw_rate_cost(env):
    phase=jump.state(env).detector.phase
    return env.scene['robot'].data.root_link_ang_vel_w[:,2].square()*(phase<=4)