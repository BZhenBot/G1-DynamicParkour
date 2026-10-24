"""训练命令统一来源：当前起跳点反馈；只写命令buffer，不写物理状态。"""
from dataclasses import dataclass
import torch
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand,UniformVelocityCommandCfg
from .course_scene import PlatformTargetProvider

def feedback_prepare_command(env,offset_m=None,linear_gain=None,linear_limit=None,yaw_limit=None,preserve_landing_lane=False):
    offset_m=env.turn_launch_offset_m if offset_m is None else offset_m
    linear_gain=env.turn_linear_gain if linear_gain is None else linear_gain
    linear_limit=env.turn_linear_speed_limit if linear_limit is None else linear_limit
    yaw_limit=env.turn_maximum_velocity if yaw_limit is None else yaw_limit
    r=env.scene['robot'];h=r.data.heading_w;center,_,_=PlatformTargetProvider(env).current('middle')
    delta=center[:,:2]-r.data.root_link_pos_w[:,:2];delta=delta.clone();delta[:,0]+=offset_m
    if preserve_landing_lane:delta[:,1]=0.
    velocity=(delta*linear_gain).clamp(-linear_limit,linear_limit)
    angle=torch.atan2(torch.sin(-h),torch.cos(-h))
    yaw=(angle*1.5).clamp(-yaw_limit,yaw_limit)
    stop=torch.linalg.vector_norm(delta,dim=-1).le(.04)&(angle.abs()<=.1)
    c=h.cos();s=h.sin();command=torch.stack((c*velocity[:,0]+s*velocity[:,1],-s*velocity[:,0]+c*velocity[:,1],yaw),-1)
    command[stop]=0
    return command,stop

@dataclass(kw_only=True)
class PrepareVelocityCommandCfg(UniformVelocityCommandCfg):
    def build(self,env):return PrepareVelocityCommand(self,env)

class PrepareVelocityCommand(UniformVelocityCommand):
    def _update_command(self,env_ids=None):
        command,stop=feedback_prepare_command(self._env)
        self.vel_command_b[:]=command;self.is_standing_env[:]=stop
        h=self.robot.data.heading_w;c=h.cos();s=h.sin()
        self.vel_command_w[:,0]=c*command[:,0]-s*command[:,1]
        self.vel_command_w[:,1]=s*command[:,0]+c*command[:,1]
        self.vel_command_w[:,2]=command[:,2]
