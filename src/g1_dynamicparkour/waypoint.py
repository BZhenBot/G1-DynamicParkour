"""Current-state waypoint command; no robot state writes or future trajectory."""
from dataclasses import dataclass

import torch
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import wrap_to_pi

@dataclass(kw_only=True)
class WaypointVelocityCommandCfg(UniformVelocityCommandCfg):
    max_speed: float = .25
    position_gain: float = 1.5
    stop_radius: float = .04
    stop_on_safe_target_support: bool = False
    arrival_min_x_from_target_center: float = -.07

    def build(self, env):
        return WaypointVelocityCommand(self, env)

class WaypointVelocityCommand(UniformVelocityCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.arrival_latched = torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.foot_ids = self.robot.find_sites(("left_foot","right_foot"))[0]

    def _resample_command(self,env_ids):
        super()._resample_command(env_ids)
        self.arrival_latched[env_ids] = False

    def _update_command(self, env_ids=None):
        target = self._env.scene["target"]
        # Entity root is fixed rail base. Current slide q is the actual surface offset.
        target_xy = target.data.root_link_pos_w[:, :2].clone()
        target_xy[:, 1] += target.data.joint_pos[:, 0]
        error = target_xy - self.robot.data.root_link_pos_w[:, :2]
        distance = torch.linalg.vector_norm(error, dim=-1)
        velocity_w = self.cfg.position_gain * error
        speed = torch.linalg.vector_norm(velocity_w, dim=-1)
        velocity_w *= torch.clamp(self.cfg.max_speed / speed.clamp_min(1e-6), max=1)[:, None]
        if self.cfg.stop_on_safe_target_support:
            target_sensor = self._env.scene["feet_target_contact"].data
            start_sensor = self._env.scene["feet_ground_contact"].data
            target_support = (target_sensor.found>0) & (torch.linalg.vector_norm(target_sensor.force,dim=-1)>5)
            start_support = (start_sensor.found>0) & (torch.linalg.vector_norm(start_sensor.force,dim=-1)>5)
            feet = self.robot.data.site_pos_w[:,self.foot_ids,:2]
            # This mode is explicitly limited to the Stage1 static target geometry.
            bounds = torch.tensor([.3-.02,.5-.02],device=self.device)
            inside = ((feet-target_xy[:,None,:]).abs()<=bounds).all(dim=-1).all(dim=-1)
            deep_enough = self.robot.data.root_link_pos_w[:,0] >= target_xy[:,0] + self.cfg.arrival_min_x_from_target_center
            self.arrival_latched |= target_support.all(dim=-1) & ~start_support.any(dim=-1) & inside & deep_enough
        self.is_standing_env[:] = (distance < self.cfg.stop_radius) | self.arrival_latched
        velocity_w[self.is_standing_env] = 0
        self.vel_command_w[:, :2] = velocity_w
        heading = self.robot.data.heading_w
        c, s = torch.cos(heading), torch.sin(heading)
        self.vel_command_b[:, 0] = c*velocity_w[:, 0] + s*velocity_w[:, 1]
        self.vel_command_b[:, 1] = -s*velocity_w[:, 0] + c*velocity_w[:, 1]
        self.vel_command_b[:, 2] = (self.cfg.heading_control_stiffness * wrap_to_pi(-heading)).clamp(-.5,.5)
        self.vel_command_w[:, 2] = self.vel_command_b[:, 2]
        self.vel_command_b[self.is_standing_env] = 0

def attach_waypoint_command(cfg, max_speed=.25, stop_on_arrival=False):
    cfg.commands["twist"] = WaypointVelocityCommandCfg(entity_name="robot",
        resampling_time_range=(100,100), debug_vis=False, max_speed=max_speed,
        heading_control_stiffness=.5, stop_on_safe_target_support=stop_on_arrival,
        ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(0,.25),lin_vel_y=(-.25,.25),ang_vel_z=(-.5,.5)))
    return cfg