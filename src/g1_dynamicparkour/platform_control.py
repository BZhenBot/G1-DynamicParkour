"""mjlab 每个物理子步的外部平台控制；零 Actor 动作维度，reset 后不写状态。"""
from dataclasses import dataclass, replace
import math

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from g1_dynamicparkour.scene import Geometry, make_scene_cfg


@dataclass(kw_only=True)
class PeriodicPlatformActionCfg(ActionTermCfg):
    amplitude: float = .05
    frequency: float = .1
    phase: float = math.pi/2
    random_phase: bool = False
    platform_servo_kp: float = 100000.0
    platform_servo_kv: float = 4000.0

    def build(self, env):
        return PeriodicPlatformAction(self, env)


class PeriodicPlatformAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.t = torch.zeros(self.num_envs, device=self.device)
        self.phase = torch.full_like(self.t, cfg.phase)
        self._raw = torch.zeros((self.num_envs, 0), device=self.device)

    @property
    def action_dim(self):
        return 0

    @property
    def raw_action(self):
        return self._raw

    def process_actions(self, actions):
        assert actions.shape == self._raw.shape

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.t[ids] = 0
        if self.cfg.random_phase:
            self.phase[ids] = torch.rand_like(self.phase[ids])*(2*math.pi)
        else:
            self.phase[ids] = self.cfg.phase
        omega=2*math.pi*self.cfg.frequency
        q=self.cfg.amplitude*torch.sin(self.phase[ids])
        v=self.cfg.amplitude*omega*torch.cos(self.phase[ids])
        # 只有 episode reset 允许设置真实关节初态；运动阶段只写控制目标。
        self._entity.write_joint_state_to_sim(q[:,None], v[:,None], env_ids=env_ids)
        return {}

    def apply_actions(self):
        omega=2*math.pi*self.cfg.frequency
        theta=omega*self.t+self.phase
        q=self.cfg.amplitude*torch.sin(theta)
        v=self.cfg.amplitude*omega*torch.cos(theta)
        a=-self.cfg.amplitude*omega*omega*torch.sin(theta)
        reference=q+(self.cfg.platform_servo_kv/self.cfg.platform_servo_kp)*v+(100./self.cfg.platform_servo_kp)*a
        self._entity.set_joint_position_target(reference[:,None])
        self.t += self._env.physics_dt


def attach_platform_scene(cfg, geometry=Geometry()):
    previous = cfg.scene
    scene=make_scene_cfg(geometry, num_envs=previous.num_envs)
    sensors=[]
    for sensor in previous.sensors:
        if isinstance(sensor, ContactSensorCfg) and sensor.secondary is not None and sensor.secondary.pattern == "terrain":
            target_sensor=replace(sensor, name=sensor.name.replace("ground", "target"),
                secondary=ContactMatch(mode="body", pattern="target_surface", entity="target"))
            sensors.append(target_sensor)
            sensor=replace(sensor, secondary=ContactMatch(mode="body", pattern="start_surface", entity="start"))
        sensors.append(sensor)
    scene.sensors=tuple(sensors)
    cfg.scene=scene
    cfg.actions["periodic_platform"]=PeriodicPlatformActionCfg(entity_name="target", amplitude=geometry.amplitude,frequency=geometry.frequency,platform_servo_kp=geometry.platform_servo_kp,platform_servo_kv=geometry.platform_servo_kv)
    cfg.terminations["illegal_target_contact"]=TerminationTermCfg(func=illegal_target_contact)
    return cfg


def illegal_target_contact(env):
    force=env.scene["nonfoot_target"].data.force
    return (torch.linalg.vector_norm(force,dim=-1)>10).any(dim=-1)
