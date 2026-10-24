"""mjlab 1.6.0 单间隙场景；固定起点、真实滑轨驱动目标，无机器人状态覆盖。"""
from dataclasses import dataclass, replace
from math import pi, sin, cos

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.scene import SceneCfg


@dataclass(frozen=True)
class Geometry:
    start_center_x: float = -0.2
    robot_start_offset_x: float = 0.0
    start_half_length: float = 0.4
    start_half_width: float = 0.5
    target_half_length: float = 0.3
    target_half_width: float = 0.5
    gap: float = 0.15
    top_height: float = 0.0
    height_difference: float = 0.0
    half_thickness: float = 0.1
    amplitude: float = 0.05
    frequency: float = 0.1
    platform_servo_kp: float = 100000.0
    platform_servo_kv: float = 4000.0

    @property
    def target_center_x(self):
        return self.start_center_x + self.start_half_length + self.gap + self.target_half_length


def start_spec(geometry):
    return mujoco.MjSpec.from_string(f'''<mujoco><worldbody>
    <body name="start_surface">
    <geom name="start_collision" type="box" size="{geometry.start_half_length} {geometry.start_half_width} {geometry.half_thickness}"
    friction=".8 .005 .0001" rgba=".3 .5 .3 1"/>
    </body></worldbody></mujoco>''')


def target_spec(geometry):
    return mujoco.MjSpec.from_string(f'''<mujoco><worldbody>
    <body name="rail_base" mocap="true"><body name="target_surface">
    <joint name="lateral_slide" type="slide" axis="0 1 0" range="-.2 .2"/>
    <geom name="target_collision" type="box" size="{geometry.target_half_length} {geometry.target_half_width} {geometry.half_thickness}"
    mass="100" friction=".8 .005 .0001" rgba=".15 .35 .8 1"/>
    </body></body></worldbody><actuator>
    <position name="platform_servo" joint="lateral_slide" kp="{geometry.platform_servo_kp}" kv="{geometry.platform_servo_kv}"/>
    </actuator></mujoco>''')


def make_scene_cfg(geometry=Geometry(), num_envs=1):
    if not 0 <= geometry.amplitude <= 0.1 or not 0 < geometry.frequency <= 0.25:
        raise ValueError("初版物理验证范围为 A<=0.1m、0<f<=0.25Hz")
    if not 4000 <= geometry.platform_servo_kv <= 36000:
        raise ValueError("Current declared platform damping candidates are4000–36000 N s/m")
    if not 100000 <= geometry.platform_servo_kp <= 900000:
        raise ValueError("Current platform stiffness candidates100000–900000 N/m")
    if geometry.gap <= 0:
        raise ValueError("必须保留真实间隙")
    if abs(geometry.robot_start_offset_x) >= geometry.start_half_length:
        raise ValueError("Initial root XY must remain on the starting platform; physical foot bounds still need verification")
    robot = get_g1_robot_cfg()
    robot.init_state = replace(robot.init_state, pos=(geometry.start_center_x+geometry.robot_start_offset_x, 0.0, robot.init_state.pos[2]+geometry.top_height))
    start = EntityCfg(spec_fn=lambda: start_spec(geometry), init_state=EntityCfg.InitialStateCfg(
        pos=(geometry.start_center_x, 0.0, geometry.top_height-geometry.half_thickness)))
    target = EntityCfg(spec_fn=lambda: target_spec(geometry), init_state=EntityCfg.InitialStateCfg(
        pos=(geometry.target_center_x, 0.0, geometry.top_height+geometry.height_difference-geometry.half_thickness)),
        articulation=EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(target_names_expr=("lateral_slide",)),)))
    return SceneCfg(num_envs=num_envs, entities={"robot": robot, "start": start, "target": target}, extent=3.0)


def servo_reference(t, amplitude, frequency, phase=pi/2, *, platform_servo_kp=100000.0, platform_servo_kv=4000.0):
    omega = 2*pi*frequency
    position = amplitude*sin(omega*t+phase)
    velocity = amplitude*omega*cos(omega*t+phase)
    acceleration = -amplitude*omega*omega*sin(omega*t+phase)
    control = position+(platform_servo_kv/platform_servo_kp)*velocity+(100/platform_servo_kp)*acceleration
    return position, velocity, control
