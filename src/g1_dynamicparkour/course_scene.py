"""三平台场景与可测当前目标；复用 V1 机器人、真实滑轨和接触配置。"""
from dataclasses import dataclass, replace
import math
import torch
import mujoco
from mjlab.entity import EntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from .scene import Geometry, start_spec, make_scene_cfg

SURFACES = ('start', 'middle', 'end')
SENSOR_SUFFIX = {'start': 'ground', 'middle': 'middle', 'end': 'target'}

@dataclass(frozen=True)
class CourseGeometry:
    gap1: float = .02
    gap2: float = .02
    middle_half_length: float = .3
    end_half_length: float = .3
    half_width: float = .5
    amplitude: float = .05
    frequency: float = .1
    @property
    def middle_x(self): return -.2 + .4 + self.gap1 + self.middle_half_length
    @property
    def end_x(self): return self.middle_x + self.middle_half_length + self.gap2 + self.end_half_length
    def validate(self):
        if not all(math.isfinite(v) for v in self.__dict__.values()): raise ValueError('非有限地形')
        if min(self.gap1,self.gap2,self.middle_half_length,self.end_half_length,self.half_width)<=0: raise ValueError('须有真实间隙/支撑面')
        if not 0<self.amplitude<=.1 or not 0<self.frequency<=.25: raise ValueError('动态终点超出验证候选范围')


def middle_spec(course):
    return mujoco.MjSpec.from_string(f'''<mujoco><worldbody><body name="middle_surface">
    <geom name="middle_collision" type="box" size="{course.middle_half_length} {course.half_width} .1"
    friction=".8 .005 .0001" rgba=".7 .5 .2 1"/></body></worldbody></mujoco>''')


def attach_course_scene(cfg, course):
    course.validate()
    # 先从真实 V1 factory 获取机器人/导轨/驱动；只追加固定中间台并定位终点。
    g=Geometry(gap=course.gap1,robot_start_offset_x=.2,target_half_length=course.end_half_length,
               target_half_width=course.half_width,amplitude=course.amplitude,frequency=course.frequency,
               platform_servo_kp=900000.,platform_servo_kv=36000.)
    scene=make_scene_cfg(g,num_envs=cfg.scene.num_envs)
    target=scene.entities['target']
    target.init_state=replace(target.init_state,pos=(course.end_x,0.,-.1))
    scene.entities['middle']=EntityCfg(spec_fn=lambda:middle_spec(course),
        init_state=EntityCfg.InitialStateCfg(pos=(course.middle_x,0.,-.1)))
    sensors=[]
    for sensor in cfg.scene.sensors:
        sensors.append(sensor)
        if isinstance(sensor,ContactSensorCfg) and sensor.secondary is not None and sensor.secondary.entity=='target':
            sensors.append(replace(sensor,name=sensor.name.replace('target','middle'),
                secondary=ContactMatch(mode='body',pattern='middle_surface',entity='middle')))
    scene.sensors=tuple(sensors); scene.extent=4.; cfg.scene=scene
    cfg.actions['periodic_platform'].amplitude=course.amplitude
    cfg.actions['periodic_platform'].frequency=course.frequency
    for name in ('reset_platform_state','reset_jump_state'): cfg.events.pop(name,None)
    for name in ('update_jump_state','jump_success','jump_failure','illegal_ground_contact','illegal_target_contact'):
        cfg.terminations.pop(name,None)
    cfg.terminations['course_unsafe']=TerminationTermCfg(func=unsafe)
    # 冻结策略评价暂保留本体奖励，移除依赖单段任务FSM的奖励。未来微调独立配置。
    for name in list(cfg.rewards):
        func=cfg.rewards[name].func
        if getattr(func,'__module__','') in ('g1_dynamicparkour.jump','g1_dynamicparkour.stage1') and name!='action_rate_l2':
            cfg.rewards.pop(name)
    for group in ('actor','critic'):
        cfg.observations[group].terms['platform_relative_state']=ObservationTermCfg(func=current_relative)
        cfg.observations[group].terms['task_clock']=ObservationTermCfg(func=skill_clock)
    cfg.observations['actor'].terms['current_unlabelled_foot_contact']=ObservationTermCfg(func=unlabelled_contact)
    cfg.observations['critic'].terms['foot_contact'].func=feet_contact
    cfg.observations['critic'].terms['foot_contact'].params={}
    cfg.observations['critic'].terms['foot_contact_forces'].func=feet_forces
    cfg.observations['critic'].terms['foot_contact_forces'].params={}
    cfg.viewer.distance,cfg.viewer.azimuth,cfg.viewer.elevation=3.6,90,-25
    cfg.viewer.width,cfg.viewer.height=640,480
    return cfg,g


def contacts(env):
    result={}; forces={}
    for name in SURFACES:
        d=env.scene['feet_'+SENSOR_SUFFIX[name]+'_contact'].data
        f=torch.linalg.vector_norm(d.force,dim=-1)
        result[name]=(d.found>0)&(f>5)
        forces[name]=d.force
    return result,forces


def feet_contact(env):
    c,_=contacts(env)
    return torch.stack(list(c.values())).any(0).float()


def feet_forces(env):
    _,f=contacts(env); total=torch.stack(list(f.values())).sum(0).flatten(1)
    return torch.sign(total)*torch.log1p(total.abs())


def unlabelled_contact(env): return torch.cat((feet_contact(env),feet_forces(env)),-1)


def unsafe(env):
    robot=env.scene['robot']
    result=(robot.data.root_link_pos_w[:,2]<.5)|(-robot.data.projected_gravity_b[:,2]<math.cos(math.radians(70)))
    for name in SENSOR_SUFFIX.values():
        result |= torch.linalg.vector_norm(env.scene['nonfoot_'+name].data.force,dim=-1).gt(10).any(-1)
    return result


class PlatformTargetProvider:
    def __init__(self,env): self.env=env
    def current(self,platform_id):
        e=self.env.scene['target' if platform_id=='end' else platform_id]
        center=e.data.root_link_pos_w.clone(); center[:,2]+=.1
        velocity=torch.zeros_like(center)
        if platform_id=='end':
            center[:,1]+=e.data.joint_pos[:,0]; velocity[:,1]=e.data.joint_vel[:,0]
        half_x={'start':.4,'middle':self.env.course.middle_half_length,'end':self.env.course.end_half_length}[platform_id]
        return center,velocity,torch.tensor([half_x,self.env.course.half_width],device=self.env.device)
    def relative(self,platform_id):
        center,velocity,_=self.current(platform_id); r=self.env.scene['robot']; h=r.data.heading_w
        p=center-r.data.root_link_pos_w; v=velocity-r.data.root_link_lin_vel_w
        c,s=h.cos(),h.sin()
        def rotate(x): return torch.stack((c*x[:,0]+s*x[:,1],-s*x[:,0]+c*x[:,1],x[:,2]),-1)
        return torch.cat((rotate(p),rotate(v)),-1)
    def relative_feet(self,platform_id):
        center,_,_=self.current(platform_id); r=self.env.scene['robot']; ids=r.find_sites(('left_foot','right_foot'))[0]
        return (r.data.site_pos_w[:,ids]-center[:,None,:]).flatten(1)


def current_relative(env): return PlatformTargetProvider(env).relative(getattr(env,'course_target_id','middle'))
def skill_clock(env):
    return ((env.episode_length_buf.float()*env.step_dt-getattr(env,'course_skill_start_s',0.))/1.).clamp(0,1)[:,None]


class SkillObservationAdapter:
    def __init__(self,env): self.env=env; self.provider=PlatformTargetProvider(env)
    def for_skill(self,obs,target_id,skill_start_s):
        actor=obs['actor'].clone()
        if actor.shape[-1]!=114: raise ValueError('必须保留V1 Actor114布局')
        relative=self.provider.relative(target_id)
        offset=getattr(self.env,'landing_goal_y_offset',0.) if target_id=='end' else 0.
        if offset:
            h=self.env.scene['robot'].data.heading_w
            relative[:,0]+=h.sin()*offset;relative[:,1]+=h.cos()*offset
        actor[:,99:105]=relative
        actor[:,105]=max(0.,min(1.,float(self.env.episode_length_buf[0])*self.env.step_dt-skill_start_s))
        # 所有表面的无身份接触已经由 ObservationManager 在同一当前步采集。
        return actor,self.provider.relative_feet(target_id)