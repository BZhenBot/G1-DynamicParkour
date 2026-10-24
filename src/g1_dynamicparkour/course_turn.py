"""真实中间落地状态的转向子任务；训练reset允许恢复，完整路线禁止。"""
import json,math
from pathlib import Path
import torch
from .course_scene import PlatformTargetProvider,contacts,SENSOR_SUFFIX

def ensure(env):
    if not hasattr(env,'turn_state'):
        n=env.num_envs;d=env.device
        env.turn_state={'last_step':-1,'stable_time':torch.zeros(n,device=d),'success':torch.zeros(n,device=d,dtype=torch.bool),
            'failed':torch.zeros(n,device=d,dtype=torch.bool),'air_time':torch.zeros((n,2),device=d),
            'previous_cos':torch.zeros(n,device=d),'valid':torch.zeros(n,device=d,dtype=torch.bool),'progress':torch.zeros(n,device=d),
            'success_pulse':torch.zeros(n,device=d,dtype=torch.bool),'failure_pulse':torch.zeros(n,device=d,dtype=torch.bool)}
    return env.turn_state

def goal_error(env):
    if getattr(env,'turn_world_heading',False):
        h=env.scene['robot'].data.heading_w
        return torch.atan2(torch.sin(-h),torch.cos(-h))
    p=PlatformTargetProvider(env).relative('end')[:,:2]
    return torch.atan2(p[:,1],p[:,0])

def desired_turn_velocity(env):
    if getattr(env,'consistent_prepare_command',False):return env.command_manager.get_command('twist')[:,2]
    maximum=getattr(env,'turn_maximum_velocity',.8)
    return (goal_error(env)*1.5).clamp(-maximum,maximum)

def safe_middle(env):
    r=env.scene['robot'];p=PlatformTargetProvider(env);center,_,half=p.current('middle');ids=r.find_sites(('left_foot','right_foot'))[0]
    offset=r.data.site_pos_w[:,ids]-center[:,None,:];c,_=contacts(env)
    inside=(offset[:,:,:2].abs()<=half-.02).all(-1).all(-1)
    top=((offset[:,:,2]>=-.01)&(offset[:,:,2]<=.04)).all(-1)
    stable=c['middle'].all(-1)&~c['start'].any(-1)&~c['end'].any(-1)&inside&top
    stable &= -r.data.projected_gravity_b[:,2]>=math.cos(math.radians(15))
    stable &= torch.linalg.vector_norm(r.data.root_link_lin_vel_w[:,:2],dim=-1)<=.15
    stable &= r.data.root_link_lin_vel_w[:,2].abs()<=.1
    stable &= torch.linalg.vector_norm(r.data.root_link_ang_vel_w,dim=-1)<=.5
    return stable

def update(env):
    s=ensure(env)
    if s['last_step']==env.common_step_counter:return torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
    s['last_step']=env.common_step_counter;c,_=contacts(env);r=env.scene['robot']
    any_support=torch.stack(list(c.values())).any(0)
    s['air_time']=torch.where(any_support,torch.zeros_like(s['air_time']),s['air_time']+env.step_dt)
    cos=goal_error(env).cos();s['progress']=torch.where(s['valid'],(cos-s['previous_cos'])/env.step_dt,0)
    s['previous_cos']=cos;s['valid'][:]=True
    s['failed']=c['start'].any(-1)|c['end'].any(-1)|(r.data.root_link_pos_w[:,2]<.5)|(-r.data.projected_gravity_b[:,2]<math.cos(math.radians(70)))
    for suffix in SENSOR_SUFFIX.values():s['failed']|=torch.linalg.vector_norm(env.scene['nonfoot_'+suffix].data.force,dim=-1).gt(10).any(-1)
    middle_safe=safe_middle(env);s['diagnostic_middle_safe']=middle_safe
    good=middle_safe&goal_error(env).abs().le(getattr(env,'settle_heading_bound_rad',.1))&~s['failed']
    if getattr(env,'turn_require_launch_pose',False):
        center,_,_=PlatformTargetProvider(env).current('middle');ids=r.find_sites(('left_foot','right_foot'))[0]
        error=(r.data.root_link_pos_w[:,0]-center[:,0]-getattr(env,'turn_launch_offset_m',.1)).abs()
        square=(r.data.site_pos_w[:,ids[0],0]-r.data.site_pos_w[:,ids[1],0]).abs()
        s['diagnostic_launch_x_error_m']=error;s['diagnostic_stance_x_difference_m']=square
        good &= error.le(getattr(env,'settle_position_tolerance_m',.04))&square.le(.1)
    if getattr(env,'settle_quiet_joint_limit',None) is not None:
        good &= r.data.joint_vel[:,:12].abs().le(env.settle_quiet_joint_limit).all(-1)
        good &= torch.linalg.vector_norm(r.data.root_link_lin_vel_w[:,:2],dim=-1).le(.1)&torch.linalg.vector_norm(r.data.root_link_ang_vel_w,dim=-1).le(.25)
    s['stable_time']=torch.where(good,s['stable_time']+env.step_dt,0)
    new=s['stable_time']>=.20;s['success_pulse']=new&~s['success'];s['success']=new
    s['failure_pulse']=s['failed']
    env.extras['log']['Metrics/turn/mean_abs_goal_error_rad']=float(goal_error(env).abs().mean())
    return torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)

def success(env):return ensure(env)['success']
def failed(env):return ensure(env)['failed']
def heading_reward(env):return goal_error(env).cos()
def progress(env):return ensure(env)['progress']
def angular_cost(env):return (env.scene['robot'].data.root_link_ang_vel_w[:,2]-desired_turn_velocity(env)).square()
def center_cost(env):
    center,_,_=PlatformTargetProvider(env).current('middle');desired=center[:,:2].clone();desired[:,0]+=getattr(env,'turn_launch_offset_m',.1)
    return (env.scene['robot'].data.root_link_pos_w[:,:2]-desired).square().sum(-1)
def velocity_cost(env):return env.scene['robot'].data.root_link_lin_vel_w.square().sum(-1)
def outside_cost(env):
    center,_,half=PlatformTargetProvider(env).current('middle');r=env.scene['robot'];ids=r.find_sites(('left_foot','right_foot'))[0]
    delta=(r.data.site_pos_w[:,ids,:2]-center[:,None,:2]).abs();return (delta-(half-.02)).clamp_min(0).square().sum(-1).mean(-1)
def height_cost(env):return (.68-env.scene['robot'].data.root_link_pos_w[:,2]).clamp_min(0).square()
def success_bonus(env):return ensure(env)['success_pulse'].float()
def failure_bonus(env):return ensure(env)['failure_pulse'].float()
def feet_air_time(env):return ensure(env)['air_time']

class MiddleResetBank:
    def __init__(self,paths):
        self.paths=[Path(p) for p in paths];self.records=[json.loads(p.read_text()) for p in self.paths]
        if not self.records:raise ValueError('缺少真实中间落地快照')
        for r in self.records:
            assert r['event']=='actual_middle_stable_before_second_skill'
    def restore(self,env,ids,indices):
        records=[self.records[i] for i in indices.cpu().tolist()]
        assert list(env.scene['robot'].joint_names)==records[0]['joint_names']
        assert env.course.__dict__==records[0]['course']
        def v(key):return torch.tensor([r[key] for r in records],device=env.device)
        robot=env.scene['robot'];root=v('root_state_world_link_13');root[:,:3]+=env.scene.env_origins[ids]
        robot.write_root_state_to_sim(root,env_ids=ids);robot.write_joint_state_to_sim(v('joint_position'),v('joint_velocity'),env_ids=ids)
        env.scene['target'].write_joint_state_to_sim(v('platform_joint_position'),v('platform_joint_velocity'),env_ids=ids)
        env.sim.data.qacc_warmstart[ids]=v('qacc_warmstart')
        term=env.action_manager.get_term('joint_pos');term._raw_actions[ids]=v('raw_actual_joint_action')
        term._processed_actions[ids]=v('robot_joint_position_target');robot.set_joint_position_target(v('robot_joint_position_target'),env_ids=ids)
        current_action=v('raw_actual_joint_action');previous_action=v('previous_actual_joint_action')
        if env.action_manager.total_action_dim==30:
            zero=torch.zeros((len(ids),1),device=env.device)
            current_action=torch.cat((current_action,zero),-1);previous_action=torch.cat((previous_action,zero),-1)
            term.previous_joint_actions[ids]=v('previous_actual_joint_action');term.last_recovery_coordinate[ids]=0.
        env.action_manager.action[ids]=current_action;env.action_manager.prev_action[ids]=previous_action
        if getattr(env,'bank_original_units',None) is not None:
            # 真实bank来自Jump的sagittal×2；Locomotion使用原动作单位×1。
            # 按记录的实际position target重建当前raw，并将历史换回同一物理单位。
            scale=term.scale[ids] if torch.is_tensor(term.scale) else term.scale
            offset=term.offset[ids] if torch.is_tensor(term.offset) else term.offset
            current=(v('robot_joint_position_target')-offset)/scale
            previous=v('previous_actual_joint_action')*env.bank_original_units
            term._raw_actions[ids]=current;env.action_manager.action[ids]=current;env.action_manager.prev_action[ids]=previous
        periodic=env.action_manager.get_term('periodic_platform');periodic.t[ids]=v('platform_clock_s');periodic.phase[ids]=v('platform_phase_rad')
        s=ensure(env)
        for key in ('stable_time','success','failed','air_time','previous_cos','valid','progress','success_pulse','failure_pulse'):s[key][ids]=0

def roll_pitch_rate_cost(env):
    return env.scene['robot'].data.root_link_ang_vel_b[:,:2].square().sum(-1)

def planar_tilt_cost(env):
    return env.scene['robot'].data.projected_gravity_b[:,:2].square().sum(-1)


def ankle_roll_velocity_cost(env):
    robot=env.scene['robot'];ids=[i for i,name in enumerate(robot.joint_names) if name.endswith('_ankle_roll_joint')]
    assert len(ids)==2
    return robot.data.joint_vel[:,ids].square().sum(-1)
