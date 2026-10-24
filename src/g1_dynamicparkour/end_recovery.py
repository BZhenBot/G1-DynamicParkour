"""真实终点接管reset课程；仅恢复子任务，保持三表面动力学与V1严格稳定。"""
import hashlib,json,math
from pathlib import Path
import torch
from .course_turn import MiddleResetBank
from .course_scene import PlatformTargetProvider,contacts,SURFACES,unsafe
from . import stage1,jump

class EndResetBank(MiddleResetBank):
    def __init__(self,manifest):
        m=json.loads(Path(manifest).read_text());chosen=[r for r in m['records'] if r['selected_for_initial_recovery_curriculum']]
        self.paths=[Path(r['path']) for r in chosen];self.records=[]
        assert len(chosen)>=2
        for item,path in zip(chosen,self.paths):
            assert item['two_airborne_origin_verified_200hz'] and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
            record=json.loads(path.read_text());assert record['event']=='actual_end_handoff_after_second_airborne'
            assert all(record['surface_supports']['end']) and not any(record['surface_supports']['start']) and not any(record['surface_supports']['middle'])
            self.records.append(record)

def state(env):
    if not hasattr(env,'end_recovery_state'):
        n=env.num_envs;d=env.device
        env.end_recovery_state={'last_step':-1,'stable_since':torch.full((n,),-1.,device=d,dtype=torch.float64),'stable_time':torch.zeros(n,device=d),'success':torch.zeros(n,device=d,dtype=torch.bool),
            'failed':torch.zeros(n,device=d,dtype=torch.bool),'success_pulse':torch.zeros(n,device=d,dtype=torch.bool),'failure_pulse':torch.zeros(n,device=d,dtype=torch.bool)}
    return env.end_recovery_state

def stable(env):
    r=env.scene['robot'];p=PlatformTargetProvider(env);center,velocity,half=p.current('end');c,_=contacts(env)
    feet=r.data.site_pos_w[:,r.find_sites(('left_foot','right_foot'))[0]]-center[:,None,:]
    speed=r.data.root_link_lin_vel_w-velocity
    return (c['end'].all(-1)&~c['start'].any(-1)&~c['middle'].any(-1)&(feet[:,:,:2].abs()<=half-.02).all(-1).all(-1)
        &((feet[:,:,2]>=-.01)&(feet[:,:,2]<=.04)).all(-1)&(-r.data.projected_gravity_b[:,2]>=math.cos(math.radians(15)))
        &torch.linalg.vector_norm(speed[:,:2],dim=-1).le(.15)&speed[:,2].abs().le(.1)
        &torch.linalg.vector_norm(r.data.root_link_ang_vel_w,dim=-1).le(.5)&~unsafe(env))

def update(env):
    s=state(env)
    if s['last_step']==env.common_step_counter:return torch.zeros_like(s['failed'])
    s['last_step']=env.common_step_counter;stage1.update_state(env);c,_=contacts(env)
    s['failed']=unsafe(env)|c['start'].any(-1)|c['middle'].any(-1)
    good=stable(env)&~s['failed'];now=env.episode_length_buf.double()*env.step_dt
    s['stable_since'],s['stable_time']=advance_stability(s['stable_since'],good,now)
    success=s['stable_time']>=2.;s['success_pulse']=success&~s['success'];s['success']=success;s['failure_pulse']=s['failed']
    env.extras['log']['Metrics/end_recovery/stable_fraction']=float(good.float().mean())
    return torch.zeros_like(s['failed'])

def completed(env):return state(env)['success']
def failed(env):return state(env)['failed']
def success_bonus(env):return state(env)['success_pulse'].float()
def failure_bonus(env):return state(env)['failure_pulse'].float()


def advance_stability(since,good,now):
    since=torch.where(good,torch.where(since<0,now,since),torch.full_like(since,-1.))
    elapsed=torch.where(good,now-since,torch.zeros_like(now))
    return since,elapsed

def stable_reward(env):return stable(env).float()


def initial_contact_measurement(env):
    """仅reset首帧复用实际采集的最近公开测量，物理传感器/奖励不修改。"""
    from .course_scene import unlabelled_contact
    value=unlabelled_contact(env)
    history=getattr(env,'initial_public_contact_measurement',None)
    if history is not None:
        first=env.episode_length_buf==0
        value=torch.where(first[:,None],history,value)
    return value