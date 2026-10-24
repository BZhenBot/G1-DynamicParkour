"""真实准备态到终点的第二跳子任务；保留过去源台碰撞，Actor无表面ID。"""
import torch
from . import handoff_subtask as hs,stage1

def other_failed(env):
    foot=env.scene['feet_past_start_contact'].data
    supported=(foot.found>0)&torch.linalg.vector_norm(foot.force,dim=-1).gt(5)
    illegal=torch.linalg.vector_norm(env.scene['nonfoot_past_start'].data.force,dim=-1).gt(10)
    return supported.any(-1)|illegal.any(-1)

def update(env):
    hs.update(env);s=hs.state(env);s.qualified &= ~other_failed(env)
    limit=env.settings.get('handoff_velocity_contract')
    if limit:
        actor=s.actor
        s.qualified &= torch.linalg.vector_norm(actor[:,102:104],dim=-1).le(limit['horizontal_speed_mps'])
        s.qualified &= actor[:,104].abs().le(limit['vertical_speed_mps'])
        s.qualified &= torch.linalg.vector_norm(actor[:,3:6],dim=-1).le(limit['angular_speed_radps'])
    capture_limit=env.settings.get('handoff_maximum_backward_capture_offset_m')
    if capture_limit is not None:
        if not 0<capture_limit<=.2:raise ValueError('近似捕获偏差候选阈值无效')
        value=capture_offset(env)
        s.qualified &= torch.isfinite(value)&value.le(capture_limit)
    return torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)

def failure_event(env):return other_failed(env).float()

def current_contact(env):
    a,b,_=stage1.contacts(env);past=env.scene['feet_past_start_contact'].data
    c=(past.found>0)&torch.linalg.vector_norm(past.force,dim=-1).gt(5)
    net=env.scene['feet_ground_contact'].data.force+env.scene['feet_target_contact'].data.force+past.force
    net=net.flatten(1)
    return torch.cat(((a|b|c).float(),torch.sign(net)*torch.log1p(net.abs())),-1)

def lane_relative_state(env):
    """与部署一致的当前随动着陆指令；偏移只在训练reset锁定，不读未来。"""
    from . import jump
    relative=jump.platform_relative_state(env)
    offset=getattr(env,'prepared_goal_y_offset',None)
    if offset is not None:
        h=env.scene['robot'].data.heading_w
        relative[:,0]+=h.sin()*offset;relative[:,1]+=h.cos()*offset
    return relative

def capture_inputs(env):
    r=env.scene['robot'];target=env.scene['target'];center_x=target.data.root_link_pos_w[:,0]
    feet=r.data.site_pos_w[:,stage1.state(env).foot_ids,0]-center_x[:,None]
    return {'root_x':r.data.root_link_pos_w[:,0]-center_x,'root_vx':r.data.root_link_lin_vel_w[:,0],
        'height':r.data.root_link_pos_w[:,2]-target.data.root_link_pos_w[:,2]-env.geometry.half_thickness,
        'mean_foot_x':feet.mean(-1),'gravity':abs(float(env.sim.mj_model.opt.gravity[2]))}

def capture_offset(env):
    from .capture_proxy import backward_capture_offset
    return backward_capture_offset(**capture_inputs(env))
