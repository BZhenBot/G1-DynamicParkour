"""Stage1 task signals from actual dual-platform contacts, integrated once/control step."""
import torch

def contacts(env):
    start = env.scene["feet_ground_contact"].data
    target = env.scene["feet_target_contact"].data
    sf = torch.linalg.vector_norm(start.force, dim=-1)
    tf = torch.linalg.vector_norm(target.force, dim=-1)
    supported_start = (start.found > 0) & (sf > 5)
    supported_target = (target.found > 0) & (tf > 5)
    return supported_start, supported_target, start.force + target.force

def target_xy(env):
    target = env.scene["target"]
    xy = target.data.root_link_pos_w[:, :2].clone()
    xy[:,1] += target.data.joint_pos[:,0]
    return xy

class TaskState:
    def __init__(self, env):
        n,device = env.num_envs,env.device
        self.foot_ids = env.scene["robot"].find_sites(("left_foot","right_foot"))[0]
        self.air_time = torch.zeros((n,2),device=device)
        self.last_support = torch.ones((n,2),device=device,dtype=torch.bool)
        self.distance = torch.zeros(n,device=device)
        self.previous_distance = torch.zeros_like(self.distance)
        self.valid = torch.zeros(n,device=device,dtype=torch.bool)
        self.start_seen = torch.zeros_like(self.valid)
        self.progress_rate = torch.zeros(n,device=device)
        self.stable_time = torch.zeros(n,device=device)
        self.stable = torch.zeros_like(self.valid)
        self.success = torch.zeros_like(self.valid)
        self.success_pulse = torch.zeros_like(self.valid)
        self.success_claimed = torch.zeros_like(self.valid)
        self.first_target = torch.zeros_like(self.valid)
        self.reached_target = torch.zeros_like(self.valid)
        self.relative_speed_sq = torch.zeros(n,device=device)
        self.peak_target_force = torch.zeros(n,device=device)
        self.control_step = -1

def state(env):
    if not hasattr(env,"platform_task_state"):
        env.platform_task_state = TaskState(env)
    return env.platform_task_state

def reset_state(env, env_ids):
    s = state(env)
    for name in ("air_time","distance","previous_distance","valid","progress_rate","stable_time",
                 "stable","success","first_target","reached_target","relative_speed_sq","peak_target_force","start_seen","success_pulse","success_claimed"):
        getattr(s,name)[env_ids] = 0
    s.last_support[env_ids] = True

def update_state(env):
    s = state(env)
    if s.control_step == env.common_step_counter:
        return torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
    s.control_step = env.common_step_counter
    robot,target = env.scene["robot"],env.scene["target"]
    start_support,target_support,forces = contacts(env)
    supported = start_support | target_support
    s.start_seen |= start_support.all(dim=-1)
    s.air_time = torch.where(supported,torch.zeros_like(s.air_time),s.air_time+env.step_dt)
    s.last_support[:] = supported
    xy = target_xy(env)
    s.distance[:] = torch.linalg.vector_norm(xy-robot.data.root_link_pos_w[:,:2],dim=-1)
    s.progress_rate[:] = torch.where(s.valid,(s.previous_distance-s.distance)/env.step_dt,0)
    s.previous_distance[:] = s.distance
    s.valid[:] = True
    feet = robot.data.site_pos_w[:,s.foot_ids,:2]
    bounds = torch.tensor([env.geometry.target_half_length-.02,env.geometry.target_half_width-.02],device=env.device)
    inside = ((feet-xy[:,None,:]).abs() <= bounds).all(dim=-1).all(dim=-1)
    target_vel = torch.zeros_like(xy)
    target_vel[:,1] = target.data.joint_vel[:,0]
    s.relative_speed_sq[:] = (robot.data.root_link_lin_vel_w[:,:2]-target_vel).square().sum(dim=-1)
    gravity = robot.data.projected_gravity_b
    upright = -gravity[:,2] >= torch.cos(torch.tensor(15*torch.pi/180,device=env.device))
    omega = torch.linalg.vector_norm(robot.data.root_link_ang_vel_w,dim=-1)
    safe = robot.data.root_link_pos_w[:,2] >= .5
    for name in ("nonfoot_ground","nonfoot_target"):
        safe &= ~(torch.linalg.vector_norm(env.scene[name].data.force,dim=-1)>10).any(dim=-1)
    s.stable[:] = safe & target_support.all(dim=-1) & ~start_support.any(dim=-1) & inside & upright & (s.relative_speed_sq<=.15**2) & (omega<=.5)
    s.stable_time[:] = torch.where(s.stable,s.stable_time+env.step_dt,0)
    s.success[:] = (s.stable_time+1e-6 >= 2) & s.start_seen
    s.success_pulse[:] = s.success & ~s.success_claimed
    s.success_claimed |= s.success
    contact = target_support.any(dim=-1)
    s.first_target[:] = contact & ~s.reached_target & s.start_seen
    s.reached_target |= contact
    s.peak_target_force[:] = torch.maximum(s.peak_target_force,torch.linalg.vector_norm(env.scene["feet_target_contact"].data.force,dim=-1).sum(dim=-1))
    return torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)

def task_success(env):
    return state(env).success

def progress(env):
    return state(env).progress_rate

def goal_proximity(env):
    return torch.exp(-state(env).distance.square()/.15**2)

def stop_velocity_cost(env):
    s = state(env)
    return s.relative_speed_sq*(s.distance<.2)

def stable_reward(env):
    return state(env).stable.float()

def target_contact_event(env):
    return state(env).first_target.float()

def success_event(env):
    return state(env).success_pulse.float()

def feet_contact(env):
    a,b,_ = contacts(env)
    return (a|b).float()

def feet_forces(env):
    forces = contacts(env)[2].flatten(start_dim=1)
    # Preserve mjlab privileged force encoding for checkpoint compatibility.
    return torch.sign(forces) * torch.log1p(torch.abs(forces))

def feet_air_time(env):
    return state(env).air_time

def air_time_reward(env):
    command = env.command_manager.get_command("twist")
    active = torch.linalg.vector_norm(command[:,:2],dim=-1)>.05
    air = state(env).air_time
    return ((air>.05)&(air<.5)).float().sum(dim=-1)*active

def foot_slip_cost(env):
    s = state(env)
    a,b,_ = contacts(env)
    robot,target = env.scene["robot"],env.scene["target"]
    velocity = robot.data.site_lin_vel_w[:,s.foot_ids,:2]
    target_velocity = torch.zeros_like(velocity)
    target_velocity[:,:,1] = target.data.joint_vel[:,0,None]
    # Start has zero world velocity; target contact uses surface-relative velocity.
    start_cost = velocity.square().sum(dim=-1)*a
    target_cost = (velocity-target_velocity).square().sum(dim=-1)*b
    return (start_cost+target_cost).sum(dim=-1)

def impact_cost(env):
    # Continuous force cost; explicitly not a first-contact event.
    target = env.scene["feet_target_contact"].data.force
    start = env.scene["feet_ground_contact"].data.force
    return (target.square().sum(dim=(-1,-2))+start.square().sum(dim=(-1,-2)))/100000