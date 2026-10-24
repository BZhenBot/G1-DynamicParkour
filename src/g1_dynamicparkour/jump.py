"""Current-state jump observations and physical event rewards; no robot-state writes."""
import math
import re
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE
import torch
from torch.nn.functional import one_hot
from g1_dynamicparkour import stage1,action_basis
from g1_dynamicparkour.batched_events import BatchedJumpDetector

class JumpState:
    def __init__(self,env):
        self.detector=BatchedJumpDetector(env.num_envs,env.device)
        self.event_counts=torch.zeros((env.num_envs,7),device=env.device,dtype=torch.long)
        self.last_step=-1

def state(env):
    if not hasattr(env,"jump_task_state"):
        env.jump_task_state=JumpState(env)
    return env.jump_task_state

def reset_state(env,env_ids):
    s=state(env)
    s.detector.reset(env_ids)
    s.event_counts[env_ids]=0

def episode_clock(env,duration=.5):
    # Known task clock; not a future platform trajectory or privileged contact flag.
    return (env.episode_length_buf.float()*env.step_dt/duration).clamp(0,1)[:,None]

def platform_relative_state(env):
    robot,target=env.scene["robot"],env.scene["target"]
    xy=stage1.target_xy(env)
    top=target.data.root_link_pos_w[:,2]+env.geometry.half_thickness
    position=torch.cat((xy,top[:,None]),dim=-1)-robot.data.root_link_pos_w
    target_velocity=torch.zeros_like(position)
    target_velocity[:,1]=target.data.joint_vel[:,0]
    velocity=target_velocity-robot.data.root_link_lin_vel_w
    h=robot.data.heading_w
    c,s=torch.cos(h),torch.sin(h)
    def heading_frame(vector):
        return torch.stack((c*vector[:,0]+s*vector[:,1],-s*vector[:,0]+c*vector[:,1],vector[:,2]),dim=-1)
    return torch.cat((heading_frame(position),heading_frame(velocity)),dim=-1)

def privileged_phase(env):
    return one_hot(state(env).detector.phase,num_classes=7).float()

def privileged_geometry_motion(env):
    action=env.action_manager.get_term("periodic_platform")
    theta=2*math.pi*env.geometry.frequency*action.t+action.phase
    constants=torch.tensor([env.geometry.gap,env.geometry.amplitude,env.geometry.frequency],device=env.device)
    return torch.cat((constants.expand(env.num_envs,3),theta.sin()[:,None],theta.cos()[:,None]),dim=-1)

def illegal_contact(env):
    illegal=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
    for name in ("nonfoot_ground","nonfoot_target"):
        illegal |= (torch.linalg.vector_norm(env.scene[name].data.force,dim=-1)>10).any(dim=-1)
    return illegal

def stable_on_target(env):
    robot,target=env.scene["robot"],env.scene["target"]
    base=stage1.state(env).stable
    vertical_slow=robot.data.root_link_lin_vel_w[:,2].abs()<=.1
    feet_z=robot.data.site_pos_w[:,stage1.state(env).foot_ids,2]
    top=target.data.root_link_pos_w[:,2]+env.geometry.half_thickness
    foot_height=feet_z-top[:,None]
    on_top=((foot_height>=-.01)&(foot_height<=.04)).all(dim=-1)
    return base&vertical_slow&on_top

def update_state(env):
    s=state(env)
    if s.last_step==env.common_step_counter:
        return torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
    s.last_step=env.common_step_counter
    stage1.update_state(env)
    start,target,_=stage1.contacts(env)
    robot=env.scene["robot"]
    fallen=(robot.data.root_link_pos_w[:,2]<.5)|(-robot.data.projected_gravity_b[:,2]<math.cos(math.radians(70)))
    events=s.detector.update(env.episode_length_buf.double()*env.step_dt,start,target,
        robot.data.root_link_pos_w[:,2],robot.data.root_link_lin_vel_w[:,2],
        stable_on_target(env),illegal_contact(env),fallen)
    s.event_counts+=events.long()
    env.extras["log"]["Metrics/jump/airborne_fraction"]=float((s.detector.phase==3).float().mean())
    env.extras["log"]["Metrics/jump/target_contact_fraction"]=float((s.event_counts[:,4]>0).float().mean())
    return torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)

def failure(env):
    return state(env).detector.phase==6

def success(env):
    return state(env).detector.phase==5

def airborne_event(env):
    return state(env).detector.events[:,3].float()

def target_event(env):
    return state(env).detector.events[:,4].float()

def success_event(env):
    return state(env).detector.events[:,5].float()

def failure_event(env):
    return state(env).detector.events[:,6].float()

def preparation_height(env,duration=.30,target_height=.70,std=.035):
    t=env.episode_length_buf.float()*env.step_dt
    z=env.scene["robot"].data.root_link_pos_w[:,2]-env.geometry.top_height
    phase=state(env).detector.phase
    return torch.exp(-(z-target_height).square()/std**2)*((t<duration)&(phase<=1))

def propulsion(env,start_time=.30,end_time=float("inf")):
    t=env.episode_length_buf.float()*env.step_dt
    phase=state(env).detector.phase
    vz=env.scene["robot"].data.root_link_lin_vel_w[:,2]
    return vz.clamp(0,1.5)*((t>=start_time)&(t<end_time)&(phase==1))

def flight_progress(env):
    phase=state(env).detector.phase
    return stage1.state(env).progress_rate*(phase==3)

def landed_position(env):
    phase=state(env).detector.phase
    distance=stage1.state(env).distance
    return torch.exp(-distance.square()/.2**2)*((phase==4)|(phase==5))

def landed_relative_velocity(env):
    relative=platform_relative_state(env)[:,3:]
    phase=state(env).detector.phase
    return relative.square().sum(dim=-1)*((phase==4)|(phase==5))

def target_stability(env):
    phase=state(env).detector.phase
    return stable_on_target(env).float()*((phase==4)|(phase==5))

def energy(env):
    return env.scene["robot"].data.actuator_force.square().sum(dim=-1)

def action_units(env):
    if hasattr(env,"jump_action_units"):
        return env.jump_action_units
    term=env.action_manager.get_term("joint_pos")
    original=[]
    for name in term.target_names:
        matches=[value for pattern,value in G1_ACTION_SCALE.items() if re.fullmatch(pattern,name)]
        assert len(matches)==1,(name,matches)
        original.append(matches[0])
    scale=torch.tensor(original,device=env.device)
    env.jump_action_units=term.scale/scale
    return env.jump_action_units


def scaled_previous_actions(env):
    # Preserve pretrained observation units after changing only joint-target range.
    return action_basis.map_actions(env,env.action_manager.action)*action_units(env)


def scaled_action_rate(env):
    difference=(action_basis.map_actions(env,env.action_manager.action)-action_basis.map_actions(env,env.action_manager.prev_action))*action_units(env)
    return difference.square().sum(dim=-1)

def source_wait(env,start_time=.5):
    # Task urgency until confirmed flight, including short takeoff/contact chatter.
    t=env.episode_length_buf.float()*env.step_dt
    return ((t>=start_time)&(state(env).detector.phase<=2)).float()

def progress_before_landing(env):
    # Current relative distance only; walking still fails the unchanged airborne detector.
    phase=state(env).detector.phase
    return stage1.state(env).progress_rate*((phase>=1)&(phase<=3))

def flight_foot_position_cost(env,include_landing=False,x_margin=.02):
    # Current two-foot sites outside the actual target landing rectangle; no future extrapolation.
    feet=env.scene["robot"].data.site_pos_w[:,stage1.state(env).foot_ids,:2]
    relative=(feet-stage1.target_xy(env)[:,None,:]).abs()
    if not math.isfinite(x_margin) or not .02 <= x_margin <= .10 or x_margin >= env.geometry.target_half_length:
        raise ValueError("Landing shaping x margin must be finite, in[.02,.10]m and smaller than target half-length")
    # Reward-only interior target; physical success/supervisor retain the original2cm margin.
    bounds=torch.tensor([env.geometry.target_half_length-x_margin,env.geometry.target_half_width-.02],device=env.device)
    outside=(relative-bounds).clamp_min(0)
    phase=state(env).detector.phase
    active=(phase==3)|((phase==4) if include_landing else torch.zeros_like(phase,dtype=torch.bool))
    return outside.square().sum(dim=-1).mean(dim=-1)*active


def flight_tilt_cost(env,include_landing=False):
    # Current projected gravity measures roll/pitch without constraining yaw or disabling dynamics.
    phase=state(env).detector.phase
    active=(phase==3)|((phase==4) if include_landing else torch.zeros_like(phase,dtype=torch.bool))
    return env.scene["robot"].data.projected_gravity_b[:,:2].square().sum(dim=-1)*active
def recovery_previous_actions(env):
    """Preserve original99 proprioceptive prefix using actual applied29 targets, not auxiliary coordinate."""
    return env.action_manager.get_term("joint_pos").raw_action*action_units(env)

def recovery_action_rate(env):
    term=env.action_manager.get_term("joint_pos")
    return ((term.raw_action-term.previous_joint_actions)*action_units(env)).square().sum(dim=-1)

def landing_height_deficit_cost(env,minimum_height=.68):
    """Continuous landing shaping, separate from unchanged physical fall threshold0.5m."""
    phase=state(env).detector.phase
    top=env.scene["target"].data.root_link_pos_w[:,2]+env.geometry.half_thickness
    height=env.scene["robot"].data.root_link_pos_w[:,2]-top
    return (minimum_height-height).clamp_min(0).square()*((phase==4)|(phase==5))

def landed_dual_support(env):
    """Intermediate two-foot contact progress; never substitutes continuous2s stability success."""
    robot=env.scene["robot"]
    phase=state(env).detector.phase
    source,target,_=stage1.contacts(env)
    feet=robot.data.site_pos_w[:,stage1.state(env).foot_ids]
    center=stage1.target_xy(env)
    bounds=torch.tensor([env.geometry.target_half_length-.02,env.geometry.target_half_width-.02],device=env.device)
    inside=((feet[:,:,:2]-center[:,None,:]).abs()<=bounds).all(dim=-1).all(dim=-1)
    safe=(robot.data.root_link_pos_w[:,2]>=.5)&~illegal_contact(env)
    return (((phase==4)|(phase==5))&target.all(dim=-1)&~source.any(dim=-1)&inside&safe).float()

def actor_unlabelled_foot_contacts(env):
    """Current per-foot support and signed log net force; no surface identity, FSM, or future state."""
    return torch.cat((stage1.feet_contact(env),stage1.feet_forces(env)),dim=-1)
