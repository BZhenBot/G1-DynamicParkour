"""准备策略的当前可测起跳点/朝向，无未来、接触面身份或验收FSM。"""
import torch
from .course_scene import PlatformTargetProvider

def launch_goal_features(root_position,heading,middle_center,offset_m,preserve_landing_lane=False):
    delta=middle_center[:,:2]-root_position[:,:2]
    delta=delta.clone();delta[:,0]+=offset_m
    if preserve_landing_lane:delta[:,1]=0.
    c=heading.cos();s=heading.sin()
    return torch.stack((c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1],s,c),-1)

def measured_launch_goal_features(env,offset_m,preserve_landing_lane=False):
    r=env.scene['robot'];center,_,_=PlatformTargetProvider(env).current('middle')
    return launch_goal_features(r.data.root_link_pos_w,r.data.heading_w,center,offset_m,preserve_landing_lane)
