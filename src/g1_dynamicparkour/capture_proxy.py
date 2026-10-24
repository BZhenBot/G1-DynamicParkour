"""仅用当前root link/速度/足FK的LIPM近似；root link为CoM代理，非真实可捕获性判定。"""
import torch

def backward_capture_offset(root_x,root_vx,height,mean_foot_x,gravity):
    omega=torch.sqrt(gravity/height.clamp_min(.1))
    return mean_foot_x-root_x-root_vx/omega
