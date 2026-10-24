"""当前观测的左右反射PPO ensemble；没有时序动画或状态写入。"""
import torch

class SagittalReflection:
    def __init__(self,joint_names,device):
        names=list(joint_names);assert len(names)==29
        partner=[];sign=[]
        for name in names:
            other=name.replace('left_','right_',1) if name.startswith('left_') else name.replace('right_','left_',1) if name.startswith('right_') else name
            partner.append(names.index(other));sign.append(-1. if ('roll_joint' in name or 'yaw_joint' in name) else 1.)
        self.partner=torch.tensor(partner,device=device);self.sign=torch.tensor(sign,device=device)
        assert torch.equal(self.partner[self.partner],torch.arange(29,device=device))
        assert torch.equal(self.sign*self.sign[self.partner],torch.ones(29,device=device))
    def action(self,x):return x[:,self.partner]*self.sign
    def observation(self,x):
        if x.shape[-1]!=114:raise ValueError('固定V1 Actor114布局')
        y=x.clone();y[:,1]*=-1;y[:,3]*=-1;y[:,5]*=-1;y[:,7]*=-1
        for start in (9,38,67):y[:,start:start+29]=self.action(x[:,start:start+29])
        for i in (97,98,100,103):y[:,i]*=-1
        y[:,106:108]=x[:,[107,106]]
        force=x[:,108:114].reshape(-1,2,3)[:,[1,0],:].clone();force[:,:,1]*=-1;y[:,108:114]=force.flatten(1)
        return y