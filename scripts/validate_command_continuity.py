"""仅命令层合成闸门；不冒充机器人任务或视频流畅性通过。"""
import json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from g1_dynamicparkour.command_continuity import JointCommandContinuity
raw=torch.linspace(-1,1,29).reshape(1,29);original=raw.clone()
plain=JointCommandContinuity(.02,{})
for mode,leg,sign in [('jump',0,1),('recovery',0,-1),('align',0,1),('final_settle',0,-1),('jump',1,1)]:
    command=raw*sign;assert torch.equal(plain(command,mode,leg),command)
f=JointCommandContinuity(.02,{'command_switch_blend_seconds':.06,'preparation_command_ema_alpha':.5})
first=f(raw,'recovery',0);second=f(-raw,'align',0)
assert torch.allclose(second,raw*(2/3),atol=1e-6)
assert torch.equal(raw,original)
trace=[]
for i in range(200):trace.append(f(raw*((-1.)**i),'align',0))
x=torch.stack(trace[-100:]);variance=float(x.var(dim=0).mean());raw_variance=float(raw.square().mean())
assert variance<raw_variance*.2 and x.abs().max()<=1
f.reset();assert torch.equal(f(-raw,'jump',0),-raw)
for settings in ({'command_switch_blend_seconds':.13},{'preparation_command_ema_alpha':0},{'preparation_command_ema_alpha':float('nan')}):
    try:JointCommandContinuity(.02,settings);raise AssertionError('非法设置未拒绝')
    except ValueError:pass
report={'pass':True,'disabled_identity':True,'current_and_past_commands_only':True,'no_input_mutation':True,'reset_clears_cross_episode_history':True,'synthetic_25hz_alternation_variance_ratio':variance/raw_variance,'scope':'合成控制命令测试，无物理episode，不能据此宣称动作流畅性达标'}
(Path(os.environ['G1_RUN_DIR'])/'command_continuity_gate.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
