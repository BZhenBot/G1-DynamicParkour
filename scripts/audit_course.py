"""独立重算两段标量事件、三表面200Hz安全和平台轨迹；失败也保留。"""
import argparse,collections,hashlib,json,math,os,random,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
meta=json.loads((r/'metadata.json').read_text());assert meta['exit_code']==0 and meta['end_utc'] and meta['status']=='success'
for name,digest in meta['log_sha256'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest
sys.path.insert(0,str(r/'source/src'))
from g1_dynamicparkour.course_events import CourseEventDetector,CourseSample
from g1_dynamicparkour.course_scene import SURFACES
cfg=json.loads((r/'course_evaluation_config.json').read_text());evaluation=json.loads((r/'course_evaluation.json').read_text())
config_path=cfg['config']
assert hashlib.sha256((r/'source'/config_path).read_bytes()).hexdigest()==meta['source_sha256'][config_path]
assert json.loads((r/'source'/config_path).read_text())==cfg['settings'],'实际加载配置与源码快照不一致'
rows=evaluation['episodes'];assert len(set(x['seed'] for x in rows))==len(rows),'独立路线种子重复'
from formal_course_protocol import validate_formal
formal_receipt=validate_formal(cfg['settings'],len(rows),cfg['seed'],cfg['max_steps_diagnostic'])
if formal_receipt is not None:assert formal_receipt==cfg['formal_protocol_receipt']
assert all(x['episode']==i and x['seed']==cfg['seed']+i for i,x in enumerate(rows)),'独立种子/episode序列不一致'
for row in rows:
    assert row['phase_bin']==row['episode']%8
    expected=2*math.pi*(row['phase_bin']+random.Random(row['seed']).random())/8
    assert abs(row['platform_initial_phase_rad']-expected)<1e-6,'预登记初始相位不一致'
for relative,digest in meta['source_sha256'].items():
    if relative.endswith('.py'):assert hashlib.sha256((r/'source'/relative).read_bytes()).hexdigest()==digest
for role in ('jump','recovery'):
    assert hashlib.sha256(Path(cfg['settings'][role+'_checkpoint']).read_bytes()).hexdigest()==cfg['settings'][role+'_sha256']
for role in ('alignment','position','second_jump','end_recovery','settle_recovery'):
    if cfg['settings'].get(role+'_checkpoint'):
        assert hashlib.sha256(Path(cfg['settings'][role+'_checkpoint']).read_bytes()).hexdigest()==cfg['settings'][role+'_sha256']
contact_manifest=json.loads((r/'course_contact_manifest.json').read_text())
assert contact_manifest['foot_geom_counts']==[7,7] and contact_manifest['read_only_checks']==40
assert contact_manifest['state_write_attempts']==[] and cfg['physical_dimensions']==[37,36,30]
measured_geometry=None
boxes=contact_manifest.get('platform_collision_boxes')
if formal_receipt is not None and formal_receipt['protocol_id'].startswith('v22-'):assert boxes,'V2.2正式验收必须有实际碰撞box证据'
if boxes:
    for name,box in boxes.items():
        expected=[.4,.5,.1] if name=='start' else [cfg['course'][name+'_half_length'],cfg['course']['half_width'],.1]
        assert box['geom_type']==6 and max(abs(v-w) for v,w in zip(box['half_size_m'],expected))<1e-7,'实际碰撞box尺寸与配置不一致'
        assert max(abs(v) for v in box['local_position_m'])<1e-8
        assert max(abs(v-w) for v,w in zip(box['local_quaternion_wxyz'],[1.,0.,0.,0.]))<1e-8
    measured_geometry={'verified_compiled_box_sizes':True,'platform_collision_boxes':boxes,'gap1_range_m':[float('inf'),-float('inf')],'gap2_range_m':[float('inf'),-float('inf')],'samples':0}
physics={};force_first={};impact={};physical_fail={};cross=0.
for line in (r/'course_physics_samples.jsonl').read_text().splitlines():
    x=json.loads(line);e=x['episode'];bucket=physics.setdefault(e,[])
    assert x['physics_step']==len(bucket)+1 and abs(x['interval_end_s']-x['physics_step']*.005)<1e-12
    assert abs(x['interval_start_s']-(x['physics_step']-1)*.005)<1e-12
    bucket.append(x);physical_fail[e]=physical_fail.get(e,False) or x['illegal_contact'] or x['fallen']
    assert x['illegal_contact']==any(v['nonfoot_peak_n']>10 for v in x['surfaces'].values())
    assert x['fallen']==(x['root_height_m']<.5 or x['tilt_deg']>70)
    for name,v in x['surfaces'].items():
        cross=max(cross,v['native_crosscheck_error_n']);assert v['native_crosscheck_error_n']<.005
        key=(e,name)
        if max(v['normal_forces_n'])>5 and key not in force_first:force_first[key]=x['interval_start_s']
        first=force_first.get(key)
        if first is not None and x['interval_start_s']>=first-1e-12 and x['interval_start_s']<first+.2-1e-12:
            imp=impact.setdefault(key,{'samples':0,'impulse_ns':0.,'peak_normal_n':0.})
            imp['samples']+=1;imp['impulse_ns']+=sum(v['normal_forces_n'])*.005;imp['peak_normal_n']=max(imp['peak_normal_n'],sum(v['normal_forces_n']))
controls={};tracking={};detectors={};resets={};clock={};reward_sums={};time_checks={};second_starts={};prepare_feature_checks=0
for line in (r/'course_samples.jsonl').read_text().splitlines():
    x=json.loads(line);e=x['episode'];row=evaluation['episodes'][e]
    assert x['seed']==row['seed'];bucket=controls.setdefault(e,[]);assert abs(x['time_s']-(len(bucket)+1)*.02)<1e-12
    features=x.get('preparation_goal_features_before_step')
    if cfg['settings'].get('alignment_kind')=='prepared103' and (x['coordinator_mode']=='align' or (x['coordinator_mode']=='final_settle' and cfg['settings']['coordinator'].get('prepared_final_policy_blend_seconds',0)>0)):assert features is not None
    if features is not None:
        assert len(features)==4
        assert cfg['settings'].get('alignment_kind')=='prepared103' and (x['coordinator_mode']=='align' or (x['coordinator_mode']=='final_settle' and cfg['settings']['coordinator'].get('prepared_final_policy_blend_seconds',0)>0)) and bucket
        previous=bucket[-1];w,qx,qy,qz=previous['root_quaternion_wxyz']
        h=math.atan2(2*(w*qz+qx*qy),1-2*(qy*qy+qz*qz))
        dx=previous['surfaces_position']['middle'][0]+cfg['settings']['coordinator'].get('launch_point_from_middle_center_m',.2)-previous['root_position'][0]
        dy=previous['surfaces_position']['middle'][1]-previous['root_position'][1]
        if cfg['settings']['coordinator'].get('preparation_preserve_landing_lane',False):dy=0.
        expected=[math.cos(h)*dx+math.sin(h)*dy,-math.sin(h)*dx+math.cos(h)*dy,math.sin(h),math.cos(h)]
        assert max(abs(v-z) for v,z in zip(features,expected))<1e-5,'准备Actor103当前特征与真实前状态不一致'
        if cfg['settings']['coordinator'].get('consistent_feedback_command',False):
            control=cfg['settings']['coordinator'];gain=control.get('position_feedback_gain',.8);maximum=control.get('preparation_maximum_linear_speed_mps',.1)
            vx=max(-maximum,min(maximum,dx*gain));vy=max(-maximum,min(maximum,dy*gain))
            max_yaw=control.get('preparation_maximum_turn_speed_radps',.5);error=math.atan2(math.sin(-h),math.cos(-h))
            expected_command=[math.cos(h)*vx+math.sin(h)*vy,-math.sin(h)*vx+math.cos(h)*vy,max(-max_yaw,min(max_yaw,error*1.5))]
            if math.hypot(dx,dy)<=.04 and abs(error)<=.1:expected_command=[0.,0.,0.]
            observed_command=x['preparation_body_velocity_command_before_step'];assert len(observed_command)==3
            assert max(abs(v-z) for v,z in zip(expected_command,observed_command))<1e-5,'准备Actor103速度命令未对应真实当前目标反馈'
        prepare_feature_checks+=1
    assert x['actual_reset_count']==resets.setdefault(e,x['actual_reset_count'])
    assert x['platform_phase_rad']==row['platform_initial_phase_rad'],'episode内平台相位被改变'
    proxy_limit=cfg['settings'].get('coordinator',{}).get('handoff_maximum_backward_capture_offset_m')
    if proxy_limit is not None and cfg.get('gravity_world_mps2') and x['coordinator_transition']=='recover_end':
        assert bucket
        previous=bucket[-1]
        root=torch.tensor(previous['root_position'],dtype=torch.float32)
        center=torch.tensor(previous['surfaces_position']['end'],dtype=torch.float32)
        vx=previous['root_velocity'][0]-previous['surfaces_velocity']['end'][0]
        feet=torch.tensor(x['public_feet6_before_step'],dtype=torch.float32).reshape(2,3)
        offset=feet[:,0].mean()-(root[0]-center[0])-vx/torch.sqrt(abs(cfg['gravity_world_mps2'][2])/(root[2]-center[2]).clamp_min(.1))
        assert torch.isfinite(offset) and offset.le(proxy_limit+1e-6),('current_handoff_capture_proxy_mismatch',e,float(offset))
    contract=cfg['settings'].get('coordinator',{}).get('handoff_velocity_contract')
    if contract and x['coordinator_transition']=='recover_end':
        actor=torch.tensor(x['public_actor114_before_step'],dtype=torch.float32)
        assert torch.linalg.vector_norm(actor[102:104]).le(contract['horizontal_speed_mps'])
        assert actor[104].abs().le(contract['vertical_speed_mps'])
        assert torch.linalg.vector_norm(actor[3:6]).le(contract['angular_speed_radps'])
    assert x['platform_clock_s']>clock.get(e,0.);clock[e]=x['platform_clock_s']
    assert abs(x['platform_clock_s']-x['time_s'])<.002
    if measured_geometry is not None:
        positions=x['surfaces_position']
        for i,(left,right) in enumerate((('start','middle'),('middle','end')),1):
            gap=(positions[right][0]-boxes[right]['half_size_m'][0])-(positions[left][0]+boxes[left]['half_size_m'][0])
            assert abs(gap-cfg['course']['gap'+str(i)])<1e-6,'实际box边缘间隙不匹配'
            bounds=measured_geometry['gap'+str(i)+'_range_m'];bounds[0]=min(bounds[0],gap);bounds[1]=max(bounds[1],gap)
        measured_geometry['samples']+=1
    feet=torch.tensor(x['feet_position'],dtype=torch.float32);rootv=torch.tensor(x['root_velocity'],dtype=torch.float32)
    supports={}
    for name in SURFACES:
        f=torch.linalg.vector_norm(torch.tensor(x['surface_forces_n'][name],dtype=torch.float32),dim=-1)
        supports[name]=f.gt(5).tolist();assert supports[name]==x['supports'][name]
    stable={}
    for name in SURFACES:
        half_x=.4 if name=='start' else cfg['course'][name+'_half_length']
        half=torch.tensor([half_x,cfg['course']['half_width']],dtype=torch.float32)
        relative=feet-torch.tensor(x['surfaces_position'][name],dtype=torch.float32)
        velocity=rootv-torch.tensor(x['surfaces_velocity'][name],dtype=torch.float32)
        stable[name]=all(supports[name]) and not any(any(supports[k]) for k in SURFACES if k!=name) and bool((relative[:,:2].abs()<=half-.02).all()) and bool(((relative[:,2]>=-.01)&(relative[:,2]<=.04)).all()) and x['tilt_deg']<=15 and float(torch.linalg.vector_norm(velocity[:2]))<=.15 and abs(float(velocity[2]))<=.1 and x['angular_speed_radps']<=.5 and not x['illegal_contact'] and not x['fallen']
    assert stable==x['stable'],(e,x['time_s'],'stable')
    d=detectors.setdefault(e,CourseEventDetector());d.update(CourseSample(x['time_s'],supports,x['root_position'][2],x['root_velocity'][2],stable,x['illegal_contact'],x['fallen']))
    assert d.success==x['detector_success'] and d.failed==x['detector_failed'] and d.leg==x['detector_leg']
    if cfg['settings'].get('time_budget_mode')=='per_skill_with_preparation':
        from g1_dynamicparkour.course_time_budget import time_failure
        first_stable=next((v['time_s'] for v in d.events if v['leg_id']==0 and v['phase']=='stable_success'),None)
        second_start=x['coordinator_skill_start_s'] if x['coordinator_leg']==1 else None
        if second_start is not None:
            assert second_start==second_starts.setdefault(e,second_start), '第二技能clock不能再次重置'
        timed=time_failure(x['time_s'],cfg['settings'],first_stable,second_start)
        time_checks[e]=time_checks.get(e,True) and timed is None
    assert x['action'][29]==0. and x['joint_normalized_action']==x['action'][:29]
    assert max(abs(v) for v in x['joint_normalized_action'])<=1.
    theta=2*math.pi*cfg['course']['frequency']*x['platform_clock_s']+x['platform_phase_rad']
    pe=abs(x['surfaces_position']['end'][1]-cfg['course']['amplitude']*math.sin(theta))
    ve=abs(x['surfaces_velocity']['end'][1]-cfg['course']['amplitude']*2*math.pi*cfg['course']['frequency']*math.cos(theta))
    track=tracking.setdefault(e,{'max_position_error_m':0.,'max_velocity_error_mps':0.})
    track['max_position_error_m']=max(track['max_position_error_m'],pe);track['max_velocity_error_mps']=max(track['max_velocity_error_mps'],ve)
    reward_sums[e]=reward_sums.get(e,0.)+x['reward'];bucket.append(x)
results=[]
for row in evaluation['episodes']:
    e=row['episode'];d=detectors[e];assert d.events==row['events']
    assert len(controls[e])==row['control_steps'] and len(physics[e])==4*row['control_steps'] and row['mid_episode_resets']==0
    assert abs(sum(row['reward_terms'].values())-reward_sums[e])<.001
    track=tracking[e];tracking_pass=track['max_position_error_m']<.002 and track['max_velocity_error_mps']<.01
    complete=d.success and not physical_fail[e] and not row['failure_reason']
    assert complete==row['success'],(e,'full course scalar/physics verdict mismatch')
    airborne=[v for v in d.events if v['phase']=='airborne']
    if complete:assert len(airborne)==2 and sum(v['phase']=='stable_success' for v in d.events)==2
    # 离线核验包含AIR事件的同一次完整连续物理腾空，保留旧截断统计供对照。
    from course_flight_audit import verify_flight
    flight_checks=[verify_flight(physics[e],event,d.events,row['duration_s']) for event in airborne]
    physical_flight_pass=len(flight_checks)==2 and all(v['verified_200hz'] for v in flight_checks)
    landings={}
    for name in ('middle','end'):
        key=(e,name);v=impact.get(key);landings[name]=None if v is None else dict(v,window_complete=v['samples']==40)
        if complete:assert v and v['samples']==40
    results.append({'episode':e,'seed':row['seed'],'scalar_success':complete,'qualified_success':complete and tracking_pass and physical_flight_pass and time_checks.get(e,True),
        'time_budget_pass':time_checks.get(e,True),'second_skill_start_s':second_starts.get(e),
        'physical_failure_200hz':physical_fail[e],'two_airborne_events_verified_200hz':physical_flight_pass,
        'airborne_200hz_checks':flight_checks,'tracking_pass':tracking_pass,'tracking':track,'airborne_events':airborne,'landings':landings,
        'physics_samples':len(physics[e]),'failure_reason':row['failure_reason']})
report={'audit_pass':True,'scope':'独立重算每段稳定/事件、200Hz三表面安全及连续AIR、两着陆窗口、平台轨迹、reset/时间/动作/权重/日志；audit_pass表示证据一致，qualified_success必须额外满足两个AIR所在完整腾空各16连续物理无支持样本和3cm物理高度增加，旧事件前截断统计保留；不以reward判成功',
    'run':str(r),'episodes':results,'qualified_successes':sum(v['qualified_success'] for v in results),'episodes_count':len(results),'independent_unique_episode_seeds':len(set(v['seed'] for v in results)),
    'input_config_matches_source_snapshot':True,'phase_bin_counts':dict(collections.Counter(v['phase_bin'] for v in rows)),
    'seeded_initial_phase_and_mid_episode_phase_constancy_checked':True,
    'qualified_success_rate':sum(v['qualified_success'] for v in results)/len(results),'maximum_native_crosscheck_error_n':cross,
    'input_sha256':{name:hashlib.sha256((r/name).read_bytes()).hexdigest() for name in ('metadata.json','course_samples.jsonl','course_physics_samples.jsonl','course_evaluation.json')},
    'formal_protocol_receipt':formal_receipt,
    'preparation_actor103_current_feature_checks':prepare_feature_checks,
    'handoff_capture_proxy_check_applicable':cfg['settings'].get('coordinator',{}).get('handoff_maximum_backward_capture_offset_m') is not None and bool(cfg.get('gravity_world_mps2')),
    'formal_acceptance_met':cfg['task']=='two_real_jumps_course' and len(results)>=100 and sum(v['qualified_success'] for v in results)/len(results)>=.7}
if measured_geometry is not None:
    report['measured_collision_geometry']=measured_geometry
    report['input_sha256']['course_contact_manifest.json']=hashlib.sha256((r/'course_contact_manifest.json').read_bytes()).hexdigest()
out=Path(os.environ['G1_RUN_DIR']);(out/'course_independent_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('episodes','input_sha256')},ensure_ascii=False))