"""Read-only 200Hz actual contact-frame normal forces for unchanged frozen PPO evaluation.

Uses the locally inspected mujoco_warp.contact_force API and the real manager's
metrics_manager.compute_substep hook. Does not change physics, rewards or Actor.
Landing normal impulse is the sum of compressive foot-target normal forces
integrated over [first >5N foot contact, first contact +.20s) at physics dt=.005s.
"""
import json
import os
from pathlib import Path
import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp
import evaluate_jump

class ContactInstrumentedEnv(evaluate_jump.JumpEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.num_envs!=1 or self.cfg.auto_reset:
            raise ValueError('Contact evaluation requires one original-source environment without auto reset')
        self.contact_output=Path(os.environ['G1_RUN_DIR'])
        self.physics_file=(self.contact_output/'physics_contact_samples.jsonl').open('a')
        self.contact_episode_records=[]
        diagnostics_flag=os.environ.get('G1_DETAILED_CONTACT_DIAGNOSTICS','0')
        if diagnostics_flag not in ('0','1'):
            raise ValueError('G1_DETAILED_CONTACT_DIAGNOSTICS must be0 or1')
        self.detailed_contact_diagnostics=diagnostics_flag=='1'
        self.contact_body_diagnostics={}
        self.nonfoot_primary_names={name:list(self.scene[name].primary_names)
                                   for name in ('nonfoot_ground','nonfoot_target')}
        self.local_physics_step=0
        self.control_contacts=[]
        self.metric_episode=None
        self.metric_stats=None
        mjm=self.sim.mj_model
        self.nonfoot_primary_objects={}
        if self.detailed_contact_diagnostics:
            for sensor_name,names in self.nonfoot_primary_names.items():
                match=self.scene[sensor_name].cfg.primary
                objtype=mujoco.mjtObj.mjOBJ_GEOM if match.mode=='geom' else mujoco.mjtObj.mjOBJ_BODY
                objects={}
                for name in names:
                    candidates=[name,f'{match.entity}/{name}'] if match.entity else [name]
                    ids=sorted(set(mujoco.mj_name2id(mjm,objtype,v) for v in candidates)-{-1})
                    assert len(ids)==1,(sensor_name,match.mode,name,ids)
                    object_id=ids[0]
                    body_id=int(mjm.geom_bodyid[object_id]) if match.mode=='geom' else object_id
                    objects[name]={'primary_object':name,'primary_match_mode':match.mode,
                                   'resolved_mujoco_name':mujoco.mj_id2name(mjm,objtype,object_id),
                                   'resolved_body':mujoco.mj_id2name(mjm,mujoco.mjtObj.mjOBJ_BODY,body_id)}
                self.nonfoot_primary_objects[sensor_name]=objects
        self.foot_body_names=self.scene['feet_target_contact'].primary_names
        assert len(self.foot_body_names)==2
        body_ids=[]
        for name in self.foot_body_names:
            matches=[mujoco.mj_name2id(mjm,mujoco.mjtObj.mjOBJ_BODY,n) for n in (name,'robot/'+name)]
            matches=sorted(set(i for i in matches if i>=0))
            assert len(matches)==1,(name,matches)
            body_ids.append(matches[0])
        target_ids=[i for i in range(mjm.ngeom) if (mujoco.mj_id2name(mjm,mujoco.mjtObj.mjOBJ_GEOM,i) or '').split('/')[-1]=='target_collision']
        assert len(target_ids)==1
        self.target_geom_id=target_ids[0]
        self.body_ids=body_ids
        free_ids=[i for i in range(mjm.njnt) if mjm.jnt_type[i]==mujoco.mjtJoint.mjJNT_FREE]
        assert len(free_ids)==1
        self.root_qadr=int(mjm.jnt_qposadr[free_ids[0]])
        self.geom_body=torch.tensor(mjm.geom_bodyid.copy(),device=self.device,dtype=torch.int32)
        self.foot_geom_counts=[int(((mjm.geom_bodyid==i)&(mjm.geom_contype>0)).sum()) for i in body_ids]
        assert all(n>0 for n in self.foot_geom_counts)
        self.contact_geom=wp.to_torch(self.sim.wp_data.contact.geom)
        self.contact_frame=wp.to_torch(self.sim.wp_data.contact.frame)
        self.contact_positions=wp.to_torch(self.sim.wp_data.contact.pos) if self.detailed_contact_diagnostics else None
        self.nacon=wp.to_torch(self.sim.wp_data.nacon)
        self.contact_indices=torch.arange(self.contact_geom.shape[0],device=self.device,dtype=torch.int32)
        self.contact_ids_wp=wp.from_torch(self.contact_indices,dtype=wp.int32)
        with wp.ScopedDevice(self.sim.wp_device):
            self.force_wp=wp.zeros(self.contact_geom.shape[0],dtype=wp.spatial_vector)
        self.force_tensor=wp.to_torch(self.force_wp)
        self.original_substep=self.metrics_manager.compute_substep
        self.metrics_manager.compute_substep=self.sample_contacts
        self.read_only_checks=0

    def reset(self,**kwargs):
        if hasattr(self,'local_physics_step'):
            self.flush_episode()
            self.local_physics_step=0
            self.control_contacts=[]
        return super().reset(**kwargs)

    def sample_contacts(self):
        self.original_substep()
        mjm=self.sim.mj_model
        qpos=self.sim.data.qpos.clone() if self.read_only_checks<40 else None
        qvel=self.sim.data.qvel.clone() if qpos is not None else None
        self.force_tensor.zero_()
        with wp.ScopedDevice(self.sim.wp_device):
            mjwarp.contact_force(self.sim.wp_model,self.sim.wp_data,self.contact_ids_wp,False,self.force_wp)
        valid=self.contact_indices<self.nacon[0]
        ids=self.contact_geom.long().clamp(min=0,max=len(self.geom_body)-1)
        body=self.geom_body[ids]
        target_first=(self.contact_geom[:,0]==self.target_geom_id)
        target_second=(self.contact_geom[:,1]==self.target_geom_id)
        local=self.force_tensor[:,:3]
        world=torch.bmm(local[:,None,:],self.contact_frame).squeeze(1)
        pack=[]
        for body_id in self.body_ids:
            foot_second=target_first&(body[:,1]==body_id)&valid
            foot_first=target_second&(body[:,0]==body_id)&valid
            match=foot_first|foot_second
            normal=(self.force_tensor[:,0].clamp(min=0)*match).sum()
            signed=foot_second.to(world.dtype)-foot_first.to(world.dtype)
            robot_force=(world*signed[:,None]).sum(dim=0)
            pack.extend((normal,torch.linalg.vector_norm(robot_force)))
        # Native sensor is a primary->secondary wrench, opposite force on the robot.
        robot_forces=[]
        for body_id in self.body_ids:
            signed=((target_first&(body[:,1]==body_id)&valid).to(world.dtype)
                   -(target_second&(body[:,0]==body_id)&valid).to(world.dtype))
            robot_forces.append((world*signed[:,None]).sum(dim=0))
        native=self.scene['feet_target_contact'].data.force[0]
        sensor_error=(torch.stack(robot_forces)+native).abs().max()
        pack.append(sensor_error)
        ground_peak=torch.linalg.vector_norm(self.scene['nonfoot_ground'].data.force[0],dim=-1).max()
        target_peak=torch.linalg.vector_norm(self.scene['nonfoot_target'].data.force[0],dim=-1).max()
        raw_root=self.sim.data.qpos[0,self.root_qadr:self.root_qadr+7]
        raw_tilt=torch.acos((1-2*(raw_root[4].square()+raw_root[5].square())).clamp(-1,1))*(180/torch.pi)
        pack.extend((ground_peak,target_peak,raw_root[2],raw_tilt))
        self.local_physics_step+=1
        if self.detailed_contact_diagnostics:
            # Read the same existing sensors used by the original illegal guard.
            # Capture at the physical substep; no new contact filter or state write.
            details=[]
            for sensor_name,names in self.nonfoot_primary_names.items():
                vectors=self.scene[sensor_name].data.force[0].detach().cpu().tolist()
                assert len(names)==len(vectors),(sensor_name,len(names),len(vectors))
                for name,vector in zip(names,vectors):
                    norm=sum(value*value for value in vector)**.5
                    if norm>1.:
                        details.append({'sensor':sensor_name,**self.nonfoot_primary_objects[sensor_name][name],
                                        'sensor_force_vector_n':vector,'netforce_magnitude_n':norm,
                                        'exceeds_original_illegal_threshold':norm>10.})
            pairs=[]
            critical={entry['resolved_mujoco_name'] for entry in details
                      if entry['primary_match_mode']=='geom' and entry['exceeds_original_illegal_threshold']}
            if critical:
                count=int(self.nacon[0])
                geoms=self.contact_geom[:count].detach().cpu().tolist()
                forces=self.force_tensor[:count,:3].detach().cpu().tolist()
                positions=self.contact_positions[:count].detach().cpu().tolist()
                for index,(geom_ids,force,position) in enumerate(zip(geoms,forces,positions)):
                    if min(geom_ids)<0 or max(geom_ids)>=mjm.ngeom:
                        continue
                    names=[mujoco.mj_id2name(mjm,mujoco.mjtObj.mjOBJ_GEOM,gid) for gid in geom_ids]
                    if critical.intersection(names) and force[0]>1.:
                        pairs.append({'contact_index':index,'geom_ids':geom_ids,'geom_names':names,
                                      'contact_position_w_m':position,'compressive_normal_force_n':force[0]})
            self.contact_body_diagnostics[self.local_physics_step]={'primaries':details,'collision_pairs':pairs}
        self.control_contacts.append((self.local_physics_step,torch.stack(pack)))
        if qpos is not None:
            assert torch.equal(qpos,self.sim.data.qpos) and torch.equal(qvel,self.sim.data.qvel), 'Instrumentation must not write physics state'
            self.read_only_checks+=1

    def step(self,action):
        self.control_contacts=[]
        self.contact_body_diagnostics={}
        result=super().step(action)
        assert len(self.control_contacts)==self.cfg.decimation
        episode=getattr(self,'evaluation_episode_index',None)
        seed=getattr(self,'evaluation_seed',None)
        assert episode is not None and seed is not None
        if self.metric_stats is None:
            self.metric_episode=episode
            self.metric_stats={'episode':episode,'seed':seed,'physics_samples':0,'first_target_contact_buffer_time_s':None,
                'peak_target_normal_force_n':0.,'normal_impulse_first_200ms_ns':0.,'impact_window_samples':0,
                'normal_impulse_entire_episode_ns':0.,'peak_target_netforce_n':0.,'maximum_native_netforce_crosscheck_error_n':0.,
                'maximum_nonfoot_ground_netforce_n':0.,'maximum_nonfoot_target_netforce_n':0.,
                'minimum_raw_root_height_m':float('inf'),'maximum_raw_root_tilt_deg':0.,
                'any_200hz_illegal_contact':False,'any_200hz_fall':False,
                'first_200hz_illegal_contact_buffer_time_s':None,'first_200hz_fall_state_time_s':None}
        assert self.metric_episode==episode and self.metric_stats['seed']==seed
        values=torch.stack([item[1] for item in self.control_contacts]).detach().cpu().tolist()
        for (index,_),value in zip(self.control_contacts,values):
            normal=[value[0],value[2]]; net=[value[1],value[3]]; error=value[4]
            assert error<.005,(index,error,'native contact sensor crosscheck')
            start=(index-1)*self.physics_dt; end=index*self.physics_dt
            stats=self.metric_stats
            if stats['first_target_contact_buffer_time_s'] is None and max(normal)>5:
                stats['first_target_contact_buffer_time_s']=start
            total=sum(normal)
            stats['physics_samples']+=1
            ground_peak,target_peak,height,tilt=value[5:9]
            illegal=max(ground_peak,target_peak)>10
            fallen=height<.5 or tilt>70
            stats['maximum_nonfoot_ground_netforce_n']=max(stats['maximum_nonfoot_ground_netforce_n'],ground_peak)
            stats['maximum_nonfoot_target_netforce_n']=max(stats['maximum_nonfoot_target_netforce_n'],target_peak)
            stats['minimum_raw_root_height_m']=min(stats['minimum_raw_root_height_m'],height)
            stats['maximum_raw_root_tilt_deg']=max(stats['maximum_raw_root_tilt_deg'],tilt)
            stats['any_200hz_illegal_contact']|=illegal
            stats['any_200hz_fall']|=fallen
            if illegal and stats['first_200hz_illegal_contact_buffer_time_s'] is None:
                stats['first_200hz_illegal_contact_buffer_time_s']=start
            if fallen and stats['first_200hz_fall_state_time_s'] is None:
                stats['first_200hz_fall_state_time_s']=end
            stats['peak_target_normal_force_n']=max(stats['peak_target_normal_force_n'],total)
            stats['peak_target_netforce_n']=max(stats['peak_target_netforce_n'],sum(net))
            stats['normal_impulse_entire_episode_ns']+=total*self.physics_dt
            stats['maximum_native_netforce_crosscheck_error_n']=max(stats['maximum_native_netforce_crosscheck_error_n'],error)
            first=stats['first_target_contact_buffer_time_s']
            in_window=first is not None and start+1e-12>=first and start<first+.20-1e-12
            if in_window:
                stats['normal_impulse_first_200ms_ns']+=total*self.physics_dt
                stats['impact_window_samples']+=1
            self.physics_file.write(json.dumps({'episode':episode,'seed':seed,'physics_step':index,
                'interval_start_s':start,'interval_end_s':end,'contact_buffer_time_s':start,
                'target_normal_forces_n':normal,'target_netforce_magnitudes_n':net,
                'native_netforce_crosscheck_error_n':error,'inside_200ms_impact_window':in_window,
                'nonfoot_ground_netforce_peak_n':ground_peak,'nonfoot_target_netforce_peak_n':target_peak,
                'raw_root_height_m':height,'raw_root_tilt_deg':tilt,'raw_root_state_time_s':end,
                'substep_illegal_contact':illegal,'substep_fallen':fallen,
                'nonfoot_primary_force_details':self.contact_body_diagnostics.get(index) if self.detailed_contact_diagnostics else None})+'\n')
        return result

    def flush_episode(self):
        if self.metric_stats is not None:
            self.contact_episode_records.append(self.metric_stats)
            self.metric_stats=None
            self.metric_episode=None

    def close(self):
        if hasattr(self,'physics_file') and not self.physics_file.closed:
            self.flush_episode()
            self.physics_file.close()
            report={'scope':'Actual contact-frame compressive normal forces of all robot-foot/target contacts at every physics substep; separate from scalar full-jump verdict. No changes to Actor, reward or state.',
                'physics_dt_s':self.physics_dt,'sampling_frequency_hz':1/self.physics_dt,
                'detailed_contact_diagnostics':self.detailed_contact_diagnostics,'nonfoot_sensor_primary_names':self.nonfoot_primary_names,
                'nonfoot_primary_objects':self.nonfoot_primary_objects,
                'detailed_contact_force_note':'Same primary sensor wrenches as original guard; body magnitudes>1N logged, original illegal threshold remains>10N. Optional read-only capture; no actor/reward/physics changes.',
                'impact_window_definition':'[first physics contact with any foot normal force >5N, first+.20s); .005s force rectangles; all matched contact normals including any edge contact',
                'whole_episode_impulse_includes_support_load':True,
                'contact_api':'mujoco_warp.contact_force(model,data,contact_ids,False,force)',
                'foot_body_names':self.foot_body_names,'foot_collision_geom_counts':self.foot_geom_counts,
                'target_geom_id':self.target_geom_id,'native_sensor_primary_secondary_sign':'opposite robot-contact force',
                'instrumentation_qpos_qvel_unchanged_checks':self.read_only_checks,
                'physical_criterion_resolution':'Same >10N illegal / height<.5m or tilt>70deg fall thresholds sampled at200Hz; formal success additionally excludes any such substep event',
                'root_state_note':'Raw free-root qpos is post-integration at interval end; contact forces refer to pre-integration interval start. Original manager scalar verdict remains separately recorded.',
                'episodes':self.contact_episode_records}
            (self.contact_output/'contact_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        super().close()

if __name__=='__main__':
    evaluate_jump.JumpEnv=ContactInstrumentedEnv
    evaluate_jump.main()
