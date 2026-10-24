"""三支持面真实 200Hz 接触仪表；只读 MuJoCo Warp 状态。"""
import json,os
from pathlib import Path
import mujoco,mujoco_warp as mjwarp,torch,warp as wp
from course_runtime import CourseEnv
from g1_dynamicparkour.course_scene import SURFACES,SENSOR_SUFFIX

class CourseContactEnv(CourseEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.num_envs!=1 or self.cfg.auto_reset: raise ValueError('物理评价必须单环境且不自动reset')
        self.out=Path(os.environ['G1_RUN_DIR']); self.physics_file=(self.out/'course_physics_samples.jsonl').open('a')
        self.local_physics_step=0; self.control_contacts=[]; self.read_only_checks=0; self.in_episode=False
        self.state_write_attempts=[]; self.state_write_checks=0
        for entity_name in ('robot','start','middle','target'):
            entity=self.scene[entity_name]
            for method in ('write_root_state_to_sim','write_root_link_pose_to_sim','write_root_link_velocity_to_sim',
                'write_root_link_velocity_b_to_sim','write_root_com_velocity_to_sim','write_joint_state_to_sim',
                'write_joint_position_to_sim','write_joint_velocity_to_sim'):
                original=getattr(entity,method)
                def guard(*a,_original=original,_name=entity_name+'.'+method,**k):
                    if self.in_episode:
                        self.state_write_attempts.append(_name); raise RuntimeError('禁止episode内状态写入: '+_name)
                    return _original(*a,**k)
                setattr(entity,method,guard)
        mjm=self.sim.mj_model
        self.body_ids=[mujoco.mj_name2id(mjm,mujoco.mjtObj.mjOBJ_BODY,'robot/'+n) for n in self.scene['feet_target_contact'].primary_names]
        if any(i<0 for i in self.body_ids):
            self.body_ids=[mujoco.mj_name2id(mjm,mujoco.mjtObj.mjOBJ_BODY,n) for n in self.scene['feet_target_contact'].primary_names]
        assert len(self.body_ids)==2 and all(i>=0 for i in self.body_ids)
        self.surface_ids={name:mujoco.mj_name2id(mjm,mujoco.mjtObj.mjOBJ_GEOM,path) for name,path in
            {'start':'start/start_collision','middle':'middle/middle_collision','end':'target/target_collision'}.items()}
        assert all(i>=0 for i in self.surface_ids.values())
        self.platform_collision_boxes={name:{'geom_type':int(mjm.geom_type[i]),
            'half_size_m':mjm.geom_size[i].tolist(),'local_position_m':mjm.geom_pos[i].tolist(),
            'local_quaternion_wxyz':mjm.geom_quat[i].tolist()} for name,i in self.surface_ids.items()}
        self.foot_counts=[int(((mjm.geom_bodyid==body)&(mjm.geom_contype>0)).sum()) for body in self.body_ids]
        assert self.foot_counts==[7,7]
        free=[i for i in range(mjm.njnt) if mjm.jnt_type[i]==mujoco.mjtJoint.mjJNT_FREE]; assert len(free)==1
        self.root_qadr=int(mjm.jnt_qposadr[free[0]])
        self.geom_body=torch.tensor(mjm.geom_bodyid.copy(),device=self.device,dtype=torch.int32)
        self.contact_geom=wp.to_torch(self.sim.wp_data.contact.geom); self.contact_frame=wp.to_torch(self.sim.wp_data.contact.frame)
        self.nacon=wp.to_torch(self.sim.wp_data.nacon)
        self.indices=torch.arange(self.contact_geom.shape[0],device=self.device,dtype=torch.int32)
        self.ids_wp=wp.from_torch(self.indices,dtype=wp.int32)
        with wp.ScopedDevice(self.sim.wp_device): self.force_wp=wp.zeros(self.contact_geom.shape[0],dtype=wp.spatial_vector)
        self.force_tensor=wp.to_torch(self.force_wp)
        self.original_substep=self.metrics_manager.compute_substep; self.metrics_manager.compute_substep=self.sample_contacts
    def reset(self,**kwargs):
        if hasattr(self,'in_episode'):
            self.in_episode=False; self.local_physics_step=0
        return super().reset(**kwargs)
    def sample_contacts(self):
        self.original_substep()
        before_q=self.sim.data.qpos.clone() if self.read_only_checks<40 else None
        before_v=self.sim.data.qvel.clone() if before_q is not None else None
        self.force_tensor.zero_()
        with wp.ScopedDevice(self.sim.wp_device): mjwarp.contact_force(self.sim.wp_model,self.sim.wp_data,self.ids_wp,False,self.force_wp)
        valid=self.indices<self.nacon[0]; ids=self.contact_geom.long().clamp(0,len(self.geom_body)-1); body=self.geom_body[ids]
        world=torch.bmm(self.force_tensor[:,:3,None].transpose(1,2),self.contact_frame).squeeze(1)
        pack=[]
        for name in SURFACES:
            first=self.contact_geom[:,0]==self.surface_ids[name]; second=self.contact_geom[:,1]==self.surface_ids[name]
            vectors=[]
            for foot_id in self.body_ids:
                a=first&(body[:,1]==foot_id)&valid; b=second&(body[:,0]==foot_id)&valid
                pack.append((self.force_tensor[:,0].clamp_min(0)*(a|b)).sum())
                vectors.append((world*(a.float()-b.float())[:,None]).sum(0))
            native=self.scene['feet_'+SENSOR_SUFFIX[name]+'_contact'].data.force[0]
            pack.append((torch.stack(vectors)+native).abs().max())
            pack.append(torch.linalg.vector_norm(self.scene['nonfoot_'+SENSOR_SUFFIX[name]].data.force[0],dim=-1).max())
            pack.extend(torch.linalg.vector_norm(native,dim=-1).unbind())
        raw=self.sim.data.qpos[0,self.root_qadr:self.root_qadr+7]
        tilt=torch.acos((1-2*(raw[4].square()+raw[5].square())).clamp(-1,1))*180/torch.pi
        pack.extend((raw[2],tilt)); self.local_physics_step+=1
        self.control_contacts.append((self.local_physics_step,torch.stack(pack)))
        if before_q is not None:
            assert torch.equal(before_q,self.sim.data.qpos) and torch.equal(before_v,self.sim.data.qvel)
            self.read_only_checks+=1
    def step(self,action):
        self.control_contacts=[]; result=super().step(action)
        assert len(self.control_contacts)==4
        for (index,_),values in zip(self.control_contacts,torch.stack([v for _,v in self.control_contacts]).cpu().tolist()):
            surfaces={}
            for i,name in enumerate(SURFACES):
                n0,n1,error,nonfoot,f0,f1=values[i*6:i*6+6]
                assert error<.005,(name,index,error)
                surfaces[name]={'normal_forces_n':[n0,n1],'native_netforce_n':[f0,f1],'native_crosscheck_error_n':error,'nonfoot_peak_n':nonfoot}
            row={'episode':self.evaluation_episode_index,'seed':self.evaluation_seed,'physics_step':index,
                 'interval_start_s':(index-1)*.005,'interval_end_s':index*.005,'surfaces':surfaces,
                 'root_height_m':values[-2],'tilt_deg':values[-1]}
            row['illegal_contact']=any(v['nonfoot_peak_n']>10 for v in surfaces.values())
            row['fallen']=row['root_height_m']<.5 or row['tilt_deg']>70
            self.physics_file.write(json.dumps(row)+'\n')
        return result
    def close(self):
        if hasattr(self,'physics_file') and not self.physics_file.closed:
            self.physics_file.close()
            (self.out/'course_contact_manifest.json').write_text(json.dumps({'physics_dt_s':.005,'sampling_hz':200,
                'foot_geom_counts':self.foot_counts,'platform_collision_boxes':self.platform_collision_boxes,'surface_geom_ids':self.surface_ids,'read_only_checks':self.read_only_checks,
                'state_write_attempts':self.state_write_attempts,'actual_resets':self.actual_resets,
                'root_contact_time_note':'接触为子步积分前buffer；root qpos为积分后；native合力反号交叉验证',
                'state_write_guard':'所有实体root/joint直接state setter在episode内拒绝；正常servo/PD控制不受影响'},indent=2))
        return super().close()