"""Read-only collection of genuine landing states during unchanged frozen source-to-target rollouts.

No writes to robot state, no teacher actions, and no recovery success claim.
Snapshots are potential reset-curriculum inputs; exact warmstart replay is unverified.
"""
import hashlib
import json
import os
from pathlib import Path
import torch
import evaluate_jump
from g1_dynamicparkour import jump,stage1

class CaptureEnv(evaluate_jump.JumpEnv):
    collect_stable_states=False
    def __init__(self,*args,**kwargs):
        self.capture_episode=-1
        self.captured=set()
        self.capture_records=[]
        super().__init__(*args,**kwargs)
        if self.num_envs!=1 or self.cfg.auto_reset:
            raise ValueError("Collector requires one explicitly reset environment")

    def reset(self,**kwargs):
        result=super().reset(**kwargs)
        self.capture_episode+=1
        self.captured=set()
        return result

    def step(self,action):
        result=super().step(action)
        state=jump.state(self)
        source,target,_=stage1.contacts(self)
        qualified=bool(state.event_counts[0,3]>0) and int(state.detector.phase[0])==4
        qualified=qualified and not bool(source[0].any()) and not bool(jump.illegal_contact(self)[0])
        if not qualified:
            return result
        events=[]
        if bool(target[0].any()):
            events.append("first_target_after_airborne")
        if bool(target[0].all()):
            events.append("first_dual_target_after_airborne")
        if self.collect_stable_states and bool(jump.stable_on_target(self)[0]):
            events.append("first_current_stable_target_after_airborne")
        robot,platform=self.scene["robot"],self.scene["target"]
        joint=self.action_manager.get_term("joint_pos")
        periodic=self.action_manager.get_term("periodic_platform")
        def values(x):
            return x[0].detach().cpu().tolist()
        for event in events:
            if event in self.captured:
                continue
            self.captured.add(event)
            root=torch.cat((robot.data.root_link_pos_w,robot.data.root_link_quat_w,
                robot.data.root_link_lin_vel_w,robot.data.root_link_ang_vel_w),dim=-1)
            record={"episode":self.capture_episode,"collector_reset_sequence":self.capture_episode,"evaluation_episode":getattr(self,"evaluation_episode_index",None),"evaluation_seed":getattr(self,"evaluation_seed",None),"event":event,
                "episode_control_step":int(self.episode_length_buf[0]),"control_dt_s":self.step_dt,
                "root_state_world_link_13":values(root),"joint_names":list(robot.joint_names),
                "joint_position":values(robot.data.joint_pos),"joint_velocity":values(robot.data.joint_vel),
                "qpos":values(self.sim.data.qpos),"qvel":values(self.sim.data.qvel),
                "ctrl":values(self.sim.data.ctrl),"qacc_warmstart":values(self.sim.data.qacc_warmstart),
                "robot_joint_position_target":values(joint._processed_actions),
                "raw_actual_joint_action":values(joint.raw_action),"previous_actual_joint_action":values(joint.previous_joint_actions),
                "sampled_actor_action":values(self.action_manager.action),"previous_sampled_actor_action":values(self.action_manager.prev_action),
                "actor_observation":values(self.obs_buf["actor"]),"critic_observation":values(self.obs_buf["critic"]),
                "platform_joint_position":values(platform.data.joint_pos),"platform_joint_velocity":values(platform.data.joint_vel),
                "platform_clock_s":float(periodic.t[0]),"platform_phase_rad":float(periodic.phase[0]),
                "training_phase":int(state.detector.phase[0]),"training_event_counts":values(state.event_counts),
                "source_contact":values(source),"target_contact":values(target),
                "scope":"Actual frozen PPO state after a genuine flight; prospective reset data only, not recovery success. No robot writes during rollout. Contact/FSM lag .005s retained."}
            filename=f'landing_state_ep{self.capture_episode}_{event}.json'
            path=Path(os.environ["G1_RUN_DIR"])/filename
            path.write_text(json.dumps(record,indent=2))
            self.capture_records.append({"filename":filename,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "episode":self.capture_episode,"evaluation_episode":record["evaluation_episode"],"evaluation_seed":record["evaluation_seed"],"event":event,"time_s":record["episode_control_step"]*self.step_dt})
        return result

    def close(self):
        directory=Path(os.environ["G1_RUN_DIR"])
        report={"scope":"Read-only actual landing-state dataset from unchanged physical frozen PPO, not a reset/recovery success experiment",
            "snapshots":self.capture_records,"exact_warmstart_replay":"unverified", "reset_curriculum":"This new dataset has not been used for training at collection time","collect_current_stable_states":self.collect_stable_states}
        (directory/"landing_state_dataset.json").write_text(json.dumps(report,indent=2))
        super().close()

if __name__=="__main__":
    import argparse
    import sys
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--collect-stable-landing",action="store_true")
    options,remaining=parser.parse_known_args()
    CaptureEnv.collect_stable_states=options.collect_stable_landing
    sys.argv=[sys.argv[0]]+remaining
    evaluate_jump.JumpEnv=CaptureEnv
    evaluate_jump.main()
