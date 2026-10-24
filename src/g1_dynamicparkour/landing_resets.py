"""Reset-only landing-state curriculum; recovery-origin outcomes are never full jumps."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from g1_dynamicparkour import jump,stage1
from g1_dynamicparkour.scene import Geometry

class LandingResetBank:
    def __init__(self,dataset):
        self.path=Path(dataset).resolve()
        manifest=json.loads(self.path.read_text())
        raw_geometry=json.loads((self.path.parent/"evaluation_config.json").read_text())["geometry"]
        # New explicitly configurable drive fields default to the exact legacy gains.
        # This normalizes historical manifests; immutable raw files remain untouched.
        self.geometry=asdict(Geometry(**raw_geometry))
        self.records=[]
        self.hashes=[]
        for entry in manifest["snapshots"]:
            if entry["event"]!="first_dual_target_after_airborne":
                continue
            path=(self.path.parent/entry["filename"]).resolve()
            if path.parent!=self.path.parent:
                raise ValueError("Snapshot must remain inside its recorded dataset")
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=entry["sha256"]:
                raise ValueError("Recorded actual landing snapshot hash mismatch")
            record=json.loads(path.read_text())
            if record["training_phase"]!=4 or record["training_event_counts"][3]<1 or not all(record["target_contact"]) or any(record["source_contact"]):
                raise ValueError("Curriculum bank requires actual airborne-then-dual-target contact snapshots")
            self.records.append(record)
            self.hashes.append(digest)
        if not self.records:
            raise ValueError("No genuine dual-target landing states")

    def restore(self,env,env_ids,indices):
        """Called only after normal Manager reset, before forward/sense/observation."""
        records=[self.records[index] for index in indices.detach().cpu().tolist()]
        if list(env.scene["robot"].joint_names)!=records[0]["joint_names"]:
            raise ValueError("Snapshot joint ordering mismatch")
        def values(name,dtype=torch.float32):
            return torch.tensor([record[name] for record in records],device=env.device,dtype=dtype)
        robot,platform=env.scene["robot"],env.scene["target"]
        root=values("root_state_world_link_13")
        # Dataset was collected from env0 at observed world origin0; assert this
        # against its fixed target geometry before permitting batched translation.
        if asdict(env.geometry)!=self.geometry:
            raise ValueError("Recovery bank must match recorded physical geometry, motion and drive gains exactly")
        if not torch.equal(env.scene.env_origins[0],torch.zeros(3,device=env.device)):
            raise ValueError("Reference dataset world origin must be checked before translation")
        root[:,:3]+=env.scene.env_origins[env_ids]
        robot.write_root_state_to_sim(root,env_ids=env_ids)
        robot.write_joint_state_to_sim(values("joint_position"),values("joint_velocity"),env_ids=env_ids)
        platform.write_joint_state_to_sim(values("platform_joint_position"),values("platform_joint_velocity"),env_ids=env_ids)
        env.sim.data.qacc_warmstart[env_ids]=values("qacc_warmstart")
        term=env.action_manager.get_term("joint_pos")
        if env.action_manager.total_action_dim==30:
            env.action_manager.action[env_ids]=values("sampled_actor_action")
            env.action_manager.prev_action[env_ids]=values("previous_sampled_actor_action")
            term.previous_joint_actions[env_ids]=values("previous_actual_joint_action")
            term.last_recovery_coordinate[env_ids]=values("sampled_actor_action")[:,29:30]
        elif env.action_manager.total_action_dim==29 and not hasattr(term,"basis_matrix"):
            # Direct coordinates equal actual29 joint commands. Retain physical units
            # and the real applied-history prefix from the recorded30-coordinate run.
            env.action_manager.action[env_ids]=values("raw_actual_joint_action")
            env.action_manager.prev_action[env_ids]=values("previous_actual_joint_action")
        else:
            raise ValueError("Landing bank supports only explicitly validated direct29/recovery30 adapters")
        term._raw_actions[env_ids]=values("raw_actual_joint_action")
        term._processed_actions[env_ids]=values("robot_joint_position_target")
        robot.set_joint_position_target(values("robot_joint_position_target"),env_ids=env_ids)
        periodic=env.action_manager.get_term("periodic_platform")
        periodic.t[env_ids]=values("platform_clock_s")
        periodic.phase[env_ids]=values("platform_phase_rad")
        recorded_time=values("episode_control_step",torch.float64)*values("control_dt_s",torch.float64)
        current_step=(recorded_time/env.step_dt).round().long()
        if not torch.allclose(current_step.double()*env.step_dt,recorded_time,atol=1e-7,rtol=0):
            raise ValueError("Recorded physical clock must be representable at the candidate control period")
        env.episode_length_buf[env_ids]=current_step
        state=jump.state(env)
        state.detector.phase[env_ids]=4
        state.detector.last_time[env_ids]=env.episode_length_buf[env_ids].double()*env.step_dt
        state.detector.events[env_ids]=False
        # Inherited historical phase is privileged curriculum context; no inherited
        # takeoff/airborne/landing event rewards or completed episode counts.
        state.event_counts[env_ids]=0
        state.last_step=env.common_step_counter-1
        stage1.state(env).control_step=env.common_step_counter-1
        return env.episode_length_buf[env_ids].clone()
