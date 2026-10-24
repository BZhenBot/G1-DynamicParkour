"""实际编译 G1 双平台，验证初始支撑/间隙/自由根/轨道驱动，非跳跃成功测试。"""
import json
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import sys
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import mujoco
import numpy as np
from mjlab.scene import Scene
from g1_dynamicparkour.scene import Geometry, make_scene_cfg, servo_reference


def main():
    output = Path(os.environ["G1_RUN_DIR"])
    parser=argparse.ArgumentParser()
    parser.add_argument("--robot-start-offset",type=float,default=0.0)
    parser.add_argument("--gap",type=float,default=.15)
    parser.add_argument("--amplitude",type=float,default=.05)
    parser.add_argument("--duration",type=float,default=.3)
    args=parser.parse_args()
    geometry = Geometry(robot_start_offset_x=args.robot_start_offset,gap=args.gap,amplitude=args.amplitude)
    first_cfg = make_scene_cfg(Geometry(top_height=.2))
    second_cfg = make_scene_cfg(geometry)
    assert abs(first_cfg.entities["robot"].init_state.pos[2]-second_cfg.entities["robot"].init_state.pos[2]-.2) < 1e-9
    scene = Scene(second_cfg, device="cpu")
    model = scene.compile()
    model.opt.timestep = .002
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    scene.write(output/"compiled_scene")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    # Native MuJoCo keyframes do not store mocap initial poses; copy configured reset poses.
    for name in ("start", "target"):
        body = model.body(name+"/mocap_base" if name == "start" else name+"/rail_base")
        mocap_id = int(model.body_mocapid[body.id])
        data.mocap_pos[mocap_id] = scene[name].cfg.init_state.pos
        data.mocap_quat[mocap_id] = scene[name].cfg.init_state.rot
    joint_id = model.joint("target/lateral_slide").id
    qa, va = int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])
    data.qpos[qa], data.qvel[va], _ = servo_reference(0,geometry.amplitude,geometry.frequency,platform_servo_kp=geometry.platform_servo_kp,platform_servo_kv=geometry.platform_servo_kv)
    mujoco.mj_forward(model, data)
    assert model.nu == 30 and model.nq == 37 and model.nv == 36
    initial_root_x=float(data.qpos[0])
    assert int(model.jnt_type[model.joint("robot/floating_base_joint").id]) == int(mujoco.mjtJoint.mjJNT_FREE)
    assert abs((geometry.target_center_x-geometry.target_half_length)-(geometry.start_center_x+geometry.start_half_length)-geometry.gap) < 1e-12
    foot_geoms = [model.geom(i).name for i in range(model.ngeom) if "_foot" in model.geom(i).name and "collision" in model.geom(i).name]
    assert len(foot_geoms) == 14
    assert all(model.geom_contype[model.geom(name).id] != 0 or model.geom_conaffinity[model.geom(name).id] != 0 for name in foot_geoms)
    servo_id = model.actuator("target/platform_servo").id
    contacts = {"start": 0, "target": 0, "illegal_start": 0}
    tracking = []
    for step in range(round(args.duration/model.opt.timestep)):
        t = step*model.opt.timestep
        _, _, data.ctrl[servo_id] = servo_reference(t, geometry.amplitude, geometry.frequency,platform_servo_kp=geometry.platform_servo_kp,platform_servo_kv=geometry.platform_servo_kv)
        mujoco.mj_step(model, data)
        desired, _, _ = servo_reference(data.time,geometry.amplitude,geometry.frequency,platform_servo_kp=geometry.platform_servo_kp,platform_servo_kv=geometry.platform_servo_kv)
        tracking.append(abs(float(data.qpos[qa])-desired))
        for contact in data.contact[:data.ncon]:
            names = [model.geom(int(gid)).name for gid in contact.geom]
            foot = any("_foot" in name and "collision" in name for name in names)
            if foot and any(name == "start/start_collision" for name in names):
                contacts["start"] += 1
            if foot and any(name == "target/target_collision" for name in names):
                contacts["target"] += 1
            if not foot and any(name.startswith("robot/") for name in names) and "start/start_collision" in names:
                contacts["illegal_start"] += 1
    assert contacts["start"] > 0 and contacts["target"] == 0 and contacts["illegal_start"] == 0, contacts
    assert max(tracking) < .002, max(tracking)
    assert np.isfinite(data.qpos).all()
    report = {"pass": True, "duration_s": float(data.time), "nq": model.nq, "nv": model.nv, "nu": model.nu,
        "geometry":asdict(geometry),"initial_root_x_m":initial_root_x,"gap_m": geometry.gap, "foot_collision_geoms": foot_geoms, "contacts": contacts,
        "maximum_platform_position_error_m": max(tracking), "root_height_m": float(data.qpos[2]),
        "robot_is_free_floating": True, "platform_driven_by_real_slide_actuator": True,
        "jump_policy_success": "待验证：该测试仅验证场景装配和初始物理接触"}
    (output/"scene_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
