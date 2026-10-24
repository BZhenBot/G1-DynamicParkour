"""验证有质量的滑轨平台：真实 position actuator 驱动，不逐步改写位姿。"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def reference(t, amplitude, frequency, phase=math.pi / 2):
    omega = 2 * math.pi * frequency
    angle = omega * t + phase
    position = amplitude * math.sin(angle)
    velocity = amplitude * omega * math.cos(angle)
    acceleration = -amplitude * omega * omega * math.sin(angle)
    # PD 的速度/加速度前馈仅控制平台机械驱动，不提供给机器人 Actor。
    servo_target = position + 4000.0 / 100000.0 * velocity + 100.0 / 100000.0 * acceleration
    return position, velocity, servo_target


def run_case(device, amplitude, frequency):
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/platform_contact.xml"))
    model.opt.iterations = 10
    model.opt.ls_iterations = 10
    data = mujoco.MjData(model)
    platform_joint = model.joint("platform_y")
    payload_joint = model.joint("payload_free")
    platform_qpos = int(platform_joint.qposadr[0])
    payload_qpos = int(payload_joint.qposadr[0])
    actuator_id = model.actuator("platform_servo").id
    initial_position, initial_velocity, _ = reference(0, amplitude, frequency)
    data.qpos[platform_qpos] = initial_position
    data.qvel[int(platform_joint.dofadr[0])] = initial_velocity
    data.qpos[payload_qpos + 1] = initial_position
    mujoco.mj_forward(model, data)
    sim = None
    if device != "cpu-native":
        import torch
        from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
        sim = Simulation(num_envs=1, cfg=SimulationCfg(mujoco=MujocoCfg(timestep=0.002)), model=model, device=device)
        sim.data.qpos[:] = torch.as_tensor(data.qpos, device=device, dtype=torch.float32)
        sim.data.qvel[:] = torch.as_tensor(data.qvel, device=device, dtype=torch.float32)
        sim.forward()
    samples = []
    steps = round(10.0 / model.opt.timestep)
    for step in range(steps):
        t = step * model.opt.timestep
        _, _, servo_target = reference(t, amplitude, frequency)
        if sim is None:
            data.ctrl[actuator_id] = servo_target
            mujoco.mj_step(model, data)
        else:
            sim.data.ctrl[:, actuator_id] = servo_target
            sim.step()
        if step % 25 == 24:
            position, velocity, _ = reference((step+1) * model.opt.timestep, amplitude, frequency)
            if sim is None:
                qpos, qvel = data.qpos.copy(), data.qvel.copy()
                contact_count = int(data.ncon)
            else:
                qpos = sim.data.qpos[0].detach().cpu().numpy().copy()
                qvel = sim.data.qvel[0].detach().cpu().numpy().copy()
                contact_count = None
            samples.append({"t": (step+1)*model.opt.timestep,
                            "reference_y": position, "reference_velocity": velocity,
                            "platform_y": float(qpos[platform_qpos]),
                            "platform_velocity": float(qvel[int(platform_joint.dofadr[0])]),
                            "payload_y": float(qpos[payload_qpos+1]),
                            "payload_z": float(qpos[payload_qpos+2]), "native_contact_count": contact_count})
    tracking_error = max(abs(s["platform_y"]-s["reference_y"]) for s in samples)
    velocity_error = max(abs(s["platform_velocity"]-s["reference_velocity"]) for s in samples)
    relative_payload = max(abs(s["payload_y"]-s["platform_y"]) for s in samples)
    payload_excursion = max(s["payload_y"] for s in samples)-min(s["payload_y"] for s in samples)
    # 验证平台跟踪、有效支撑高度及靠真实摩擦被运载的位移。
    passed = tracking_error < 0.002 and velocity_error < 0.01 and relative_payload < 0.01
    passed = passed and min(s["payload_z"] for s in samples) > 0.24
    if amplitude:
        passed = passed and payload_excursion > 1.8 * amplitude
    return {"device": device, "amplitude_m": amplitude, "frequency_hz": frequency,
            "position_max_error_m": tracking_error, "velocity_max_error_mps": velocity_error,
            "relative_payload_max_m": relative_payload, "payload_excursion_m": payload_excursion,
            "step_state_overwrites": 0, "status": "PASS" if passed else "FAIL", "samples": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu-native")
    args = parser.parse_args()
    start = time.monotonic()
    results = [run_case(args.device, a, f) for a, f in ((0.0, 0.10), (0.05, 0.10), (0.10, 0.25))]
    artifact = Path(os.environ["G1_RUN_DIR"]) / "platform_physics.json"
    artifact.write_text(json.dumps({"type": "physical_contact_regression_not_rl", "cases": results,
        "elapsed_seconds": time.monotonic()-start}, indent=2) + "\n")
    for result in results:
        print(json.dumps({key: value for key, value in result.items() if key != "samples"}), flush=True)
    return int(any(result["status"] != "PASS" for result in results))

if __name__ == "__main__":
    sys.exit(main())