"""只读依赖检查；仅在 MuJoCo 已安装时运行 CPU 最小物理仿真。"""
import importlib
import importlib.util
import json
import platform
import sys
import traceback

print(json.dumps({"python": sys.executable, "version": sys.version,
                  "platform": platform.platform()}, ensure_ascii=False), flush=True)
modules = ("torch", "mujoco", "mjlab", "mujoco_warp", "warp", "rsl_rl")
found = {}
for name in modules:
    try:
        spec = importlib.util.find_spec(name)
        found[name] = spec is not None
        print(json.dumps({"module": name, "available": found[name],
                          "origin": spec.origin if spec else None}), flush=True)
    except Exception:
        found[name] = False
        traceback.print_exc()

if found["torch"]:
    try:
        torch = importlib.import_module("torch")
        available = torch.cuda.is_available()
        print(json.dumps({"torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
                          "cuda_available": available, "device_count": torch.cuda.device_count()}), flush=True)
        if available:
            x = torch.arange(16, device="cuda", dtype=torch.float32)
            value = (x * x).sum().item()
            torch.cuda.synchronize()
            assert value == 1240.0
            print(json.dumps({"gpu_compute": "PASS", "sum_squares": value,
                              "gpu_name": torch.cuda.get_device_name(0)}), flush=True)
    except Exception:
        print("GPU_COMPUTE=FAIL", flush=True)
        traceback.print_exc()

try:
    import mujoco
    print(json.dumps({"mujoco_version": mujoco.__version__}), flush=True)
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep="0.002" gravity="0 0 -9.81"/>
      <worldbody><geom type="plane" size="2 2 .1"/>
      <body pos="0 0 1"><freejoint/><geom type="sphere" size=".05" mass="1"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    assert 0.035 < float(data.qpos[2]) < 0.065
    assert data.ncon > 0
    print(json.dumps({"cpu_simulation": "PASS", "steps": 1000,
                      "time": data.time, "height": float(data.qpos[2]),
                      "contacts": data.ncon, "scope": "sphere only, not G1"}), flush=True)
except Exception:
    print("CPU_SIMULATION=FAIL", flush=True)
    traceback.print_exc()
    sys.exit(1)
