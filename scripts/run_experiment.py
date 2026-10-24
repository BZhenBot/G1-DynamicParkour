"""WSL 独立运行证据封装；不生成或推断机器人指标。"""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def git(root, *args):
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    return {"command": ["git", "-C", str(root), *args], "exit_code": p.returncode,
            "stdout": p.stdout, "stderr": p.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--kind", choices=("audit", "train", "eval", "replay"), default="audit")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not re.fullmatch(r"[a-zA-Z0-9_-]+", args.label):
        parser.error("需要合法 label 和 -- 后的执行命令")
    if args.timeout <= 0:
        parser.error("timeout 必须大于 0")
    if args.kind in ("train", "eval") and (args.seed is None or args.config is None):
        parser.error("train/eval 必须显式记录 seed 与 config")
    if args.config and not args.config.is_file():
        parser.error("config 必须为已存在的文件")
    if os.name != "posix":
        parser.error("所有运行必须在 WSL/Linux 执行")
    root = Path(__file__).resolve().parents[1]
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + args.label + "-" + uuid.uuid4().hex[:8]
    run_dir = root / "experiments" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    versions = {}
    for name in ("torch", "mujoco", "mjlab", "mujoco-warp", "warp-lang", "rsl-rl-lib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    commit = git(root, "rev-parse", "--verify", "HEAD")
    status = git(root, "status", "--porcelain=v1", "--untracked-files=all")
    diff = git(root, "diff", "HEAD") if commit["exit_code"] == 0 else git(root, "diff")
    (run_dir / "git.diff").write_text(diff["stdout"], encoding="utf-8")
    sources = []
    for entry in ("AGENTS.md", "README.md", ".gitignore", "scripts", "configs", "docs", "reports", "src", "tests", "assets", "pyproject.toml"):
        path = root / entry
        if path.is_file():
            sources.append(path)
        elif path.is_dir():
            sources.extend(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    hashes = {}
    for path in sorted(sources):
        relative = path.relative_to(root)
        target = run_dir / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    config_sha = None
    if args.config:
        target = run_dir / ("config" + args.config.suffix)
        shutil.copyfile(args.config, target)
        config_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    metadata = {
        "run_id": run_id, "kind": args.kind, "command_argv": command,
        "cwd": str(root), "seed": args.seed, "artifact_dir": str(run_dir),
        "config_path": str(args.config.resolve()) if args.config else None,
        "config_sha256": config_sha, "hyperparameters": "config 原件" if args.config else "不适用：审计无 RL 超参数",
        "git_commit": commit["stdout"].strip() if commit["exit_code"] == 0 else None,
        "git_commit_probe": commit, "git_status": status, "git_diff_probe_exit": diff["exit_code"],
        "source_sha256": hashes, "runner_python": sys.executable,
        "python_version": sys.version, "platform": platform.platform(), "package_versions": versions,
        "environment": {name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "OMP_NUM_THREADS", "PYTHONHASHSEED")},
        "seed_handling": "仅记录 seed；任务程序必须显式使用相同 seed，本封装器不替任务设置 RNG",
        "start_utc": utc_now(), "end_utc": None, "exit_code": None, "status": "running",
        "stdout": "stdout.log", "stderr": "stderr.log", "timeout_seconds": args.timeout,
        "episode_metrics": "不适用：审计" if args.kind == "audit" else "必须由任务程序写入；本封装器不推断",
    }
    meta_path = run_dir / "metadata.json"
    def save():
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save()
    print(json.dumps({"run_dir": str(run_dir), "command_argv": command}, ensure_ascii=False), flush=True)
    start = time.monotonic()
    child = None
    code = 1
    with (run_dir / "stdout.log").open("wb") as out, (run_dir / "stderr.log").open("wb") as err:
        try:
            child_environment = os.environ.copy()
            child_environment["G1_RUN_DIR"] = str(run_dir)
            child = subprocess.Popen(command, cwd=root, stdout=out, stderr=err, start_new_session=True, env=child_environment)
            metadata["child_pid"] = child.pid
            save()
            code = child.wait(timeout=args.timeout)
            metadata["status"] = "success" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            code = 124
            metadata["status"] = "timeout"
            metadata["child_returncode"] = child.returncode
        except KeyboardInterrupt:
            if child and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            code = 130
            metadata["status"] = "interrupted"
        except OSError as exc:
            code = 127
            metadata["status"] = "launch_failed"
            metadata["exception"] = repr(exc)
            err.write((repr(exc) + "\n").encode("utf-8"))
    metadata.update(end_utc=utc_now(), elapsed_seconds=time.monotonic() - start,
                    exit_code=code, effective_process_exit_code=code if code >= 0 else 128 - code)
    metadata["log_sha256"] = {name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
                              for name in ("stdout.log", "stderr.log")}
    save()
    print(json.dumps({"exit_code": code, "status": metadata["status"], "run_dir": str(run_dir)}, ensure_ascii=False), flush=True)
    return metadata["effective_process_exit_code"]


if __name__ == "__main__":
    sys.exit(main())
