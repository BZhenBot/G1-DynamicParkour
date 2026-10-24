"""从任意WSL本地副本复用冻结策略，强制使用本副本源码并自动独立核验。"""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--python',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--episodes',type=int,default=1);p.add_argument('--seed',type=int,default=75000);p.add_argument('--video-episodes',type=int,default=1);p.add_argument('--label',default='portable-course-replay');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
python=a.python.expanduser().absolute();assert python.is_file() and a.episodes>0 and 0<=a.video_episodes<=a.episodes
config=a.config if a.config.is_absolute() else root/a.config;config=config.absolute();config_relative=config.relative_to(root).as_posix();settings=json.loads(config.read_text())
for role in ('jump','recovery','alignment','position','second_jump','end_recovery'):
 weight=root/settings[role+'_checkpoint'];assert hashlib.sha256(weight.read_bytes()).hexdigest()==settings[role+'_sha256'],role
if settings.get('settle_recovery_checkpoint'):
 weight=root/settings['settle_recovery_checkpoint'];assert hashlib.sha256(weight.read_bytes()).hexdigest()==settings['settle_recovery_sha256']
if settings.get('formal_protocol'):
 from formal_course_protocol import validate_formal
 previous=Path.cwd()
 try:os.chdir(root);validate_formal(settings,a.episodes,a.seed,0)
 finally:os.chdir(previous)
env=os.environ.copy();env['PYTHONPATH']=str(root/'src')+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
# 先实查包来源，避免editable install意外读取原主工作区源码。
probe=subprocess.run([str(python),'-c','import g1_dynamicparkour; print(g1_dynamicparkour.__file__)'],cwd=root,env=env,capture_output=True,text=True,check=True)
assert Path(probe.stdout.strip()).resolve().is_relative_to((root/'src').resolve())
command=[str(python),'scripts/run_experiment.py','--label',a.label,'--kind','eval','--seed',str(a.seed),'--config',config_relative,'--timeout',str(max(600,a.episodes*90)),'--',str(python),'scripts/evaluate_course.py','--config',config_relative,'--episodes',str(a.episodes),'--seed',str(a.seed),'--video-episodes',str(a.video_episodes)]
print(json.dumps({'project':str(root),'python':str(python),'package_source':probe.stdout.strip(),'command':command,'dry_run':a.dry_run,'training':False},ensure_ascii=False),flush=True)
if a.dry_run:sys.exit(0)
run=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True);print(run.stdout,end='');print(run.stderr,end='',file=sys.stderr)
if run.returncode:sys.exit(run.returncode)
run_dir=Path(json.loads(run.stdout.splitlines()[0])['run_dir'])
audit=[str(python),'scripts/run_experiment.py','--label',a.label+'-independent-physics','--kind','audit','--timeout','900','--',str(python),'scripts/audit_course.py','--run',run_dir.relative_to(root).as_posix()]
raise SystemExit(subprocess.run(audit,cwd=root,env=env).returncode)