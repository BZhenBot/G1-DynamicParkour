"""V2实际PPO累计预算：同时核对真实入口，避免包装器kind误标漏计。"""
import json,math
from pathlib import Path

def effective_kind(meta,directory):
    argv=meta.get('command_argv',[])
    if len(argv)>2 and argv[1]=='-m' and argv[2] in ('py_compile','compileall'):return 'audit'
    scripts=[Path(s).name for s in meta.get('command_argv',[]) if str(s).endswith('.py')]
    if '--gate' in meta.get('command_argv',[]) and any(s in ('train_course_turn.py','train_turn_locomotion.py','train_end_recovery.py','train_prepare_settle_recovery.py') for s in scripts):return 'audit'
    if any(s.startswith('train_') for s in scripts):return 'train'
    if (directory/'training_manifest.json').exists():return 'train'
    return meta['kind']

def enforce_budget(root,out,reserved):
    consumed=0.;records=[]
    prefixes=('v21-','v22-') if out.name.split('Z-',1)[-1].startswith(('v21-','v22-')) else ('v2-',)
    for path in (root/'experiments').glob('*/metadata.json'):
        if path.parent.name<'20261014' or not path.parent.name.split('Z-',1)[-1].startswith(prefixes):continue
        meta=json.loads(path.read_text())
        if effective_kind(meta,path.parent)!='train' or path.parent==out:continue
        if meta['status']=='running':raise RuntimeError('V2训练尚未核实终态，禁止第二GPU任务: '+str(path))
        manifest=path.parent/'training_manifest.json';seconds=meta.get('elapsed_seconds')
        if manifest.exists():
            data=json.loads(manifest.read_text());seconds=data.get('elapsed_training_seconds',data.get('elapsed_seconds_monotonic',seconds))
        if seconds is None or not math.isfinite(seconds) or seconds<0:raise RuntimeError('V2训练实际耗时缺失/非法，不能填零: '+str(path))
        consumed+=seconds;records.append({'run':str(path.parent),'seconds':seconds,'exit_code':meta['exit_code'],
            'recorded_kind':meta['kind'],'effective_kind':'train','classification_basis':'真实train入口/learning manifest；保留原metadata'})
    report={'budget_seconds':21600,'consumed_seconds':consumed,'reserved_seconds':reserved,'records':records}
    (out/'training_budget_before.json').write_text(json.dumps(report,indent=2))
    if consumed+reserved>21600:raise RuntimeError('超过首轮6小时预算')
    return consumed
