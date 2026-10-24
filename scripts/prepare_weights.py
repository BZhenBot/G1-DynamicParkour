"""验证公开权重资产并只恢复已登记成员；不下载、不反序列化checkpoint。"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import os
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assets = json.loads((root / 'release/assets-manifest.json').read_text())
    allowed = [a for a in assets['allowed_assets'] if a['name'] == args.archive.name]
    if len(allowed) != 1:
        raise ValueError('未登记的权重归档名称')
    archive_bytes = args.archive.read_bytes()
    if hashlib.sha256(archive_bytes).hexdigest() != allowed[0]['sha256']:
        raise ValueError('归档SHA256不匹配')
    weights = json.loads((root / 'release/weights-manifest.json').read_text())['weights']
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('归档中有重复成员')
        # Inspect every member for traversal, including notices not extracted here.
        for name in names:
            parts = PurePosixPath(name)
            if parts.is_absolute() or '..' in parts.parts or '\\' in name or ':' in name:
                raise ValueError('归档含非法路径')
        verified = []
        for item in weights:
            path = PurePosixPath(item['path'])
            if path.is_absolute() or '..' in path.parts or not str(path).startswith('experiments/'):
                raise ValueError('清单路径非法')
            info = archive.getinfo(item['path'])
            if info.file_size != item['bytes']:
                raise ValueError('权重大小不匹配')
            data = archive.read(info)
            if hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('权重SHA256不匹配')
            target = root.joinpath(*path.parts)
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError('目标路径越界')
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('拒绝覆盖不同实验权重')
            verified.append((target, data))
        if not args.verify_only:
            for target, data in verified:
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temp:
                    temp.write(data)
                    temporary = Path(temp.name)
                os.replace(temporary, target)
    print(json.dumps({'verified_weights': len(weights), 'restored': not args.verify_only,
                      'training': False, 'task_success': '不适用：完整性检查'}))


if __name__ == '__main__':
    main()
