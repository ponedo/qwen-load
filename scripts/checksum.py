"""Hash deliverables, excluding caches, Git metadata and this manifest itself."""
from pathlib import Path
from common import sha256
root=Path('.')
paths=[]
for name in ['README.md','requirements.txt','.gitignore','scripts','reports','data','traces','stats']:
 p=root/name
 paths.extend([p] if p.is_file() else [f for f in p.rglob('*') if f.is_file()] if p.exists() else [])
paths=[p for p in paths if '__pycache__' not in p.parts and p.suffix not in ['.log','.pyc']]
Path('SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.as_posix()}\n' for p in sorted(paths)))
