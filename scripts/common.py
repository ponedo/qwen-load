import hashlib,json
from pathlib import Path
MODEL_ID='Qwen/Qwen3-30B-A3B'
REVISION='ad44e777bcd18fa416d9da3bd8f70d33ebb85d39'
DATASET_ID='THUDM/LongBench'
DATASET_REVISION='5e628be450b7e67fb7ae6e201bd6d8f7056f7672'
CACHE=Path('/data/qwen-router-cache')
def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def save_json(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
