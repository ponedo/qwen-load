"""Download immutable sources, verify HF LFS hashes, build explicit LongBench excerpts."""
import argparse,json,zipfile,time
from pathlib import Path
import requests
from common import *

def download(url,path,expected_sha=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and (not expected_sha or sha256(path)==expected_sha):return
 partial=path.with_suffix(path.suffix+'.part')
 start=partial.stat().st_size if partial.exists() else 0
 with requests.get(url,headers={'Range':f'bytes={start}-'} if start else {},stream=True,timeout=(30,90)) as r:
  r.raise_for_status()
  append=start>0 and r.status_code==206
  if append and not r.headers.get('Content-Range','').startswith(f'bytes {start}-'):raise RuntimeError('Bad range')
  with partial.open('ab' if append else 'wb') as f:
   for chunk in r.iter_content(1024*1024):f.write(chunk)
 if expected_sha and sha256(partial)!=expected_sha:raise RuntimeError(f'Hash mismatch: {path}')
 partial.rename(path)

def fetch():
 api=requests.get(f'https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true',timeout=30)
 api.raise_for_status();meta=api.json();assert meta['sha']==REVISION
 save_json('reports/model_files.json',meta)
 hashes={}
 for f in meta['siblings']:
  name=f['rfilename']
  if name.startswith('.'):continue
  print('Downloading/verifying',name,flush=True)
  path=CACHE/'model'/name
  download(f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}',path,f.get('lfs',{}).get('sha256'))
  hashes[name]=sha256(path)
 save_json('reports/model_hashes.json',{'revision':REVISION,'files':hashes})
 download(f'https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/data.zip',CACHE/'longbench.zip')

def inputs():
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(CACHE/'model',local_files_only=True)
 # Public narrative QA text. Store an explicitly labeled prefix of the original
 # benchmark prompt, never represent an excerpt as a complete original request.
 if (CACHE/'narrativeqa.jsonl').exists():
  rows=[json.loads(line) for line in (CACHE/'narrativeqa.jsonl').read_text().splitlines()]
 else:
  with zipfile.ZipFile(CACHE/'longbench.zip') as z:
   name=next(n for n in z.namelist() if n.endswith('/narrativeqa.jsonl') or n=='narrativeqa.jsonl')
   rows=[json.loads(line) for line in z.read(name).decode().splitlines()]
 prompts=[]
 for i,row in enumerate(rows):
  text='Read the following story and answer the question.\n\n'+row['context']+'\n\nQuestion: '+row['input']+'\nAnswer:'
  ids=tok(text,add_special_tokens=False)['input_ids']
  if len(ids)>=8192:prompts.append((i,row,text,ids))
  if len(prompts)==3:break
 assert len(prompts)>=3
 batches=[];requests_out=[]
 specs=[('minimal',192,[192])]+[(f'single_{n}',n,[n]) for n in [256,512,1024,2048,4096,8192]]+[(f'batch_{n}',n,[n//4,3*n//4]) for n in [256,1024,4096]]
 for bid,target,lengths in specs:
  reqs=[]
  for j,length in enumerate(lengths):
   index,row,text,ids=prompts[j];rid=f'{bid}_r{j}'
   req={'request_id':rid,'dataset':DATASET_ID,'dataset_revision':DATASET_REVISION,'subset':'narrativeqa','row_index':index,'sample_id':row.get('_id',str(index)), 'original_prompt_tokens':len(ids),'token_start':0,'token_end':length,'actual_length':length,'is_truncated':True,'is_complete_original_request':False,'prompt_method':'raw instruction + context + question + Answer; tokenize without special tokens, then take explicit prefix; no chat template','input_ids':ids[:length],'text':tok.decode(ids[:length]),'original_prompt_sha256':__import__('hashlib').sha256(text.encode()).hexdigest()}
   requests_out.append(req);reqs.append(rid)
  batches.append({'batch_id':bid,'target_valid_tokens':target,'actual_valid_tokens':sum(lengths),'request_ids':reqs,'request_lengths':lengths,'request_count':len(reqs)})
 save_json('data/requests.json',requests_out)
 save_json('data/batches.json',batches)
 save_json('data/provenance.json',{'model':MODEL_ID,'model_revision':REVISION,'tokenizer_revision':REVISION,'tokenizer_class':type(tok).__name__,'add_special_tokens':False,'chat_template_applied':False,'padding_side':'right','dataset':DATASET_ID,'dataset_revision':DATASET_REVISION,'source_sha256':sha256(CACHE/'narrativeqa.jsonl') if (CACHE/'narrativeqa.jsonl').exists() else sha256(CACHE/'longbench.zip'),'note':'All requests are explicitly truncated real-text prefixes. Question may fall outside prefix. They are not complete LongBench evaluation requests.'})
 print('Prepared',len(batches),'batches',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs-only',action='store_true');a=p.parse_args()
 if not a.inputs_only:fetch()
 inputs()
