"""Real BF16 Qwen3 prefill capture. One model instance, stop immediately on errors."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,time,traceback,platform,importlib.metadata,hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from common import *
from observer import RouterObserver
from validate import validate

def sync():
 for d in range(torch.cuda.device_count()):torch.cuda.synchronize(d)

def run(model,batch,requests,output,audit):
 selected=[requests[r] for r in batch['request_ids']];lengths=[len(r['input_ids']) for r in selected]
 width=max(lengths);ids=torch.full((len(selected),width),151643,dtype=torch.long);mask=torch.zeros_like(ids)
 for i,r in enumerate(selected):ids[i,:lengths[i]]=torch.tensor(r['input_ids']);mask[i,:lengths[i]]=1
 for d in range(2):torch.cuda.reset_peak_memory_stats(d)
 sync();start=time.perf_counter()
 with torch.inference_mode(),RouterObserver(model,audit=audit) as observer:
  result=model(input_ids=ids.to(0),attention_mask=mask.to(0),use_cache=False,output_router_logits=False,return_dict=True,logits_to_keep=1)
 sync();duration=time.perf_counter()-start
 assert set(observer.records)==set(range(48))
 valid=mask.numpy().astype(bool).ravel()
 experts=np.stack([observer.records[l][0][valid] for l in range(48)],axis=1)
 weights=np.stack([observer.records[l][1][valid] for l in range(48)],axis=1)
 request_index=np.concatenate([np.full(n,i,dtype=np.int32) for i,n in enumerate(lengths)])
 positions=np.concatenate([np.arange(n,dtype=np.int32)+selected[i]['token_start'] for i,n in enumerate(lengths)])
 flat_tokens=ids.numpy().ravel()[valid]
 meta={**batch,'model':MODEL_ID,'revision':REVISION,'tokenizer_revision':REVISION,'dtype':'bfloat16','attention':'sdpa','prefill_only':True,'use_cache':False,'layer_ids':list(range(48)),'elapsed_seconds':duration,'padding_tokens':int((mask==0).sum()),'gpu_peak_allocated_bytes':[torch.cuda.max_memory_allocated(d) for d in range(2)],'gpu_peak_reserved_bytes':[torch.cuda.max_memory_reserved(d) for d in range(2)],'observer_source_sha256':hashlib.sha256(observer.source.encode()).hexdigest(),'dispatch_count_check_all_layers':True,'expert_input_audit':observer.samples,'request_lengths':lengths,'actual_valid_tokens':sum(lengths)}
 path=Path(output)/(batch['batch_id']+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(path,expert_ids=experts,router_weights=weights,request_index=request_index,token_position=positions,token_id=flat_tokens,request_ids=np.array(batch['request_ids']),layer_ids=np.arange(48,dtype=np.uint8))
 save_json(path.with_suffix('.json'),meta)
 checked=validate(path);save_json(Path(output)/(batch['batch_id']+'.validation.json'),checked)
 del result
 print(batch['batch_id'],experts.shape,'seconds',round(duration,2),flush=True)
 return experts,meta

def main():
 p=argparse.ArgumentParser();p.add_argument('--model-dir',default=str(CACHE/'model'));p.add_argument('--offload-dir',default=str(CACHE/'offload'));p.add_argument('--output',default='traces');p.add_argument('--minimal-only',action='store_true');a=p.parse_args()
 assert importlib.metadata.version('transformers')=='4.51.3'
 assert torch.cuda.device_count()==2
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.use_deterministic_algorithms(True)
 config=json.loads((Path(a.model_dir)/'config.json').read_text())
 assert (config['num_hidden_layers'],config['num_experts'],config['num_experts_per_tok'])==(48,128,8)
 hashes=json.loads(Path('reports/model_hashes.json').read_text());assert hashes['revision']==REVISION
 # Reverify downloaded bytes before loading official weights (no random fallback).
 for name,digest in hashes['files'].items():
  assert sha256(Path(a.model_dir)/name)==digest,f'Changed model file: {name}'
 batches=json.loads(Path('data/batches.json').read_text());requests={r['request_id']:r for r in json.loads(Path('data/requests.json').read_text())}
 free=[torch.cuda.mem_get_info(d)[0] for d in range(2)]
 # Reserve >=6 GiB GPU0 for offloaded layer staging and activations, >=3 GiB GPU1.
 budgets=[max(0,free[0]-6*2**30),max(0,free[1]-3*2**30)]
 assert min(budgets)>2**30,'Insufficient free GPU memory; stop, do not retry'
 env={'python':platform.python_version(),'packages':{x:importlib.metadata.version(x) for x in ['torch','transformers','accelerate','huggingface-hub','tokenizers','numpy']},'cuda':torch.version.cuda,'gpu_free_before_load':free,'max_memory_bytes':budgets,'deterministic_algorithms':True,'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],'tf32':False}
 save_json('reports/run_environment.json',env)
 start=time.perf_counter()
 model,loading_info=AutoModelForCausalLM.from_pretrained(a.model_dir,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map='auto',max_memory={0:budgets[0],1:budgets[1],'cpu':0},offload_folder=a.offload_dir,offload_state_dict=True,low_cpu_mem_usage=True,output_loading_info=True)
 assert not any(loading_info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),loading_info
 model.eval()
 save_json('reports/model_load.json',{'elapsed_seconds':time.perf_counter()-start,'device_map':model.hf_device_map,'parameter_dtype':'bfloat16'})
 minimum=batches[0]
 first,meta=run(model,minimum,requests,a.output,True)
 repeat={**minimum,'batch_id':'minimal_repeat'}
 second,_=run(model,repeat,requests,a.output,True)
 mismatches=int(np.count_nonzero(first!=second))
 save_json('reports/repeatability.json',{'exact_match':mismatches==0,'mismatched_slots':mismatches,'total_slots':int(first.size),'same_input_ids':True,'same_batch_shape':True,'note':'Checks repeatability on this software/device layout only; BF16 boundaries can differ across kernels/hardware.'})
 assert mismatches==0,'Repeated routes differ; inspect report before expansion'
 assert meta['expert_input_audit'],'Missing actual dispatch audit'
 save_json('reports/stage1.json',{'status':'passed','batch_id':'minimal','repeatability':'exact','all_48_layers':True,'expert_dispatch_inputs_checked':True})
 if not a.minimal_only:
  for batch in batches[1:]:run(model,batch,requests,a.output,False)
 save_json('reports/collection_status.json',{'status':'completed','minimal_only':a.minimal_only})
if __name__=='__main__':
 try:main()
 except Exception as e:
  save_json('reports/collection_status.json',{'status':'stopped','exception':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'policy':'No automatic OOM retry or scale expansion after failure'})
  raise
