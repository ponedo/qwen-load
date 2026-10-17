"""Validate full token/request/layer coverage, not just row counts."""
import argparse,json
from pathlib import Path
import numpy as np
from common import save_json,sha256

def validate(path):
 path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
 requests={r['request_id']:r for r in json.loads(Path('data/requests.json').read_text())}
 with np.load(path,allow_pickle=False) as z:
  x=z['expert_ids'];n=sum(meta['request_lengths'])
  assert x.shape==(n,48,8)
  assert x.dtype==np.uint8 and x.max()<128
  assert np.all(np.diff(np.sort(x.astype(np.int16),axis=-1),axis=-1)>0)
  assert np.array_equal(z['layer_ids'],np.arange(48))
  assert z['request_ids'].tolist()==meta['request_ids']
  assert len(meta['request_ids'])==meta['request_count']==len(set(meta['request_ids']))
  offset=0
  for i,(rid,length) in enumerate(zip(meta['request_ids'],meta['request_lengths'])):
   r=requests[rid];s=slice(offset,offset+length)
   assert np.all(z['request_index'][s]==i)
   assert np.array_equal(z['token_position'][s],np.arange(r['token_start'],r['token_end']))
   assert np.array_equal(z['token_id'][s],r['input_ids'])
   assert length==r['actual_length']
   offset+=length
  assert len(z['request_index'])==len(z['token_position'])==len(z['token_id'])==n
  w=z['router_weights'];assert w.shape==x.shape and np.isfinite(w).all() and np.all(w>=0)
  assert np.allclose(w.sum(-1),1,atol=0.01) # weights stored after official BF16 cast
 assert meta['dispatch_count_check_all_layers']
 return {'status':'passed','batch_id':meta['batch_id'],'valid_tokens':n,'moe_layers':48,'records':n*48,'expert_slots':n*48*8,'range_unique_shape_coverage_token_ids':True,'dispatch_counts_checked_during_forward':True,'sha256':sha256(path)}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('files',nargs='+');a=p.parse_args()
 for path in a.files:print(json.dumps(validate(path)))
