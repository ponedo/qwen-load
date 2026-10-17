import argparse,csv
import numpy as np
p=argparse.ArgumentParser();p.add_argument('trace');p.add_argument('output');p.add_argument('--layer',type=int);a=p.parse_args()
if a.layer is not None and not 0<=a.layer<48:p.error('--layer must be 0..47')
with np.load(a.trace,allow_pickle=False) as z,open(a.output,'w',newline='') as f:
 w=csv.writer(f);w.writerow(['request_id','token_position','layer_id']+[f'expert_{i}' for i in range(8)])
 for t,ri in enumerate(z['request_index']):
  for layer in ([a.layer] if a.layer is not None else z['layer_ids']):
   w.writerow([z['request_ids'][ri],z['token_position'][t],layer,*z['expert_ids'][t,layer].tolist()])
