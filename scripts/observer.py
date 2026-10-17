"""Read-only observation of the pinned official Python forward frame; no monkey patch."""
import inspect,sys,hashlib
import numpy as np
import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

class RouterObserver:
 def __init__(self,model,audit=False):
  self.code=Qwen3MoeSparseMoeBlock.forward.__code__
  self.source=inspect.getsource(Qwen3MoeSparseMoeBlock.forward)
  assert 'routing_weights, selected_experts = torch.topk(' in self.source
  assert 'one_hot(selected_experts' in self.source
  self.layers={id(layer.mlp):i for i,layer in enumerate(model.model.layers)}
  assert len(self.layers)==48
  self.audit=audit;self.records={};self.counts={};self.samples=[];self.handles=[]
  for layer in model.model.layers:
   lid=self.layers[id(layer.mlp)];self.counts[lid]=np.zeros(128,dtype=np.int64)
   for eid,expert in enumerate(layer.mlp.experts):
    self.handles.append(expert.register_forward_pre_hook(self.expert_hook(lid,eid)))
 def expert_hook(self,lid,eid):
  def hook(module,args):
   self.counts[lid][eid]+=args[0].shape[0]
   if self.audit and eid in (0,32,64,96) and args[0].shape[0]:
    f=inspect.currentframe().f_back
    while f is not None and f.f_code is not self.code:f=f.f_back
    assert f is not None,'Expert did not execute inside official MoE forward'
    loc=f.f_locals
    assert loc['expert_idx']==eid
    top_x,idx=loc['top_x'],loc['idx']
    assert torch.all(loc['selected_experts'][top_x,idx]==eid).item()
    expected=loc['hidden_states'][None,top_x].reshape(-1,loc['hidden_dim'])
    assert torch.equal(args[0],expected),'Actual expert input differs from selected token states'
    self.samples.append({'layer_id':lid,'expert_id':eid,'tokens_checked':len(top_x),'first_token_indices':top_x[:8].cpu().tolist(),'first_topk_slots':idx[:8].cpu().tolist()})
    del f
  return hook
 def trace(self,frame,event,arg):
  if frame.f_code is not self.code:return None
  frame.f_trace_lines=False
  if event=='return':
   x=frame.f_locals;lid=self.layers[id(x['self'])]
   assert lid not in self.records,'Repeated MoE execution in one prefill'
   ids=x['selected_experts'].detach().cpu().numpy().astype(np.uint8)
   assert np.array_equal(np.bincount(ids.ravel(),minlength=128),self.counts[lid]),'Expert execution counts do not match Top-k'
   if self.audit:
    recomputed=torch.topk(torch.softmax(x['router_logits'],dim=1,dtype=torch.float),8,dim=-1).indices
    assert torch.equal(recomputed,x['selected_experts'])
   self.records[lid]=(ids,x['routing_weights'].detach().float().cpu().numpy())
  return self.trace
 def __enter__(self):
  assert sys.gettrace() is None,'Do not use a debugger during capture'
  sys.settrace(self.trace);return self
 def __exit__(self,*exc):
  sys.settrace(None)
  for h in self.handles:h.remove()
