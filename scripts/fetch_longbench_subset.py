"""Fetch only narrativeqa.jsonl from the immutable ZIP using HTTP byte ranges."""
import io,zipfile,requests
from pathlib import Path
from common import *
class RemoteZip(io.RawIOBase):
 def __init__(self,url):
  self.url=url;self.pos=0;self.session=requests.Session()
  r=self.session.get(url,headers={'Range':'bytes=0-0'},timeout=(30,90));r.raise_for_status()
  assert r.status_code==206
  self.size=int(r.headers['Content-Range'].split('/')[-1])
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  return self.pos
 def read(self,n=-1):
  end=self.size-1 if n<0 else min(self.size-1,self.pos+n-1)
  if end<self.pos:return b''
  r=self.session.get(self.url,headers={'Range':f'bytes={self.pos}-{end}'},timeout=(30,90));r.raise_for_status()
  assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {self.pos}-')
  data=r.content;assert len(data)==end-self.pos+1;self.pos+=len(data);return data
if __name__=='__main__':
 url=f'https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/data.zip'
 with zipfile.ZipFile(RemoteZip(url)) as z:
  name=next(n for n in z.namelist() if n.endswith('narrativeqa.jsonl'))
  info=z.getinfo(name);print(name,info.compress_size,info.file_size,flush=True)
  # ZIP CRC is checked by ZipFile.read; retain SHA256 for the extracted member.
  content=z.read(name)
  path=CACHE/'narrativeqa.jsonl';path.write_bytes(content)
  save_json('reports/dataset_member.json',{'dataset':DATASET_ID,'revision':DATASET_REVISION,'url':url,'member':name,'zip_crc32':info.CRC,'sha256':sha256(path),'bytes':len(content)})
 print('Saved',path,flush=True)
