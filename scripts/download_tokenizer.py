"""Small-file preparation independent of weight availability."""
from concurrent.futures import ThreadPoolExecutor
from common import *
from prepare import download
files=['config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']
def fetch(name):
 download(f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}',CACHE/'model'/name)
 print(name,flush=True)
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(fetch,files))
 save_json('reports/tokenizer_hashes.json',{'revision':REVISION,'files':{name:sha256(CACHE/'model'/name) for name in files}})
