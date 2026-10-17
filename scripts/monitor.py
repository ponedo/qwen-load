"""Run one collector, sample device-wide memory and container RAM every second."""
import argparse,subprocess,time,csv
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
assert a.command
Path('reports').mkdir(exist_ok=True)
child=subprocess.Popen(a.command)
try:
 with open('reports/resource_samples.csv','w',newline='') as f:
  w=csv.writer(f);w.writerow(['unix_time','gpu_index','device_memory_used_mib','gpu_util_percent','container_memory_bytes'])
  while child.poll() is None:
   r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
   ram=''
   for path in ['/sys/fs/cgroup/memory.current','/sys/fs/cgroup/memory/memory.usage_in_bytes']:
    if Path(path).exists():ram=Path(path).read_text().strip();break
   for line in r.stdout.splitlines():w.writerow([time.time(),*[s.strip() for s in line.split(',')],ram])
   f.flush();time.sleep(1)
except BaseException:
 child.terminate();child.wait();raise
raise SystemExit(child.returncode)
