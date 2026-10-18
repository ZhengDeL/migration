"""Run the complete checkpoint comparison in this model's exact runtime image."""
import json
import subprocess
import sys
from common import ROOT, PROFILE, own_path, write_json

left,right,output=[own_path(x, exists=i<2) for i,x in enumerate(sys.argv[1:])]
image=json.loads((ROOT/'images.lock.json').read_text())['images']['runtime']
probe=subprocess.run(['docker','image','inspect','--format','{{.Id}}',image['tag']],capture_output=True,text=True,check=True)
assert probe.stdout.strip()==image['image_id']
command=['docker','run','--rm','--network','none','--memory','16g','--cpus','4',
         '-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0','-e','OMP_NUM_THREADS=2',
         '-v',str(ROOT)+':'+str(ROOT)+':ro','-w',str(ROOT),
         '--entrypoint',PROFILE['python'],image['tag'],'tools/resume_control.py',str(left),str(right)]
result=subprocess.run(command,capture_output=True,text=True)
if result.returncode and not result.stdout.lstrip().startswith('{'):
    print(result.stdout)
    print(result.stderr,file=sys.stderr)
    raise SystemExit(result.returncode)
value=json.loads(result.stdout)
write_json(output,value)
print(json.dumps(value,indent=2))
raise SystemExit(result.returncode)
