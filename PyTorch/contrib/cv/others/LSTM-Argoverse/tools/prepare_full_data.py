"""Stream every official train/val CSV through the pinned original XY coordinate functions."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import tarfile
import time
from pathlib import Path
import numpy as np
from common import ROOT, own_path, sha256, write_json

FUNCTIONS = None

def initialize():
    global FUNCTIONS
    from upstream_components import coordinate_functions
    FUNCTIONS = coordinate_functions()

def convert_chunk(entries):
    from prepare_data import convert
    return convert(entries, FUNCTIONS)

def chunks(archive, chunk_size=256):
    batch=[]
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            if member.isfile() and member.name.endswith('.csv'):
                batch.append((member.name, stream.extractfile(member).read()))
                if len(batch)==chunk_size:
                    yield batch
                    batch=[]
        if batch:yield batch

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--archives',type=Path,default=ROOT/'data/raw')
    parser.add_argument('--output_dir',type=Path,default=ROOT/'data/full')
    args=parser.parse_args()
    if args.workers<1 or args.workers>32:raise RuntimeError('workers must be 1..32.')
    lock=json.loads((ROOT/'full_data_sources.lock.json').read_text())
    target=own_path(args.output_dir)
    archives=own_path(args.archives,exists=True)
    if (target/'metadata.json').exists():
        from model_task import check_data
        print(check_data(target,'full')['scope']);return
    if target.exists() and any(target.iterdir()):
        raise RuntimeError('Incomplete full data is preserved; use a fresh model workspace.')
    target.mkdir(parents=True,exist_ok=True)
    metadata={'scope':'official_argoverse_v11_full_train_val','synthetic_data':False,
        'obs_len':20,'pred_len':30,'coordinate_system':'Argoverse city XY, meters',
        'source_archives':lock['archives'],'counts':{},'coordinate_roundtrip_max_error_m':0.0,
        'history_normalization_uses_only_observed_positions':True,'timestamp_resampling_applied':False,
        'train_val_raw_csv_hashes_disjoint':False,'formal_accuracy_claimed':False}
    hashes={}
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
        for split,spec in lock['archives'].items():
            archive=archives/spec['filename']
            if archive.stat().st_size!=spec['bytes'] or sha256(archive)!=spec['sha256']:
                raise RuntimeError('Full official archive size/SHA mismatch: '+split)
            results=[];pending=deque();hashes[split]=set();count=0;begin=time.time()
            manifest=target/(split+'_source_sequences.jsonl.gz')
            with gzip.open(manifest,'wt',encoding='utf-8') as records:
                def consume(future):
                    nonlocal count
                    arrays,ids,digests,error,intervals=future.result()
                    results.append(arrays)
                    metadata['coordinate_roundtrip_max_error_m']=max(metadata['coordinate_roundtrip_max_error_m'],error)
                    for name,digest,times in zip(ids,digests,intervals):
                        if digest in hashes[split]:raise RuntimeError('Duplicate official trajectory CSV content.')
                        hashes[split].add(digest)
                        records.write(json.dumps({'sequence':name,'sha256':digest,**times},separators=(',',':'))+'\n')
                    count+=len(ids)
                    if count%2048==0:print(json.dumps({'split':split,'sequences':count,'seconds':round(time.time()-begin,1)}),flush=True)
                for batch in chunks(archive):
                    pending.append(pool.submit(convert_chunk,batch))
                    if len(pending)>=args.workers*2:consume(pending.popleft())
                while pending:consume(pending.popleft())
            if count!=spec['sequences']:raise RuntimeError('Official full sequence count differs: '+str((split,count,spec['sequences'])))
            combined={key:np.concatenate([item[key] for item in results]) for key in results[0]}
            np.savez_compressed(target/(split+'.npz'),**combined)
            metadata['counts'][split]=count
            del results,combined
            print(json.dumps({'split_complete':split,'sequences':count,'seconds':time.time()-begin}),flush=True)
    if hashes['train'] & hashes['val']:raise RuntimeError('Official train/val contain identical CSV content.')
    metadata['train_val_raw_csv_hashes_disjoint']=True
    metadata['files']=[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(target.iterdir()) if p.is_file()]
    write_json(target/'metadata.json',metadata)
    print(json.dumps({'result':'full_real_data_prepared','counts':metadata['counts'],
        'roundtrip_max_error_m':metadata['coordinate_roundtrip_max_error_m']}),flush=True)

if __name__=='__main__':main()
