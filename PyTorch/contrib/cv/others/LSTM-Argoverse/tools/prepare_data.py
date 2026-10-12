"""Prepare real official trajectory CSVs with the original coordinate functions."""
import argparse
import hashlib
import io
import json
import tarfile
from types import SimpleNamespace
import numpy as np
import pandas as pd
from common import ROOT, PROFILE, sha256, write_json
from upstream_components import coordinate_functions


def csv_entries(archive, limit):
    result = []
    with tarfile.open(archive, 'r|gz') as source:
        for member in source:
            if member.isfile() and member.name.endswith('.csv'):
                content = source.extractfile(member).read()
                result.append((member.name, content))
                if limit and len(result) == limit:
                    break
    if not result:
        raise RuntimeError('No official trajectory CSVs found.')
    return result


def convert(entries, functions):
    features, truth, ids, csv_hashes, intervals = [], [], [], [], []
    for member, blob in entries:
        frame = pd.read_csv(io.BytesIO(blob))
        agent = frame[frame.OBJECT_TYPE == 'AGENT'].sort_values('TIMESTAMP')
        if len(agent) != 50 or len(agent.TRACK_ID.unique()) != 1 or len(agent.CITY_NAME.unique()) != 1:
            raise RuntimeError('Expected one real AGENT trajectory with 50 positions: ' + member)
        times = agent.TIMESTAMP.to_numpy(dtype=np.float64)
        delta_t = np.diff(times)
        if not np.isfinite(times).all() or not np.all(delta_t > 0):
            raise RuntimeError('Official AGENT timestamps must be finite and strictly increasing: ' + member)
        intervals.append({'sequence':member,'mean_dt_seconds':float(delta_t.mean()),
                          'min_dt_seconds':float(delta_t.min()),'max_dt_seconds':float(delta_t.max())})
        xy = agent[['X','Y']].to_numpy(dtype=np.float64)
        if not np.isfinite(xy).all():
            raise RuntimeError('Invalid original city coordinates.')
        array = np.zeros((50, 11), dtype=object)
        array[:, 3:5] = xy
        features.append(array)
        truth.append(xy)
        ids.append(member)
        csv_hashes.append(hashlib.sha256(blob).hexdigest())
    args = SimpleNamespace(obs_len=20,pred_len=30,use_delta=True,normalize=True,use_map=False)
    frame = pd.DataFrame({'FEATURES':features})
    normalized = functions['get_normalized_traj'](frame,args).astype(np.float64)
    # Check that normalization of historical input does not use future coordinates.
    observed_frame = pd.DataFrame({'FEATURES':[feature[:20] for feature in features]})
    observed = functions['get_normalized_traj'](observed_frame,args).astype(np.float64)
    if not np.allclose(observed, normalized[:, :20], atol=1e-10, rtol=0):
        raise RuntimeError('History preprocessing depends on future positions.')
    reference = functions['get_relative_distance'](normalized,'train',args)
    helpers = {'REFERENCE':reference,'TRANSLATION':frame.TRANSLATION.to_list(),'ROTATION':frame.ROTATION.to_list()}
    restored_history, restored_future = functions['get_abs_traj'](normalized[:,:20].copy(),normalized[:,20:].copy(),args,helpers)
    truth = np.asarray(truth)
    maximum = float(np.max(np.abs(np.concatenate([restored_history,restored_future],axis=1)-truth)))
    if maximum > 1e-7:
        raise RuntimeError('Original normalization/inverse does not reconstruct real city coordinates.')
    return {'x':normalized[:,:20].astype(np.float32),'y':normalized[:,20:].astype(np.float32),
            'truth_city':truth[:,20:],'reference':reference,
            'translation':np.asarray(frame.TRANSLATION.to_list(),dtype=np.float64),
            'rotation':np.asarray(frame.ROTATION.to_list(),dtype=np.float64)}, ids, csv_hashes, maximum, intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=['official_subset'],default='official_subset')
    args = parser.parse_args()
    raw = ROOT / 'data/raw'
    output = ROOT / 'data/functional'
    if (output / 'metadata.json').is_file():
        from model_task import check_data
        check_data(output, 'functional')
        print('Existing official 128/32 trajectory inputs verified.')
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Preserve existing trajectory data; incomplete directory cannot be overwritten.')
    functions = coordinate_functions()
    archive=raw/'argoverse_official_functional_subset.tar.gz'
    lock=json.loads((ROOT/'assets.lock.json').read_text())['files']
    expected=next(item['sha256'] for item in lock if item['path'].endswith(archive.name))
    if sha256(archive)!=expected: raise RuntimeError('Official selected CSV archive SHA256 mismatch.')
    with tarfile.open(archive,'r:gz') as source:
        manifest=json.loads(source.extractfile('source_manifest.json').read())
        splits={'train':[],'val':[]}
        for item in manifest['files']:
            blob=source.extractfile(item['path']).read()
            if hashlib.sha256(blob).hexdigest()!=item['sha256']:
                raise RuntimeError('Selected CSV SHA256 differs from its actual original source record.')
            splits[item['split']].append((item['path'],blob))
    if len(splits['train'])!=128 or len(splits['val'])!=32:
        raise RuntimeError('Expected the frozen original train/val subset.')
    scope=manifest['scope']
    archives=manifest['source_archives']
    converted = {split:convert(entries,functions) for split,entries in splits.items()}
    if set(converted['train'][2]) & set(converted['val'][2]):
        raise RuntimeError('Original train and held-out trajectory CSVs overlap.')
    output.mkdir(parents=True,exist_ok=True)
    metadata = {'scope':scope,'synthetic_data':False,'obs_len':20,'pred_len':30,
                'source_archives':archives,'coordinate_system':'Argoverse city XY, meters',
                'train_sequences':converted['train'][1],'val_sequences':converted['val'][1],
                'train_csv_sha256':converted['train'][2],'val_csv_sha256':converted['val'][2],
                'coordinate_roundtrip_max_error_m':max(item[3] for item in converted.values()),
                'history_normalization_uses_only_observed_positions':True,'formal_accuracy_claimed':False}
    metadata['actual_timestamp_intervals']={split:item[4] for split,item in converted.items()}
    metadata['timestamp_resampling_applied']=False
    for split, item in converted.items():
        np.savez_compressed(output / (split+'.npz'),**item[0])
    metadata['files']=[{'path':split+'.npz','sha256':sha256(output/(split+'.npz'))} for split in splits]
    write_json(output/'metadata.json',metadata)
    print(json.dumps(metadata,indent=2),flush=True)


if __name__ == '__main__':
    main()
