#!/usr/bin/env python3
"""Download fixed official WorldEngine functional-training assets with SHA256."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
import requests
from prepare import LOCK, require_checkout
from world_mini_data import ASSETS
CHUNK = 32 * 1024 * 1024


def download(url, destination, total, expected_hash, prefix):
    destination = Path(destination)
    prefix = Path(prefix)
    if destination.is_file():
        prefix_size = 0
    else:
        prefix_size = prefix.stat().st_size if prefix.exists() else 0
        if not 0 <= prefix_size < total:
            raise RuntimeError('Unexpected prefix size')
        parts = destination.parent / ('.' + destination.name + '.ranges')
        parts.mkdir(exist_ok=True)
        plan_file = parts / 'plan.json'
        plan = {'prefix_size': prefix_size, 'total': total, 'sha256': expected_hash}
        if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
            raise RuntimeError('Download range plan changed')
        plan_file.write_text(json.dumps(plan))
        stride = math.ceil((total - prefix_size) / 8)
        ranges = [(i, prefix_size + i * stride, min(total - 1, prefix_size + (i + 1) * stride - 1))
                  for i in range(8) if prefix_size + i * stride < total]
        print(json.dumps({'parallel_fetch': destination.name, 'prefix_bytes': prefix_size, 'total': total}), flush=True)

        def fetch(item):
            index, start, end = item
            expected = end - start + 1
            progress = parts / f'part-{index:02d}'
            session = requests.Session()
            done = progress.stat().st_size if progress.exists() else 0
            if done > expected or (done < expected and done % CHUNK):
                raise RuntimeError('Invalid saved segment size')
            while done < expected:
                left = start + done
                right = min(end, left + CHUNK - 1)
                target_range = f'bytes {left}-{right}/{total}'
                temp = parts / f'.piece-{index:02d}'
                for attempt in range(6):
                    try:
                        with session.get(url, headers={'Range': f'bytes={left}-{right}',
                                           'Cache-Control': 'no-cache', 'Pragma': 'no-cache'},
                                          stream=True, timeout=(20, 120)) as response:
                            response.raise_for_status()
                            if response.status_code != 206 or response.headers.get('Content-Range') != target_range:
                                raise RuntimeError('Unexpected range: status=' + str(response.status_code)
                                                   + ' got=' + str(response.headers.get('Content-Range'))
                                                   + ' wanted=' + target_range
                                                   + ' type=' + response.headers.get('Content-Type', ''))
                            received = 0
                            with temp.open('wb') as output:
                                for chunk in response.iter_content(4 * 1024 * 1024):
                                    output.write(chunk)
                                    received += len(chunk)
                                    if received > right - left + 1:
                                        raise RuntimeError('Range larger than requested')
                        if received != right - left + 1:
                            raise RuntimeError('Range length mismatch')
                        with progress.open('ab') as output, temp.open('rb') as piece:
                            for chunk in iter(lambda: piece.read(4 * 1024 * 1024), b''):
                                output.write(chunk)
                        temp.unlink()
                        done += received
                        print(json.dumps({'file': destination.name, 'segment': index,
                                          'bytes': done, 'segment_total': expected}), flush=True)
                        break
                    except (requests.RequestException, RuntimeError) as exc:
                        print(json.dumps({'file': destination.name, 'segment': index, 'retry': attempt + 1,
                                          'error': type(exc).__name__, 'message': str(exc)[:160]}), flush=True)
                        if attempt == 5:
                            raise
                        time.sleep(min(3 * (attempt + 1), 15))
            return progress

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(fetch, item) for item in ranges]
            for future in futures:
                future.result()
        if prefix_size and prefix.stat().st_size != prefix_size:
            raise RuntimeError('Original writer still changed the prefix')
        assembled = destination.with_suffix(destination.suffix + '.assembled')
        digest = hashlib.sha256()
        with assembled.open('wb') as output:
            ordered = ([prefix] if prefix_size else []) + [parts / f'part-{i:02d}' for i, _, _ in ranges]
            for file in ordered:
                with file.open('rb') as source:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                        output.write(chunk)
                        digest.update(chunk)
        if assembled.stat().st_size != total or (expected_hash and digest.hexdigest() != expected_hash):
            raise RuntimeError('Assembled publisher SHA256/size mismatch')
        if destination.name.endswith(('.tar.gz', '.tgz')):
            subprocess.run(['tar', '-tzf', str(assembled)], stdout=subprocess.DEVNULL, check=True, timeout=1800)
        os.replace(assembled, destination)
    digest = hashlib.sha256()
    with destination.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if destination.stat().st_size != total or (expected_hash and digest.hexdigest() != expected_hash):
        raise RuntimeError('Final file checksum/size mismatch')
    result = {'file': destination.name, 'bytes': total, 'sha256': digest.hexdigest(),
              'checksum_scope': 'publisher checksum' if expected_hash else 'observed checksum; archive integrity checked'}
    print(json.dumps({'verified': result}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    require_checkout('worldengine', source)
    directory = args.destination.resolve()
    print(json.dumps({'scope': 'NAVTEST functional training only', 'download_bytes': sum(v[3] for v in ASSETS),
                      'destination': str(directory), 'execute': args.execute}, indent=2))
    if not args.execute:
        return
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for repo, revision, path, size, checksum in ASSETS:
        destination = directory / Path(path).name
        url = 'https://huggingface.co/datasets/' + repo + '/resolve/' + revision + '/' + path
        result = download(url, destination, size, checksum, destination.with_suffix(destination.suffix + '.part'))
        result.update(repo=repo, revision=revision, path=path)
        records.append(result)
        (directory / 'download_manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    import sys
    subprocess.run([sys.executable, str(Path(__file__).with_name('fetch_world_vocab.py')),
                    '--source', str(source), '--execute'], check=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name('prepare_world_mini.py')),
                    '--source', str(source), '--data-dir', str(directory)], check=True)


if __name__ == '__main__':
    main()
