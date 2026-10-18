"""Exercise real shared source preparation from concurrent Linux processes."""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))


def worker(rank, rounds, barrier, results):
    from prepare_source import prepare_source
    try:
        expected = json.loads((ROOT / 'validation/patch_manifest.json').read_text())
        for _ in range(rounds):
            barrier.wait(timeout=60)
            source = prepare_source()
            assert source == ROOT / 'runtime/ResWorld'
            assert json.loads((ROOT / 'runtime/patch_manifest.json').read_text()) == expected
            for record in expected['changes']:
                actual = hashlib.sha256((source / record['path']).read_bytes()).hexdigest()
                assert actual == record['adapted_sha256']
        results.put({'rank': rank, 'rounds': rounds, 'result': 'pass'})
    except BaseException as error:
        results.put({'rank': rank, 'result': 'failed', 'error': repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.workers > 1 and args.rounds > 0
    context = mp.get_context('spawn')
    barrier = context.Barrier(args.workers)
    results = context.Queue()
    children = [context.Process(target=worker, args=(rank, args.rounds, barrier, results))
                for rank in range(args.workers)]
    for process in children:
        process.start()
    records = [results.get(timeout=180) for _ in children]
    for process in children:
        process.join(timeout=15)
        assert process.exitcode == 0, (process.pid, process.exitcode)
    assert all(record['result'] == 'pass' for record in records)
    value = {'result': 'pass_shared_runtime_source_preparation', 'workers': args.workers,
             'rounds_per_worker': args.rounds, 'total_preparations': args.workers * args.rounds,
             'real_upstream_files_and_adapted_hashes_checked': True,
             'records': sorted(records, key=lambda item: item['rank'])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
