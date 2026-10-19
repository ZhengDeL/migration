#!/usr/bin/env python3
"""Use the pinned upstream SFT entry with bounded per-frame metadata reuse."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sources/LLaMA-Factory/src'))
from drivelm_metadata_cache import install_metadata_cache

# Also installed when a preprocessing worker imports this module under spawn.
install_metadata_cache()

if __name__ == '__main__':
    from llamafactory.train.tuner import run_exp

    print('DriveLM: original SFT entry with one-frame image metadata reuse.', flush=True)
    run_exp()

