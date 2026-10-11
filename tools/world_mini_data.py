"""Pinned official assets and subset for AlgEngine functional training."""

import hashlib

WORLD_REVISION = 'd0e88d858533ca407a3387a26577c083aa7b3617'
OPENSCENE_REVISION = 'a76f840b65e972bc45e56c2adced897498e9a026'
SCOPE = 'Official NAVTEST samples reused for functional training only; no heldout metric'
LOGS = (
    '2021.05.25.15.59.03_veh-30_04463_04606',
    '2021.09.09.17.18.51_veh-48_01173_01237',
    '2021.09.16.19.12.04_veh-42_01221_01380',
)
ASSETS = (
    ('OpenDriveLab/WorldEngine', WORLD_REVISION,
     'data/alg_engine/ckpts/track_map_nuplan_r50_navtrain_50pct_bs1x8.pth',
     319956525, '2a81595fcee9df0ef93e9f76d7e735752786802f8a907c0c06949afb03ecea7a'),
    ('OpenDriveLab/WorldEngine', WORLD_REVISION,
     'data/alg_engine/merged_infos_navformer/nuplan_openscene_navtest.pkl',
     848516627, '59a8bdf1a9a2c6066dd562333679f54ffee9cbfe9cb3420bab27dce62c5da21b'),
    ('OpenDriveLab/WorldEngine', WORLD_REVISION,
     'data/alg_engine/pdms_cache/pdm_8192_gt_cache_navtest.pkl',
     2791314758, '58d82e73bc3dfaa6464ea7d90dee86f8e0139ea1679145d610a9babf86fe7316'),
    ('OpenDriveLab/OpenScene', OPENSCENE_REVISION,
     'openscene-v1.1/openscene_sensor_test_camera/openscene_sensor_test_camera_30.tgz',
     1629921998, '67a9c6ca0cb7ed650e93e2790425f8676f4749125e72f6f9f9f003eedfc36c57'),
)


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def require_hash(path, expected):
    actual = digest(path)
    if actual != expected:
        raise RuntimeError('Data checksum mismatch: ' + str(path))
    return actual
