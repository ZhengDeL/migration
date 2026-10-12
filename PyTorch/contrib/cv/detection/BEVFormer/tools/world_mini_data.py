import hashlib

def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda : stream.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()

def require_hash(path, expected):
    actual = digest(path)
    if actual != expected:
        raise RuntimeError('Data checksum mismatch: ' + str(path))
    return actual
