from contextlib import contextmanager
import os
from pathlib import Path
import tempfile


@contextmanager
def atomic_open(path, mode="w", **kwargs):


    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, mode, **kwargs) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
