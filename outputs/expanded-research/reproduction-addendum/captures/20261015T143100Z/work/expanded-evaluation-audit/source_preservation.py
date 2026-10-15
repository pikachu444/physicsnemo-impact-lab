"""Content-addressed, append-only source and prior-record preservation."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time


def preserve(path, *, record=False):
    path = Path(path).resolve()
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    folder = Path(__file__).resolve().parent/("record-snapshots" if record else "source-snapshots")/digest
    folder.mkdir(parents=True, exist_ok=True)
    target = folder/path.name
    try:
        with target.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        for attempt in range(6):
            if target.read_bytes() == data:
                break
            if attempt == 5:
                raise RuntimeError("Immutable snapshot differs: "+str(target))
            time.sleep(.05)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    return {"file": path.name, "sha256": digest, "bytes": len(data), "snapshot": str(target),
        "preserved_utc": datetime.now(timezone.utc).isoformat(), "policy": "content-addressed exact bytes; never overwrite an existing snapshot"}


def sources(*paths):
    return {Path(path).name: preserve(path) for path in paths}
