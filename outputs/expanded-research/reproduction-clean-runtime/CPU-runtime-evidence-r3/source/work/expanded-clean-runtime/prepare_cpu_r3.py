"""Create the approved CPU admission-only revision without changing R2."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

assert sys.flags.no_site
root = Path(__file__).resolve().parents[2]
own = root / "work/expanded-clean-runtime"
out = root / "outputs/expanded-research/reproduction-clean-runtime"
source = own / "run_guarded_r2.py"
raw = source.read_bytes()
old_sha = hashlib.sha256(raw).hexdigest()
assert old_sha == "45d6bfb697b95f8df6721d2d42146c53095dc81e062e885f9fa91d17b5fdda1a"
old = b'"preflight_min_available_bytes": int(1.5 * 1024**3)'
new = b'"preflight_min_available_bytes": int(1.0 * 1024**3)'
assert raw.count(old) == 1
changed = raw.replace(old, new)
assert changed.replace(new, old) == raw
destination = own / "run_guarded_cpu_r3.py"
with destination.open("xb") as stream:
    stream.write(changed)
protocol = {"status": "approved_CPU_admission_revision_prepared_not_executed", "created_utc": datetime.now(timezone.utc).isoformat(),
            "authority": "Root explicitly authorized the one-constant CPU-only revision and subsequent exact build/install stages.",
            "original_source": "work/expanded-clean-runtime/run_guarded_r2.py", "original_sha256": old_sha,
            "revision_source": "work/expanded-clean-runtime/run_guarded_cpu_r3.py", "revision_sha256": hashlib.sha256(changed).hexdigest(),
            "exact_change": "One preflight admission constant 1.5 GiB to 1.0 GiB; all other source bytes equal.",
            "operating_admission_bytes": 1024**3, "stable_observations": 2, "spacing_seconds": 10,
            "family_RSS_cap_bytes": 350*1024**2, "running_host_floor_bytes": 512*1024**2,
            "preflight_disk_bytes": 15*1024**3, "running_disk_floor_bytes": 5*1024**3,
            "admission_arithmetic_MiB": {"child_ceiling": 350, "host_floor": 512, "parent_and_headroom": 162, "total": 1024},
            "scope": "Existing CPU-only stage list, exact one-shot ANTLR source build and offline clean install through static checks.",
            "unchanged": ["Windows Job helper", "one-build sentinel", "runtime floors", "familyRSS ceiling", "disk bounds", "CPU1 BelowNormal", "GPU hidden", "no overwrite or automatic retry"],
            "model_minimum_RAM_claimed": False, "framework_import_authorized": False,
            "original_source_unchanged": hashlib.sha256(source.read_bytes()).hexdigest() == old_sha}
path = out / "CPU-admission-protocol-r3.json"
with path.open("x", encoding="utf-8") as stream:
    json.dump(protocol, stream, indent=2)
    stream.write("\n")
print(json.dumps({"status": protocol["status"], "revision_sha256": protocol["revision_sha256"], "protocol_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
