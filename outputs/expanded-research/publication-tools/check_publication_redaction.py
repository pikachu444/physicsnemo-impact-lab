"""Check actual log escaping cases without importing any research runtime."""
import hashlib
import json
from pathlib import Path
import runpy

source = Path(__file__).with_name("package_evidence.py")
redact = runpy.run_path(str(source), run_name="publication_library")["redact_local_paths"]
workspace = Path(__file__).resolve().parents[2]
home = Path.home()
checks = 0
for separator in ("\\", "\\\\", "\\\\\\\\", "/"):
    value = str(workspace).replace("\\", separator) + separator + "example.txt"
    assert redact(value, workspace, home) == "<LOCAL_WORKSPACE>" + separator + "example.txt"
    value = str(home).replace("\\", separator) + separator + "outside.txt"
    assert redact(value, workspace, home) == "<LOCAL_ACCOUNT>" + separator + "outside.txt"
    checks += 2
for value in (str(workspace).lower(), str(workspace).upper()):
    assert redact(value, workspace, home) == "<LOCAL_WORKSPACE>"
    checks += 1
encoded = json.dumps({"nested": json.dumps({"path": str(workspace / "example.txt")})})
decoded = json.loads(json.loads(redact(encoded, workspace, home))["nested"])
assert decoded["path"].startswith("<LOCAL_WORKSPACE>")
checks += 1
print(json.dumps({"checks": checks, "failures": 0,
                  "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "scope": "Path redaction only; no source evidence is modified."}))
