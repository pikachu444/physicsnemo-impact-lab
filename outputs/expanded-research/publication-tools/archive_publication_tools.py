"""Preserve publication tools and their small encoding check, without ML imports."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

workspace = Path(__file__).resolve().parents[2]
source = workspace / "work/expanded-coordination"
target = workspace / "outputs/expanded-research/publication-tools"
target.mkdir(parents=True, exist_ok=True)
names = ("package_evidence.py", "verify_publication.py", "check_publication_redaction.py",
         "archive_publication_tools.py", "scan_publication.py")
rows = []
for name in names:
    shutil.copyfile(source / name, target / name)
    data = (target / name).read_bytes()
    rows.append({"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
result = subprocess.check_output([sys.executable, "-S", str(source / "check_publication_redaction.py")])
(target / "redaction-selfcheck.json").write_bytes(result)
(target / "source-manifest.json").write_text(json.dumps({
    "scope": "Exact publication helper sources, not additional model runs.",
    "files": rows}, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"source_files": len(rows), "selfcheck": json.loads(result)}))
