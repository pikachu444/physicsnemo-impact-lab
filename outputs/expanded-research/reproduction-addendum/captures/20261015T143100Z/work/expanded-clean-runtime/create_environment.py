"""Create one absent isolated venv, without pip or inherited packages."""
import argparse
import json
from pathlib import Path
import sys
import venv

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--workspace", type=Path, required=True)
parser.add_argument("--build-venv", action="store_true")
args = parser.parse_args()
assert sys.flags.no_site
root = args.workspace.resolve()
target = root / "work/expanded-clean-runtime" / ("build-venv" if args.build_venv else "venv")
if target.exists() or target.is_symlink():
    raise FileExistsError("Existing environment preserved; explicit recovery review required")
venv.EnvBuilder(system_site_packages=False, clear=False, with_pip=False).create(target)
assert "include-system-site-packages = false" in (target / "pyvenv.cfg").read_text()
print(json.dumps({"status": "isolated_venv_created_without_pip", "build_environment": args.build_venv, "system_site_packages": False, "framework_imported": False}))
