"""Check packaged text for this workstation path and credential-shaped strings.

Report locations and pattern names only, never matched content. This is a narrow
publication check, not a comprehensive secret scanner or security certification.
"""
import json
from pathlib import Path
import re
import sys


def scan(root):
    patterns = {
        "local_account_path": re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+pikac(?:[\\/]|\b)", re.I),
        "github_classic_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
        "github_fine_grained_token": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{50,}\b"),
        "openai_key_shape": re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{40,}\b"),
        "private_key_header": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "aws_access_key_shape": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    }
    text_extensions = {".md", ".json", ".jsonl", ".csv", ".py", ".ps1", ".txt", ".log", ".yaml", ".yml", ".toml", ".out"}
    files = 0
    findings = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in text_extensions:
            continue
        files += 1
        with path.open(encoding="utf-8-sig", errors="strict") as stream:
            for number, line in enumerate(stream, 1):
                for name, pattern in patterns.items():
                    if pattern.search(line):
                        findings.append({"path": path.relative_to(root).as_posix(), "line": number, "pattern": name})
    return {"scope": "narrow packaged-text account-path/credential-shape check", "text_files": files, "findings": findings}


if __name__ == "__main__":
    result = scan(Path(sys.argv[1]).resolve(strict=True))
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(bool(result["findings"]))
