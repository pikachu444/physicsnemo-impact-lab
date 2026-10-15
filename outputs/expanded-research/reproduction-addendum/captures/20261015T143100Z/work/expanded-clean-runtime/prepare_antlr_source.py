"""Preserve and statically audit the approved ANTLR sdist; never build or install it."""
from __future__ import annotations

import sys

assert sys.flags.no_site, "Run with python -S to disable site and .pth initialization"

import argparse
import ast
import configparser
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tomllib
import urllib.parse
import urllib.request
import uuid

NAME = "antlr4-python3-runtime"
VERSION = "4.9.3"
ARCHIVE_NAME = NAME + "-" + VERSION + ".tar.gz"
TOP_DIRECTORY = NAME + "-" + VERSION
EXPECTED_SHA256 = "f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b"
EXPECTED_URL = "https://files.pythonhosted.org/packages/3e/38/7859ff46355f76f8d19459005ca000b6e7012f2f1ca597746cbcd1fbfe5e/" + ARCHIVE_NAME
APPROVED_SETUP_SHA256 = "00ce3bb1670e8e4961d9ef2414dfa6ac467b52465382841b432578f91733ec67"
APPROVED_CLI_SHA256 = "02d4202c724d47965d015e9d62ad2f243359b840516db9629ce964d4e1150461"
APPROVED_INACTIVE_REQUIREMENTS = b'\n[:python_version < "3.5"]\ntyping\n'
PREVIOUS_CHECKER_SHA256 = "6aca0b4e294e1ef795709d34e1516f0acc00ce604cc468f4fc0ce3a6619ae4e1"
CHUNK_BYTES = 1024**2
MAX_SOURCE_BYTES = 8 * 1024**2
MAX_EXPANDED_BYTES = 16 * 1024**2
MAX_MEMBER_BYTES = 2 * 1024**2
MAX_MEMBERS = 1000
PACKAGING_FILES = {"setup.py", "setup.cfg", "pyproject.toml", "MANIFEST.in"}
PACKAGING_CASES = {name.casefold(): name for name in PACKAGING_FILES}
EGG_INFO = "antlr4_python3_runtime.egg-info"
NATIVE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".pyx", ".pxd", ".cu", ".f", ".f90", ".so", ".pyd", ".dll", ".lib", ".a", ".o", ".obj", ".exe"}
HOOK_FILES = {"sitecustomize.py", "usercustomize.py", "cmakelists.txt", "makefile", "meson.build", "cargo.toml", "build.rs", "configure", "entry_points.txt"}
SETUP_KEYS = {"name", "version", "packages", "package_dir", "description", "long_description", "long_description_content_type", "url", "download_url", "author", "author_email", "maintainer", "maintainer_email", "license", "classifiers", "keywords", "python_requires", "install_requires", "extras_require", "include_package_data", "package_data", "exclude_package_data", "zip_safe", "platforms", "provides", "obsoletes", "project_urls", "scripts"}
NETWORK_MODULES = {"socket", "urllib", "requests", "http", "httpx", "aiohttp", "ftplib", "subprocess"}


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def public(value, root):
    value = str(value).replace("\\", "/")
    for path, token in ((root, "${WORKSPACE}"), (Path(sys.base_prefix), "${PYTHON_HOME}"), (Path.home(), "${USER_HOME}")):
        value = re.sub(re.escape(str(path).replace("\\", "/")), lambda match: token, value, flags=re.I)
    return value


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def publish_new(temporary, target):
    # A complete file becomes visible atomically; an existing target is never replaced.
    os.link(temporary, target)
    temporary.unlink()


def save_new(path, value):
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    publish_new(temporary, path)


def source_identity(root):
    output = root / "outputs/expanded-research/reproduction-clean-runtime"
    known = json.loads((output / "cached-wheel-inventory.json").read_text(encoding="utf-8"))["wheels"]
    origins = [row["origin"] for row in known if row["filename"] == "antlr4_python3_runtime-4.9.3-py3-none-any.whl"]
    if len(origins) != 2 or origins[0] != origins[1]:
        raise ValueError("Expected the two identical recorded ANTLR source origins")
    origin = origins[0]
    if origin.get("url") != EXPECTED_URL or origin.get("archive_info", {}).get("hashes", {}).get("sha256") != EXPECTED_SHA256:
        raise ValueError("Recorded ANTLR origin does not match the single approved exact source")
    metadata_path = root / "work/expanded-clean-runtime/metadata/antlr4-python3-runtime__4.9.3.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata_url = "https://pypi.org/pypi/antlr4-python3-runtime/4.9.3/json"
    if metadata.get("request_url") != expected_metadata_url or metadata.get("http_status") != 200:
        raise ValueError("Exact official source metadata is missing or mismatched")
    document = metadata["document"]
    if canonical(document["info"]["name"]) != NAME or document["info"]["version"] != VERSION:
        raise ValueError("Exact official source metadata identity mismatch")
    matches = [row for row in document["urls"] if row.get("filename") == ARCHIVE_NAME and row.get("packagetype") == "sdist"]
    if len(matches) != 1 or matches[0].get("url") != EXPECTED_URL or matches[0].get("digests", {}).get("sha256") != EXPECTED_SHA256:
        raise ValueError("Publisher source URL/SHA256 does not agree with the recorded origin")
    size = matches[0]["size"]
    if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAX_SOURCE_BYTES:
        raise ValueError("Publisher source size is outside the bounded source budget")
    return {"name": NAME, "version": VERSION, "filename": ARCHIVE_NAME, "url": EXPECTED_URL,
            "published_sha256": EXPECTED_SHA256, "published_bytes": size, "origin": origin,
            "metadata_url": expected_metadata_url, "metadata_cache": public(metadata_path, root)}


def hash_stream(stream, output=None, limit=MAX_SOURCE_BYTES):
    digest, size = hashlib.sha256(), 0
    while chunk := stream.read(min(CHUNK_BYTES, limit - size + 1)):
        size += len(chunk)
        if size > limit:
            raise ValueError("Source stream exceeds its exact published byte size")
        digest.update(chunk)
        if output is not None:
            output.write(chunk)
    return size, digest.hexdigest()


class SourceRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org":
            raise ValueError("Refusing source redirect outside official files.pythonhosted.org HTTPS")
        return super().redirect_request(request, response, code, message, headers, newurl)


def acquire_source(path, identity, root):
    expected_size = identity["published_bytes"]
    if path.exists():
        if not path.is_file() or path.stat().st_size != expected_size:
            raise ValueError("Existing preserved source has the wrong type or size; it will not be replaced")
        with path.open("rb") as stream:
            size, digest = hash_stream(stream, limit=expected_size)
        if digest != EXPECTED_SHA256:
            raise ValueError("Existing preserved source SHA256 mismatch; it will not be replaced")
        return {"kind": "existing_preserved_exact_source", "path": public(path, root), "bytes": size, "sha256": digest}
    cache_digest = hashlib.sha224(EXPECTED_URL.encode("utf-8")).hexdigest()
    cached = Path.home() / "AppData/Local/pip/Cache/http-v2" / Path(*cache_digest[:5]) / (cache_digest + ".body")
    cache_check = {"path": public(cached, root), "present": cached.is_file(), "verified": False}
    source = None
    if cached.is_file():
        cache_check["bytes"] = cached.stat().st_size
        if cached.stat().st_size == expected_size:
            with cached.open("rb") as stream:
                _, digest = hash_stream(stream, limit=expected_size)
            cache_check["sha256"] = digest
            cache_check["verified"] = digest == EXPECTED_SHA256
            if cache_check["verified"]:
                source = cached.open("rb")
    if source is None:
        request = urllib.request.Request(EXPECTED_URL, headers={"User-Agent": "ApprovedAntlrSourceAudit/1.0", "Accept-Encoding": "identity"})
        source = urllib.request.build_opener(SourceRedirects()).open(request, timeout=30)
        if source.status != 200:
            source.close()
            raise ValueError("Exact official source request did not return HTTP 200")
        length = source.headers.get("Content-Length")
        if length and int(length) != expected_size:
            source.close()
            raise ValueError("Exact official source Content-Length differs from published bytes")
        acquisition = "single_official_source_request"
    else:
        acquisition = "verified_pip_http_cache"
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with source, temporary.open("xb") as output:
        size, digest = hash_stream(source, output, expected_size)
        output.flush()
        os.fsync(output.fileno())
    if size != expected_size or digest != EXPECTED_SHA256:
        raise ValueError("Exact source size/SHA256 mismatch; unpublished partial is preserved")
    publish_new(temporary, path)
    return {"kind": acquisition, "path": public(path, root), "bytes": size, "sha256": digest, "pip_cache_check": cache_check}


def safe_member_name(name):
    if not isinstance(name, str) or not name or "\\" in name or name.startswith("/") or "\x00" in name:
        raise ValueError("Unsafe archive path")
    name = name.removesuffix("/")
    parts = name.split("/")
    reserved = {"con", "prn", "aux", "nul", *("com" + str(i) for i in range(1, 10)), *("lpt" + str(i) for i in range(1, 10))}
    if parts[0] != TOP_DIRECTORY or any(not part or part in {".", ".."} or part.endswith((".", " ")) or re.search(r'[\x00-\x1f<>:"|?*]', part) or part.split(".")[0].casefold() in reserved for part in parts):
        raise ValueError("Archive path escapes or is not a safe Windows path: " + name)
    return PurePosixPath(*parts)


def literal_value(node, values):
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = literal_value(node.left, values), literal_value(node.right, values)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        raise ValueError("Only static string concatenation is permitted")
    if isinstance(node, (ast.List, ast.Tuple)):
        return [literal_value(item, values) for item in node.elts]
    if isinstance(node, ast.Dict):
        result = {literal_value(key, values): literal_value(value, values) for key, value in zip(node.keys, node.values)}
        if not all(isinstance(key, str) for key in result):
            raise ValueError("Packaging dictionary keys must be literal strings")
        return result
    if isinstance(node, (ast.Constant, ast.UnaryOp)):
        result = ast.literal_eval(node)
        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError("Nonfinite packaging literal is unsupported")
        if result is None or isinstance(result, (str, int, float, bool)):
            return result
    raise ValueError("Packaging values must be JSON-compatible static literals")


def relative_packaging_path(value, patterns=False):
    if not isinstance(value, str) or value.startswith("/") or "\\" in value or re.search(r"[\x00-\x1f:]", value):
        return False
    if value == "":
        return True
    if any(part in {"", ".", ".."} for part in value.split("/")):
        return False
    return patterns or not re.search(r'[<>"|?*]', value)


def audit_setup(text):
    tree = ast.parse(text, filename="setup.py")
    imports, calls, findings, values, bindings, setup_fields = [], [], [], {}, set(), None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(("." * node.level) + (node.module or "") + ":" + ",".join(alias.name for alias in node.names))
        elif isinstance(node, ast.Call):
            calls.append(ast.unparse(node.func))
    for statement in tree.body:
        try:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                continue
            if isinstance(statement, ast.ImportFrom) and statement.level == 0 and statement.module == "setuptools" and len(statement.names) == 1 and statement.names[0].name == "setup" and statement.names[0].asname is None:
                if "setup" in bindings:
                    raise ValueError("setup binding repeated")
                bindings.add("setup")
            elif isinstance(statement, ast.Import) and len(statement.names) == 1 and statement.names[0].name == "setuptools" and statement.names[0].asname is None:
                if "setuptools" in bindings:
                    raise ValueError("setuptools binding repeated")
                bindings.add("setuptools")
            elif isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                target = statement.targets[0].id
                if target in {"setup", "setuptools", "__builtins__"} or target in values:
                    raise ValueError("Packaging binding cannot be replaced")
                values[target] = literal_value(statement.value, values)
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
                function = ast.unparse(call.func)
                if function not in {"setup", "setuptools.setup"} or function.split(".")[0] not in bindings or setup_fields is not None or call.args:
                    raise ValueError("Only one explicitly imported setuptools.setup call is permitted")
                setup_fields = {}
                for keyword in call.keywords:
                    if keyword.arg not in SETUP_KEYS or keyword.arg in setup_fields:
                        raise ValueError("Unexpected/dynamic setup keyword: " + str(keyword.arg))
                    setup_fields[keyword.arg] = literal_value(keyword.value, values)
            else:
                raise ValueError("Unexpected top-level packaging statement: " + type(statement).__name__)
        except (ValueError, TypeError, SyntaxError) as error:
            findings.append({"line": statement.lineno, "reason": str(error)})
    if setup_fields is None:
        findings.append({"reason": "No statically resolved setup declaration"})
    else:
        if setup_fields.get("name") != NAME or setup_fields.get("version") != VERSION:
            findings.append({"reason": "setup name/version differs from exact approved source"})
        if "scripts" in setup_fields and setup_fields["scripts"] != ["bin/pygrun"]:
            findings.append({"reason": "Only the exact approved bin/pygrun literal copy is permitted"})
        if "install_requires" in setup_fields and setup_fields["install_requires"] != ["typing ; python_version<'3.5'"]:
            findings.append({"reason": "Only the exact inactive Python-before-3.5 typing marker is permitted"})
        packages = setup_fields.get("packages")
        if not isinstance(packages, list) or not packages or any(not isinstance(package, str) or not re.fullmatch(r"antlr4(?:\.[A-Za-z_][A-Za-z0-9_]*)*", package) for package in packages):
            findings.append({"reason": "Expected an explicit list containing only antlr4 Python packages"})
        package_dir = setup_fields.get("package_dir", {})
        if not isinstance(package_dir, dict) or any(not isinstance(key, str) or (key != "" and not re.fullmatch(r"antlr4(?:\.[A-Za-z_][A-Za-z0-9_]*)*", key)) or not relative_packaging_path(value) for key, value in package_dir.items()):
            findings.append({"reason": "Unexpected or unsafe literal package_dir mapping"})
        for keyword in ("package_data", "exclude_package_data"):
            data = setup_fields.get(keyword, {})
            if not isinstance(data, dict) or any(not isinstance(items, list) or any(not relative_packaging_path(item, patterns=True) for item in items) for items in data.values()):
                findings.append({"reason": "Unsafe or dynamic package-data paths: " + keyword})
    network_imports = sorted(item for item in imports if item.split(":")[0].split(".")[0] in NETWORK_MODULES)
    return {"imports": sorted(imports), "calls": sorted(calls), "setup_fields": setup_fields,
            "network_imports": network_imports, "custom_command_or_extension_keywords": sorted({keyword.arg for node in ast.walk(tree) if isinstance(node, ast.Call) for keyword in node.keywords if keyword.arg in {"cmdclass", "ext_modules", "setup_requires", "entry_points"}}),
            "literal_copied_scripts": (setup_fields or {}).get("scripts", []),
            "findings": findings}


def audit_cli(data):
    digest = hashlib.sha256(data).hexdigest()
    tree = ast.parse(data.decode("utf-8"), filename="bin/pygrun")
    imports, calls, findings, main_guards = [], [], [], 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(("." * node.level) + (node.module or "") + ":" + ",".join(alias.name for alias in node.names))
        elif isinstance(node, ast.Call):
            calls.append(ast.unparse(node.func))
    if digest != APPROVED_CLI_SHA256 or len(data) != 6111:
        findings.append("CLI bytes differ from the exact approved source member")
    if sorted(imports) != ["antlr4:*", "importlib", "optparse", "os", "sys"]:
        findings.append("CLI imports differ from the exact inspected CLI")
    if any(name in {"eval", "exec", "compile", "os.system", "os.popen"} or name.startswith(("subprocess.", "socket.", "urllib.", "requests.", "http.", "httpx.", "aiohttp.")) for name in calls):
        findings.append("CLI has an unexpected execution/network/process call")
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "__author__" and isinstance(node.value, ast.Constant) and node.value.value == "jszheng":
            continue
        if isinstance(node, ast.FunctionDef) and node.name == "beautify_lisp_string" and not node.decorator_list and not node.args.defaults and not any(node.args.kw_defaults) and node.returns is None and all(argument.annotation is None for argument in node.args.args):
            continue
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__" and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq) and len(node.test.comparators) == 1 and isinstance(node.test.comparators[0], ast.Constant) and node.test.comparators[0].value == "__main__" and not node.orelse:
            main_guards += 1
            continue
        findings.append("Unexpected CLI top-level statement: " + type(node).__name__)
    if main_guards != 1:
        findings.append("CLI must have the single inspected __main__ guard")
    return {"path": "bin/pygrun", "bytes": len(data), "sha256": digest, "imports": sorted(imports), "calls": sorted(set(calls)),
            "main_guard_count": main_guards, "script_executed": False,
            "packaging_behavior": "Literal setuptools scripts entry copies this exact file; the CLI is never imported or executed by this audit",
            "runtime_dynamic_imports": "The guarded CLI loads user-selected grammar modules only when explicitly invoked later; no such invocation occurs here",
            "findings": findings}


def audit_configuration(files):
    result, findings = {}, []
    if "setup.cfg" in files:
        parser = configparser.ConfigParser(interpolation=None, strict=True)
        parser.read_string(files["setup.cfg"])
        sections = {section: dict(parser.items(section)) for section in parser.sections()}
        result["setup.cfg"] = sections
        if parser.defaults() or any(section != "egg_info" for section in sections) or any(key not in {"tag_build", "tag_date"} or (key == "tag_build" and value != "") or (key == "tag_date" and value not in {"0", "false", "False"}) for values in sections.values() for key, value in values.items()):
            findings.append("setup.cfg is outside the inert egg_info metadata allowlist")
    if "pyproject.toml" in files:
        project = tomllib.loads(files["pyproject.toml"])
        result["pyproject.toml"] = json.loads(json.dumps(project, default=str))
        system = project.get("build-system", {})
        if not isinstance(system, dict):
            return result, ["pyproject.toml build-system must be a literal table"]
        requirements = system.get("requires", [])
        if set(project) != {"build-system"} or set(system) - {"requires", "build-backend"} or system.get("build-backend") not in {"setuptools.build_meta", "setuptools.build_meta:__legacy__"} or not isinstance(requirements, list) or any(not isinstance(item, str) or not re.fullmatch(r"(?:setuptools|wheel)(?:\s*[<>=!~]+\s*[A-Za-z0-9.*+!_-]+(?:\s*,\s*[<>=!~]+\s*[A-Za-z0-9.*+!_-]+)*)?", item) for item in requirements):
            findings.append("pyproject.toml has unexpected backend, requirements, hook paths or configuration")
    if "MANIFEST.in" in files:
        lines = [line.strip() for line in files["MANIFEST.in"].splitlines() if line.strip() and not line.lstrip().startswith("#")]
        result["MANIFEST.in"] = lines
        commands = {"include", "exclude", "recursive-include", "recursive-exclude", "global-include", "global-exclude", "graft", "prune"}
        if any(line.split()[0] not in commands or any(part.startswith("/") or "\\" in part or ":" in part or ".." in part.split("/") for part in line.split()[1:]) for line in lines):
            findings.append("MANIFEST.in has unexpected commands or escaping paths")
    return result, findings


def inspect_archive(path):
    records, selected, findings, members, names, cli, typing_metadata = [], {}, [], [], {}, None, None
    total = 0
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive:
            if len(members) >= MAX_MEMBERS:
                raise ValueError("Archive exceeds member-count budget")
            relative = safe_member_name(member.name)
            key = relative.as_posix().casefold()
            if key in names or member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE) or getattr(member, "sparse", None) or any(key.startswith("GNU.sparse") for key in member.pax_headers):
                raise ValueError("Archive contains duplicate/case-colliding paths, links or unsupported entry types")
            if not 0 <= member.size <= MAX_MEMBER_BYTES:
                raise ValueError("Archive member exceeds individual expanded-size budget")
            if member.isdir() and member.size != 0:
                raise ValueError("Archive directory unexpectedly declares file data")
            total += member.size
            if total > MAX_EXPANDED_BYTES:
                raise ValueError("Archive exceeds total expanded-size budget")
            names[key] = member.isfile()
            members.append(member)
            inside = relative.relative_to(TOP_DIRECTORY).as_posix()
            parts = relative.parts[1:]
            if member.isdir() and parts and parts[0] not in {"src", "antlr4", EGG_INFO, "bin"}:
                findings.append("Unexpected root directory/import shadow: " + inside)
            if member.isdir() and parts and parts[0] == "bin" and parts != ("bin",):
                findings.append("Unexpected CLI directory: " + inside)
            if member.isdir() and len(parts) >= 2 and parts[0] == "src" and parts[1] not in {"antlr4", EGG_INFO}:
                findings.append("Unexpected source directory/import shadow: " + inside)
            record = {"path": relative.as_posix(), "type": "file" if member.isfile() else "directory", "bytes": member.size,
                      "archive_mode": member.mode, "archive_mtime": member.mtime}
            if member.isfile():
                with archive.extractfile(member) as stream:
                    data = stream.read(MAX_MEMBER_BYTES + 1)
                if len(data) != member.size:
                    raise ValueError("Archive member size differs from extracted bytes")
                record["sha256"] = hashlib.sha256(data).hexdigest()
                basename = relative.name
                canonical_packaging = PACKAGING_CASES.get(basename.casefold())
                if len(parts) > 1 and parts[0] not in {"src", "antlr4", EGG_INFO, "bin"}:
                    findings.append("Unexpected root directory/import-shadowing member: " + inside)
                if parts and parts[0] == "bin" and inside != "bin/pygrun":
                    findings.append("Unexpected CLI member: " + inside)
                if inside == "bin/pygrun":
                    cli = audit_cli(data)
                    findings.extend("bin/pygrun: " + reason for reason in cli["findings"])
                if len(parts) > 2 and parts[0] == "src" and parts[1] not in {"antlr4", EGG_INFO}:
                    findings.append("Unexpected source-directory/import-shadowing member: " + inside)
                if len(parts) == 2 and parts[0] == "src":
                    findings.append("Unexpected source-root file/import shadow: " + inside)
                if len(parts) == 1 and canonical_packaging is None and basename not in {"PKG-INFO", "LICENSE", "COPYING", "AUTHORS", "NOTICE", "README"} and PurePosixPath(basename).suffix.lower() not in {".txt", ".md", ".rst"}:
                    findings.append("Unexpected root file/import shadow: " + inside)
                if PurePosixPath(basename).suffix.lower() in NATIVE_SUFFIXES:
                    findings.append("Native source/binary member: " + inside)
                if basename.casefold() in HOOK_FILES or basename.lower().endswith((".pth", ".bat", ".cmd", ".ps1", ".sh")) or (canonical_packaging and "/" in inside):
                    findings.append("Unexpected hook/build member: " + inside)
                if EGG_INFO in parts:
                    allowed_metadata = {"PKG-INFO", "SOURCES.txt", "dependency_links.txt", "top_level.txt", "requires.txt", "zip-safe", "not-zip-safe"}
                    inactive_typing = inside == "src/" + EGG_INFO + "/requires.txt" and data == APPROVED_INACTIVE_REQUIREMENTS
                    if inside == "src/" + EGG_INFO + "/requires.txt":
                        typing_metadata = {"path": inside, "bytes": len(data), "sha256": record["sha256"], "content": data.decode("utf-8"),
                                           "exact_inactive_metadata_match": inactive_typing, "active_on_target_python_3_13": False, "marker": "python_version < '3.5'"}
                    if basename not in allowed_metadata or (basename == "dependency_links.txt" and data.strip()) or (basename == "requires.txt" and not inactive_typing):
                        findings.append("Unexpected egg-info dependency/hook metadata: " + inside)
                if canonical_packaging and len(parts) == 1:
                    if basename != canonical_packaging:
                        findings.append("Noncanonical Windows packaging filename casing: " + inside)
                    if len(data) > 256 * 1024:
                        raise ValueError("Packaging file exceeds static-audit budget")
                    selected[canonical_packaging] = {"archive_path": inside, "bytes": len(data), "sha256": record["sha256"], "content": data.decode("utf-8-sig")}
            records.append(record)
        for name, is_file in names.items():
            if any(names.get(parent.as_posix()) is True for parent in PurePosixPath(name).parents):
                raise ValueError("Archive contains a file used as another member's directory")
        if "setup.py" not in selected:
            findings.append("Expected top-level setup.py is absent")
            setup = None
        else:
            if selected["setup.py"]["sha256"] != APPROVED_SETUP_SHA256:
                findings.append("setup.py differs from the exact inspected publisher source member")
            try:
                setup = audit_setup(selected["setup.py"]["content"])
                findings.extend("setup.py: " + item["reason"] for item in setup["findings"])
                if not setup["findings"]:
                    fields = setup["setup_fields"]
                    package_dir = fields.get("package_dir", {})
                    for package in fields["packages"]:
                        prefixes = [key for key in package_dir if key == "" or package == key or package.startswith(key + ".")]
                        prefix = max(prefixes, key=len) if prefixes else ""
                        base = package_dir.get(prefix, "")
                        remainder = package[len(prefix):].lstrip(".")
                        package_init = PurePosixPath(TOP_DIRECTORY, base, *remainder.split("."), "__init__.py").as_posix().casefold()
                        if names.get(package_init) is not True:
                            findings.append("Declared Python package __init__.py is absent: " + package)
                if (setup.get("setup_fields") or {}).get("scripts") != ["bin/pygrun"] or cli is None:
                    findings.append("The exact inspected setup scripts declaration and CLI member must agree")
            except (SyntaxError, ValueError, TypeError) as error:
                setup = {"parse_error": str(error)}
                findings.append("setup.py could not be statically resolved: " + str(error))
        try:
            configuration, errors = audit_configuration({name: row["content"] for name, row in selected.items()})
            findings.extend(errors)
        except (ValueError, TypeError, configparser.Error, tomllib.TOMLDecodeError) as error:
            configuration = {"parse_error": str(error)}
            findings.append("Packaging configuration parse failed: " + str(error))
        if typing_metadata is None:
            findings.append("The exact inspected inactive typing metadata is absent")
    facts = {"archive_top_directory": TOP_DIRECTORY, "archive_member_count": len(records), "expanded_bytes": total,
             "members": sorted(records, key=lambda row: row["path"]), "selected_packaging_files": dict(sorted(selected.items())),
             "packaging_ast": setup, "packaging_configuration": configuration,
             "copied_cli_script_audit": cli,
             "inactive_typing_requirement": typing_metadata,
             "native_member_suffixes_checked": sorted(NATIVE_SUFFIXES), "unexpected_build_hook_names_checked": sorted(HOOK_FILES),
             "findings": sorted(findings), "packaging_shape_passed": not findings,
             "analysis_scope": "Static packaging AST/configuration and archive contents only; no source/build/runtime code executed",
             "limitations": ["A static packaging allowlist does not prove runtime safety or byte-identical wheel reproducibility", "Build-tool versions and deterministic build environment remain the parent pipeline's responsibility"]}
    return facts, members


def extract_new(path, destination, members, records):
    destination.mkdir(exist_ok=False)
    destination_root = destination.resolve()
    expected_hashes = {row["path"]: row["sha256"] for row in records if row["type"] == "file"}
    with tarfile.open(path, mode="r:gz") as archive:
        for member in members:
            relative = safe_member_name(member.name)
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination_root):
                raise ValueError("Extraction target escaped the new source directory")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("xb") as output:
                    size, digest = hash_stream(source, output, member.size)
                if size != member.size or digest != expected_hashes[relative.as_posix()]:
                    raise ValueError("Extraction differs from statically audited member bytes/hash")
                os.utime(target, (member.mtime, member.mtime))


def prepare(root):
    stage = root / "work/expanded-clean-runtime/antlr-source"
    destination = stage / "extracted-r2"
    output = root / "outputs/expanded-research/reproduction-clean-runtime"
    audit_path = output / "antlr-source-audit-r2.json"
    if destination.exists() or audit_path.exists():
        raise ValueError("Source extraction/audit destination already exists; no existing artifact will be replaced")
    if not stage.resolve().is_relative_to(root.resolve()):
        raise ValueError("Source staging path resolves outside the explicit workspace")
    identity = source_identity(root)
    stage.mkdir(parents=True, exist_ok=True)
    path = stage / ARCHIVE_NAME
    print(json.dumps({"phase": "approved_source_preparation_started", "pid": os.getpid(), "build": False, "install": False}), flush=True)
    acquisition = acquire_source(path, identity, root)
    previous_path = output / "antlr-source-audit.json"
    previous_audit = None
    if previous_path.exists():
        previous_bytes = previous_path.read_bytes()
        previous = json.loads(previous_bytes.decode("utf-8"))
        if previous.get("status") != "static_audit_failed" or previous.get("source", {}).get("published_sha256") != EXPECTED_SHA256:
            raise ValueError("Existing original audit is not the expected preserved failed exact-source audit")
        previous_audit = {"path": public(previous_path, root), "sha256": hashlib.sha256(previous_bytes).hexdigest(), "status": previous["status"]}
    try:
        facts, members = inspect_archive(path)
    except (ValueError, SyntaxError, TypeError, UnicodeError, tarfile.TarError) as error:
        facts, members = {"packaging_shape_passed": False, "structural_validation_passed": False,
                          "findings": ["Archive/static audit failed closed: " + str(error)],
                          "source_code_executed": False}, []
    result = {"schema": "antlr-exact-source-audit-v1", "status": "static_audit_failed" if facts["findings"] else "prepared_not_built",
              "source": identity, "acquisition": acquisition, "source_audit": facts,
              "source_audit_sha256": hashlib.sha256(json_bytes(facts)).hexdigest(),
              "preserved_previous_failed_audit": previous_audit,
              "checker_correction": {"scope": "Exact publisher CLI literal copy and inactive typing marker only; original source bytes unchanged", "previous_checker_sha256": PREVIOUS_CHECKER_SHA256, "current_checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "extraction_path": public(destination, root), "source_root": public(destination / TOP_DIRECTORY, root),
              "actions": {"source_request_permitted": True, "source_code_executed": False, "venv": False, "pip": False, "build": False, "install": False, "framework_import": False}}
    if not facts["findings"]:
        extract_new(path, destination, members, facts["members"])
    save_new(audit_path, result)
    print(json.dumps({"status": result["status"], "pid": os.getpid(), "audit_path": public(audit_path, root), "findings": facts["findings"]}), flush=True)
    return 1 if facts["findings"] else 0


def selfcheck():
    assert safe_member_name(TOP_DIRECTORY + "/src/antlr4/__init__.py").name == "__init__.py"
    for bad in ("../escape", "/absolute", TOP_DIRECTORY + "/../escape", TOP_DIRECTORY + "/C:/escape", TOP_DIRECTORY + "\\escape", TOP_DIRECTORY + "/CON", TOP_DIRECTORY + "/trailing.", TOP_DIRECTORY + "/a//b"):
        try:
            safe_member_name(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsafe archive path accepted: " + bad)
    safe = "from setuptools import setup\nsetup(name='antlr4-python3-runtime', version='4.9.3', packages=['antlr4'], package_dir={'': 'src'})\n"
    assert not audit_setup(safe)["findings"]
    assert audit_setup(safe.replace("packages=['antlr4']", "packages=['antlr4'], cmdclass={'x': object}"))["findings"]
    assert not audit_setup(safe.replace("packages=['antlr4']", "packages=['antlr4'], scripts=['bin/pygrun']"))["findings"]
    assert audit_setup(safe.replace("packages=['antlr4']", "packages=['antlr4'], scripts=['bin/other']"))["findings"]
    assert not audit_setup(safe.replace("packages=['antlr4']", "packages=['antlr4'], install_requires=[\"typing ; python_version<'3.5'\"]"))["findings"]
    assert audit_setup(safe.replace("packages=['antlr4']", "packages=['antlr4'], install_requires=['requests']"))["findings"]
    assert audit_setup("import subprocess\n" + safe)["findings"]
    assert audit_setup(safe.replace("packages=['antlr4']", "packages=__import__('os').system('echo denied')"))["findings"]
    assert audit_setup(safe.replace("package_dir={'': 'src'}", "package_dir={'': '../outside'}"))["findings"]
    assert audit_setup(safe.replace("packages=['antlr4']", "packages=['antlr4'], package_data={'antlr4':['../../outside']}"))["findings"]
    assert not audit_configuration({"setup.cfg": "[egg_info]\ntag_build =\ntag_date = 0\n"})[1]
    assert audit_configuration({"pyproject.toml": '[build-system]\nrequires=["setuptools"]\nbuild-backend="custom.backend"\n'})[1]
    print(json.dumps({"status": "selfcheck_passed", "network": False, "writes": False, "build": False, "site_initialization_disabled": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-source-approved", action="store_true", help="Permit only the exact approved source retrieval, static audit and new-path extraction")
    mode.add_argument("--selfcheck", action="store_true", help="Run in-memory path/AST/configuration assertions only")
    arguments = parser.parse_args()
    if arguments.selfcheck:
        selfcheck()
        return 0
    try:
        return prepare(arguments.workspace.resolve())
    except Exception as error:
        print(json.dumps({"status": "source_preparation_failed", "pid": os.getpid(), "error_type": type(error).__name__, "error": public(error, arguments.workspace.resolve()), "build": False, "install": False}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
