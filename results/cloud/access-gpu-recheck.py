import datetime, hashlib, json, os, pathlib, resource, shutil, stat, time
import urllib.request, urllib.error, urllib.parse

started = time.perf_counter()
out = pathlib.Path(__file__).with_suffix(".json")
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None
opener = urllib.request.build_opener(NoRedirect)
def request(url):
    rows = []
    for _ in range(6):
        t = time.perf_counter()
        row = {"method": "GET", "url": url, "timeout_seconds": 20}
        try:
            try:
                response = opener.open(url, timeout=20)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                row["response_code"] = response.code
                location = response.headers.get("Location")
                if location:
                    target = urllib.parse.urljoin(url, location)
                    host = urllib.parse.urlsplit(target).hostname
                    row["redirect_host"] = host
                    row["redirect_path"] = urllib.parse.urlsplit(target).path
                    row["redirect_query_present"] = bool(urllib.parse.urlsplit(target).query)
                    row["followed"] = host == "huggingface.co"
                else:
                    data = response.read(2 * 1024 * 1024 + 1)
                    row["bytes_read"] = len(data)
                    row["complete_body"] = len(data) <= 2 * 1024 * 1024
                    row["sha256_received_bytes"] = hashlib.sha256(data).hexdigest()
                    if row["complete_body"]:
                        try:
                            obj = json.loads(data)
                            row["json_valid"] = True
                            row["json_type"] = type(obj).__name__
                            if isinstance(obj, dict):
                                row["top_level_keys"] = list(obj)[:30]
                                if "sha" in obj:
                                    row["dataset_revision"] = obj["sha"]
                        except (ValueError, UnicodeError):
                            row["json_valid"] = False
        except Exception as exc:
            row["response_code"] = None
            row["error"] = str(exc)
        row["elapsed_seconds"] = time.perf_counter() - t
        rows.append(row)
        if row.get("followed"):
            url = target
        else:
            break
    return rows

def read(path):
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError as exc:
        return {"error": str(exc)}

pci_root = pathlib.Path("/sys/bus/pci/devices")
pci = []
errors = []
try:
    devices = list(pci_root.iterdir())
    for device in devices:
        vendor = read(device / "vendor")
        if isinstance(vendor, dict):
            errors.append({"path": str(device), **vendor})
        elif vendor.lower() == "0x10de":
            pci.append({"address": device.name, "vendor": vendor,
                        "device": read(device / "device"), "class": read(device / "class")})
except OSError as exc:
    errors.append({"error": str(exc)})
nodes = []
for path in pathlib.Path("/dev").glob("nvidia*"):
    try:
        st = path.stat()
        nodes.append({"path": str(path), "character_device": stat.S_ISCHR(st.st_mode),
                      "readable": os.access(path, os.R_OK), "writable": os.access(path, os.W_OK)})
    except OSError as exc:
        nodes.append({"path": str(path), "error": str(exc)})
result = {
    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "command": ["python3", str(pathlib.Path(__file__).resolve())],
    "source_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    "network": {
        "dataset_api": request("https://huggingface.co/api/datasets/AIRBORNEPANDA/BumperBeamCrashExample"),
        "global_features": request("https://huggingface.co/datasets/AIRBORNEPANDA/BumperBeamCrashExample/resolve/main/CURATED_DATA_VTP/GLOBAL_FEATURES.json"),
    },
    "gpu_exposure": {"dev_nvidia_nodes": nodes, "pci_devices_inspected": len(devices) if 'devices' in globals() else None,
                     "pci_nvidia_devices": pci, "pci_inspection_errors": errors,
                     "nvidia_smi_path": shutil.which("nvidia-smi"), "nvcc_path": shutil.which("nvcc")},
    "cgroup": {name: read("/sys/fs/cgroup/" + name) for name in [
        "cpu.max", "cpuset.cpus.effective", "memory.max", "memory.current",
        "memory.swap.max", "pids.max", "cgroup.controllers"]},
    "process_cgroup": read("/proc/self/cgroup"),
    "visible_cpu_count": os.cpu_count(),
    "training_run": False,
}
result["elapsed_seconds"] = time.perf_counter() - started
result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
text = json.dumps(result, ensure_ascii=False, indent=2)
out.write_text(text + "\n")
print(text)
print("Saved:", out)

