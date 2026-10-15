# 
# Copyright (C) 2026 SISW Siemens Industry Software Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining 
# a copy of this software and associated documentation files (the "Software"), 
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
# sell copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
# IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 
"""Read the FASTMAGI10 ANIM 2D prefix directly, using NumPy only.

Adapted from the MIT-licensed OpenCourant anim_to_vtk format reader:
https://github.com/OpenCourant/Tools/blob/c58c805696f7364f0efe80b5dd1585f70670232e/output_converters/anim_to_vtk/src/anim_to_vtk.cpp
The Vortex-Radioss reader (MPL-2.0) was inspected as a format cross-check;
its implementation is not copied here. No package installation is needed.

Scope: numbered, uncompressed FASTMAGI10 files; node geometry/vectors and
2D shell scalars/tensors. 1D/3D/SPH/hierarchy tails are deliberately unread.
Raw shell tensor components are local [xx, yy, xy], with no rotation or
cell-to-node averaging. These arrays retain binary float32 precision.
"""
import argparse
import json
from pathlib import Path
import re
import time
import numpy as np

SOURCE_COMMIT = "c58c805696f7364f0efe80b5dd1585f70670232e"


class _Cursor:
    def __init__(self, stream, size):
        self.stream, self.size, self.offsets = stream, size, {}

    def array(self, label, dtype, shape):
        dtype = np.dtype(dtype)
        count = int(np.prod(shape, dtype=np.int64))
        if count < 0 or count * dtype.itemsize > self.size - self.stream.tell():
            raise ValueError("Invalid or truncated ANIM block: " + label)
        self.offsets[label] = self.stream.tell()
        result = np.fromfile(self.stream, dtype=dtype, count=count)
        return result.astype(dtype.newbyteorder("="), copy=False).reshape(shape)

    def skip(self, label, size):
        if size < 0 or size > self.size - self.stream.tell():
            raise ValueError("Invalid or truncated ANIM block: " + label)
        self.offsets[label] = self.stream.tell()
        self.stream.seek(size, 1)

    def names(self, label, count, width):
        raw = self.array(label, "u1", (count, width))
        return [bytes(row).split(b"\0", 1)[0].decode("latin1") for row in raw]


def read_anim(path):
    """Return NumPy arrays from one frame; no VTK dependency or conversion.

    shell_connectivity contains zero-based indices into node_id/coordinates.
    shell_part_id reproduces anim_to_vtk's leading numeric part-name ID.
    shell_scalars/tensors use original ANIM names (spaces retained).
    """
    path = Path(path)
    with path.open("rb") as stream:
        c = _Cursor(stream, path.stat().st_size)
        magic = int(c.array("magic", ">i4", (1,))[0])
        if magic != 0x542C:
            raise ValueError(f"Unsupported ANIM magic 0x{magic:x}; expected FASTMAGI10")
        tm = float(c.array("time", ">f4", (1,))[0])
        c.names("header_text", 3, 81)
        flags = c.array("flags", ">i4", (10,))
        if not np.isin(flags, [0, 1]).all() or not flags[1]:
            raise ValueError("Expected numbered FASTMAGI10 file with binary flags")
        dims = c.array("dimensions", ">i4", (8,))
        if (dims < 0).any():
            raise ValueError("Negative ANIM dimension")
        n, e, p, ns, es, nv, nt, sk = map(int, dims)
        if not (n and e and p):
            raise ValueError("Expected nonempty node/shell/part data")
        c.skip("skews", sk * 6 * 2)
        coordinates = c.array("coordinates", ">f4", (n, 3))
        connectivity = c.array("shell_connectivity", ">i4", (e, 4))
        active = c.array("shell_active_raw", "u1", (e,))
        part_ends = c.array("part_ends_exclusive", ">i4", (p,))
        part_names = c.names("part_names", p, 50)
        c.skip("node_normals", n * 3 * 2)
        scalar_names = c.names("scalar_names", ns + es, 81)
        c.skip("node_scalar_values", n * ns * 4)
        scalar_values = c.array("shell_scalar_values", ">f4", (es, e))
        vector_names = c.names("vector_names", nv, 81)
        vector_values = c.array("node_vector_values", ">f4", (nv, n, 3))
        tensor_names = c.names("tensor_names", nt, 81)
        tensor_values = c.array("shell_tensor_values", ">f4", (nt, e, 3))
        if flags[0]:
            c.skip("masses", (e + n) * 4)
        node_id = c.array("node_id", ">i4", (n,))
        element_id = c.array("shell_element_id", ">i4", (e,))
        end_2d_numbering = stream.tell()
    if not np.isfinite(tm):
        raise ValueError("Nonfinite ANIM time")
    if connectivity.min() < 0 or connectivity.max() >= n:
        raise ValueError("Shell connectivity outside node array")
    if part_ends[-1] != e or (np.diff(np.r_[0, part_ends]) < 0).any():
        raise ValueError("Invalid exclusive part boundaries")
    part_ids = []
    for name in part_names:
        match = re.match(r"\s*([+-]?\d+)", name)
        part_ids.append(int(match[1]) if match else 0)
    shell_part_id = np.repeat(np.asarray(part_ids, dtype=np.int32),
                              np.diff(np.r_[0, part_ends]))
    if len(set(scalar_names)) != len(scalar_names) or len(set(tensor_names)) != len(tensor_names):
        raise ValueError("Ambiguous duplicate ANIM field names")
    vectors = dict(zip(vector_names, vector_values))
    if "Displacement" not in vectors:
        raise ValueError("Displacement vector was not requested")
    for values in (coordinates, scalar_values, vector_values, tensor_values):
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite ANIM field")
    return {
        "time": tm, "node_id": node_id, "coordinates": coordinates,
        "displacement": vectors["Displacement"],
        "shell_connectivity": connectivity, "shell_element_id": element_id,
        "shell_part_id": shell_part_id, "shell_active": active == 1,
        "shell_scalars": dict(zip(scalar_names[ns:], scalar_values)),
        "shell_tensors": dict(zip(tensor_names, tensor_values)),
        "metadata": {"magic": magic, "flags": flags.tolist(),
                     "dimensions": dims.tolist(), "offsets": c.offsets,
                     "end_2d_numbering": end_2d_numbering,
                     "file_bytes": path.stat().st_size,
                     "source_commit": SOURCE_COMMIT},
    }


def _read_ascii_vtk(path):
    """Small streaming reference reader for the official legacy ASCII output."""
    result, point_count, cell_count, association = {}, 0, 0, None
    with Path(path).open("r", encoding="ascii") as stream:
        while line := stream.readline():
            words = line.split()
            if not words:
                continue
            kind = words[0]
            if kind == "TIME":
                result["time"] = float(stream.readline())
            elif kind == "POINT_DATA":
                point_count, association = int(words[1]), "point"
            elif kind == "CELL_DATA":
                cell_count, association = int(words[1]), "cell"
            elif kind == "POINTS":
                n = int(words[1]); values = [stream.readline() for _ in range(n)]
                result["coordinates"] = np.fromstring(" ".join(values), sep=" ").reshape(n, 3)
            elif kind == "CELLS":
                n = int(words[1])
                result["connectivity"] = [np.fromstring(stream.readline(), sep=" ", dtype=np.int64)[1:] for _ in range(n)]
            elif kind in ("SCALARS", "VECTORS", "TENSORS"):
                n = point_count if association == "point" else cell_count
                width = {"SCALARS": 1, "VECTORS": 3, "TENSORS": 9}[kind]
                if kind == "SCALARS":
                    if not stream.readline().startswith("LOOKUP_TABLE"):
                        raise ValueError("Invalid VTK scalar lookup table")
                lines = [stream.readline() for _ in range(n * (3 if kind == "TENSORS" else 1))]
                values = np.fromstring(" ".join(lines), sep=" ")
                if len(values) != n * width:
                    raise ValueError("Truncated VTK reference array")
                result[(association, words[1])] = values.reshape(n, width).squeeze(-1) if width == 1 else values.reshape(n, width)
    return result


def compare_vtk(frame, path):
    """Compare all requested shell fields against official six-digit ASCII."""
    ref = _read_ascii_vtk(path)
    nn, ne = len(frame["node_id"]), len(frame["shell_element_id"])
    candidate = np.flatnonzero(np.asarray([len(x) in (3, 4) for x in ref["connectivity"]]))
    if len(candidate) != ne:
        raise AssertionError("Official VTK shell count differs")
    indices = candidate
    exact = {
        "node_id": np.array_equal(frame["node_id"], ref[("point", "NODE_ID")]),
        "shell_element_id": np.array_equal(frame["shell_element_id"], ref[("cell", "ELEMENT_ID")][indices]),
        "shell_part_id": np.array_equal(frame["shell_part_id"], ref[("cell", "PART_ID")][indices]),
        "shell_active": np.array_equal(frame["shell_active"], ref[("cell", "EROSION_STATUS")][indices]),
        "shell_connectivity": all(np.array_equal(a, b) for a, b in zip(frame["shell_connectivity"], (ref["connectivity"][i] for i in indices))),
    }
    checks = {}
    def check(name, raw, expected):
        raw, expected = np.atleast_1d(raw), np.atleast_1d(expected)
        delta = np.abs(raw.astype(np.float64) - expected)
        # anim_to_vtk uses C++ default six significant digit formatting.
        tolerance = 5.1e-6 * np.maximum(np.abs(expected), 1e-25)
        okay = bool(np.all(delta <= tolerance))
        checks[name] = {"passed": okay, "max_abs_error": float(delta.max(initial=0)),
                        "values": int(raw.size)}
        if not okay:
            raise AssertionError("ANIM / official VTK mismatch: " + name)
    if not all(exact.values()):
        raise AssertionError("ANIM / official VTK ID/connectivity mismatch: " + str(exact))
    check("time", frame["time"], ref["time"])
    check("coordinates", frame["coordinates"], ref["coordinates"])
    check("displacement", frame["displacement"], ref[("point", "Displacement")])
    for name, values in frame["shell_scalars"].items():
        check(name, values, ref[("cell", "2DELEM_" + name.replace(" ", "_"))][indices])
    for name, values in frame["shell_tensors"].items():
        expected = ref[("cell", "2DELEM_" + name.replace(" ", "_"))][indices].reshape(ne, 3, 3)
        if not (np.array_equal(expected[:, 0, 1], expected[:, 1, 0])
                and np.all(expected[:, 2, :] == 0) and np.all(expected[:, :, 2] == 0)):
            raise AssertionError("Unexpected VTK shell tensor symmetry/padding")
        check(name, values, expected[:, [0, 1, 0], [0, 1, 1]])
    return {"exact_checks": exact, "float_checks": checks, "nodes": nn, "shells": ne,
            "ascii_tolerance": "5.1e-6 relative to official six-digit values"}


def peak_working_set_mib():
    if __import__("sys").platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = wintypes.HANDLE
        info = ctypes.windll.psapi.GetProcessMemoryInfo
        info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        info.restype = wintypes.BOOL
        if info(process(), ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / 2**20
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("anim", type=Path)
    parser.add_argument("--compare-vtk", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    frame = read_anim(args.anim)
    read_seconds = time.perf_counter() - started
    read_peak = peak_working_set_mib()
    report = {"anim": args.anim.name, "read_seconds": read_seconds,
              "read_peak_working_set_mib": read_peak, "time": frame["time"],
              "metadata": frame["metadata"]}
    if args.compare_vtk:
        started = time.perf_counter()
        report["validation"] = compare_vtk(frame, args.compare_vtk)
        report["validation_seconds"] = time.perf_counter() - started
    report["total_peak_working_set_mib"] = peak_working_set_mib()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

