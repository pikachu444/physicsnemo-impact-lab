"""Fetch bounded complete TFRecord trajectories with HTTP Range, without TensorFlow."""
import argparse
import hashlib
import json
import pathlib
import struct
import time
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT.parents[1] / "outputs/expanded-research/benchmarks/deforming_plate"
TABLE = []
for value in range(256):
    for _ in range(8):
        value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
    TABLE.append(value)


def crc32c(data):
    crc = 0xFFFFFFFF
    for value in data:
        crc = TABLE[(crc ^ value) & 255] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def masked_crc(data):
    crc = crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def varint(data, offset):
    value = 0
    shift = 0
    while True:
        b = data[offset]
        offset += 1
        value |= (b & 127) << shift
        if not b & 128:
            return value, offset
        shift += 7
        assert shift <= 63


def fields(data):
    offset = 0
    while offset < len(data):
        key, offset = varint(data, offset)
        kind = key & 7
        if kind == 2:
            length, offset = varint(data, offset)
            value = data[offset:offset + length]
            assert len(value) == length
            offset += length
        elif kind == 0:
            value, offset = varint(data, offset)
        else:
            raise ValueError(f"Unexpected protobuf wire type {kind}")
        yield key >> 3, value


def decode_example(payload, meta):
    top = list(fields(payload))
    assert len(top) == 1 and top[0][0] == 1
    result = {}
    for number, entry in fields(top[0][1]):
        assert number == 1
        item = dict(fields(entry))
        name = item[1].decode("utf8")
        feature = list(fields(item[2]))
        assert len(feature) == 1 and feature[0][0] == 1
        values = list(fields(feature[0][1]))
        assert len(values) == 1 and values[0][0] == 1
        spec = meta["features"][name]
        result[name] = np.frombuffer(values[0][1], dtype=spec["dtype"]).reshape(spec["shape"]).copy()
    assert set(result) == set(meta["field_names"])
    n = result["mesh_pos"].shape[1]
    assert result["world_pos"].shape == (400, n, 3)
    assert result["stress"].shape == (400, n, 1)
    assert np.isfinite(result["world_pos"]).all() and np.isfinite(result["stress"]).all()
    assert result["cells"].min() >= 0 and result["cells"].max() < n
    assert set(np.unique(result["node_type"]).tolist()) <= {0, 1, 3}
    return result


def fetch_range(url, start, count):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + count - 1}"})
    with urllib.request.urlopen(req, timeout=120) as response:
        assert response.status == 206, (response.status, dict(response.headers))
        assert response.headers["Content-Range"].startswith(f"bytes {start}-{start + count - 1}/")
        payload = response.read(count + 1)
        assert len(payload) == count
        return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=8)
    parser.add_argument("--valid", type=int, default=2)
    parser.add_argument("--test", type=int, default=2)
    args = parser.parse_args()
    started = time.perf_counter()
    meta = json.loads((ROOT / "meta.json").read_text())
    manifest_path = OUT / "data-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "selection": "First complete trajectories of the original independent train/valid/test splits; no time-frame split.",
        "raw_strain_available": False,
        "meta_sha256": hashlib.sha256((ROOT / "meta.json").read_bytes()).hexdigest(),
        "records": [],
    }
    for split in ["train", "valid", "test"]:
        count = getattr(args, split)
        offset = 0
        for i in range(count):
            existing = [r for r in manifest["records"] if r["split"] == split and r["index"] == i]
            dest = ROOT / "data" / f"{split}_{i:04d}.npz"
            if existing and dest.exists():
                assert hashlib.sha256(dest.read_bytes()).hexdigest() == existing[0]["npz_sha256"], f"Cached trajectory was modified: {dest}"
                offset += existing[0]["record_length"]
                continue
            url = f"https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/{split}.tfrecord"
            header = fetch_range(url, offset, 12)
            length, checksum = struct.unpack("<QI", header)
            assert masked_crc(header[:8]) == checksum
            assert length < 100 * 1024 * 1024, length
            body = fetch_range(url, offset + 12, length + 4)
            assert masked_crc(body[:-4]) == struct.unpack("<I", body[-4:])[0]
            decoded = decode_example(body[:-4], meta)
            dest.parent.mkdir(exist_ok=True)
            np.savez_compressed(dest, **decoded)
            rec = {"split": split, "index": i, "url": url, "offset": offset,
                   "record_length": length + 16, "payload_sha256": hashlib.sha256(body[:-4]).hexdigest(),
                   "npz_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                   "crc32c_verified": True, "shapes": {k: list(v.shape) for k, v in decoded.items()},
                   "stress_min": float(decoded["stress"].min()), "stress_max": float(decoded["stress"].max())}
            manifest["records"].append(rec)
            manifest["elapsed_seconds"] = time.perf_counter() - started
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf8")
            print(json.dumps(rec), flush=True)
            offset += length + 16
    assert crc32c(b"123456789") == 0xE3069283
    print(f"Complete: {len(manifest['records'])} complete independent trajectories", flush=True)


if __name__ == "__main__":
    main()
