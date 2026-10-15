"""Stdlib-only manifest of the already allocated twelve-run comparison."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    protocol = root/"outputs/expanded-research/full-support-static-training-protocol.json"
    sources = [root/"work/expanded-ml"/name for name in (
        "train_budgeted.py", "evaluate_budgeted.py", "budgeted_runtime.py", "budget_protocol.py",
        "benchmark_data.py", "model_adapters.py", "run_budgeted_suite.py")]
    adapter = root/"work/expanded-batch/batch_adapters.py"
    sources.append(adapter)
    plan = {"working_directory": str(root), "protocol": str(protocol), "protocol_sha256": sha(protocol),
            "source_sha256": {str(path): sha(path) for path in sources}, "concurrent_pids": [34268], "runs": []}
    architectures = ("transolver", "flare", "geotransolver", "geoflare", "meshgraphnet", "figconvunet")
    for domain in ("phone", "bumper"):
        manifest = root/("work/expanded-ml/phone-full-cache-v2/manifest.json" if domain == "phone" else "work/expanded-ml/multifield7-cache/manifest.json")
        normalization = root/"work/expanded-ml"/("phone-full-normalization.json" if domain == "phone" else "bumper7-normalization.json")
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        for architecture in architectures:
            batch = (1 if architecture == "meshgraphnet" else 2 if architecture == "figconvunet" else 5) if domain == "phone" else (2 if architecture == "meshgraphnet" else 5)
            actual = {batch, len(meta["split"]["train"]) % batch}-{0, 1}
            approvals = []
            for size in sorted(actual):
                slot = "allocated-slot-02" if domain == "phone" and architecture == "figconvunet" else "allocated-slot-01"
                path = root/"outputs/expanded-research/benchmarks/batching"/slot/domain/architecture/f"B{size}/result.json"
                row = json.loads(path.read_text(encoding="utf-8"))
                assert row["status"] == "passed"
                assert row["timing"]["forward_backward"]["native_faster_in_every_paired_repeat"]
                approvals.append(str(path))
            out = root/("outputs/expanded-research/benchmarks/drop_test/models-full-c6" if domain == "phone" else "outputs/expanded-research/benchmarks/bumper/models-c7")/architecture
            assert not out.exists() or not any(out.iterdir()), out
            row = {"domain": domain, "architecture": architecture, "batch_size": batch,
                   "cache": str(manifest), "manifest_sha256": sha(manifest), "normalization": str(normalization),
                   "out": str(out), "batch_adapter": str(adapter) if actual else None, "batch_approvals": approvals}
            if domain == "phone":
                row["data_clarification"] = str(root/"outputs/expanded-research/benchmarks/drop_test/common-cache-provenance-clarification.json")
            plan["runs"].append(row)
    path = root/"work/expanded-ml/budgeted-suite-plan.json"
    assert not path.exists(), "Preserve the original prospective plan"
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps({"plan": str(path), "sha256": sha(path), "runs": len(plan["runs"]), "source_sha256": plan["source_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
