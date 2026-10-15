"""Download the official example locally and add explicit FEM field outputs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

URL = "https://2022.help.altair.com/2022.2/simulation/demos/hwsolvers/radioss/example/RD-E-1101_Tensile_test.zip"
ZIP_SHA256 = "b873a56df00aba34f44a80a0fa443e188dd0d0bf3844d274feb4b06494c72564"
STARTER_SHA256 = "36272441339572e8a2d63a7f89cf3a953277ed6ace9c7f6fad89215ae0130ee4"


def prepare(work, case, strength_scale=1.0, velocity_scale=1.0, end_ms=10.0, interval_ms=1.0):
    assert strength_scale > 0 and velocity_scale > 0 and 0 < interval_ms <= end_ms
    work.mkdir(parents=True, exist_ok=True)
    archive = work / "RD-E-1101_Tensile_test.zip"
    if not archive.exists():
        data = urllib.request.urlopen(URL, timeout=60).read()
        assert len(data) < 2_000_000
        archive.write_bytes(data)
    data = archive.read_bytes()
    assert hashlib.sha256(data).hexdigest() == ZIP_SHA256, "Official archive differs from the pinned source"
    original = work / "original"
    original.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as package:
        for name in ["tensile_LAW2_0000.rad", "tensile_LAW2_0001.rad"]:
            content = package.read("1_LAW2/" + name)
            (original / name).write_bytes(content)
    starter_bytes = (original / "tensile_LAW2_0000.rad").read_bytes()
    assert hashlib.sha256(starter_bytes).hexdigest() == STARTER_SHA256
    lines = starter_bytes.decode("ascii").splitlines()
    material = lines.index("/MAT/PLAS_JOHNS/2")
    assert lines[material + 4][:60].split() == ["0.3", "0.686", "0.129"]
    # Preserve E, density, UTS strain and the source's Iflag=1 hardening definition.
    lines[material + 4] = f"{0.3 * strength_scale:20.12g}{0.686 * strength_scale:20.12g}" + lines[material + 4][40:]
    prop = lines.index("/PROP/SHELL/1")
    assert lines[prop + 4][:10].strip() == "5"
    lines[prop + 4] = lines[prop + 4][:10] + f"{1:10d}" + lines[prop + 4][20:]
    function = lines.index("/FUNCT/1")
    for index in [function + 2, function + 3]:
        assert float(lines[index][20:40]) == 1.0
        lines[index] = lines[index][:20] + f"{velocity_scale:20.12g}" + lines[index][40:]
    run = work / "runs" / case
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing case: {run}")
    run.mkdir(parents=True, exist_ok=True)
    (run / "tensile_LAW2_0000.rad").write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    engine = f"""/VERS/2017
/RUN/tensile_LAW2/1/
{end_ms:20.12g}
/ANIM/DT
{0:20.12g}{interval_ms:20.12g}
/ANIM/VECT/DISP
/ANIM/VECT/VEL
/ANIM/SHELL/TENS/STRESS/3
/ANIM/SHELL/TENS/STRAIN/3
/ANIM/SHELL/EPSP/3
/ANIM/SHELL/VONM
/ANIM/SHELL/SIGX
/ANIM/SHELL/SIGY
/ANIM/SHELL/SIGXY
/TFILE/4
{0.01:20.12g}
/PRINT/-500
/MON/ON
/DT/NODA/CST/0
{0.9:20.12g}{0:20.12g}
"""
    (run / "tensile_LAW2_0001.rad").write_text(engine, encoding="ascii", newline="\n")
    parameters = {
        "case": case, "strength_scale": strength_scale, "velocity_scale": velocity_scale,
        "end_ms": end_ms, "animation_interval_ms": interval_ms,
        "source_url": URL, "source_zip_sha256": ZIP_SHA256,
        "source_starter_sha256": STARTER_SHA256,
        "source_units": {"mass": "kg", "length": "mm", "time": "ms", "stress": "GPa", "force": "kN"},
        "material": {"source_name": "Steel_DP600", "E_GPa": 210.0, "nu": 0.3, "rho_kg_per_mm3": 7.8e-6,
                     "yield_GPa": 0.3 * strength_scale, "UTS_GPa": 0.686 * strength_scale, "strain_at_UTS": 0.129},
        "strength_scale_definition": "Scale Iflag=1 engineering yield and engineering UTS together; E, density, nu, engineering strain at UTS remain fixed. Solver recalculates LAW2 A/B/n; this is not an assertion that A and B scale uniformly.",
        "parameter_bounds_scope": "Local algorithm-validation sensitivity assumptions, not measured DP600 material scatter or product design tolerances.",
        "mesh": {"source_nodes": 920, "quad_shells": 804, "tri_shells": 4, "thickness_mm": 2.5,
                 "formulation": "QEPH24", "thickness_integration_points": 5},
        "output_location": "shell thickness integration point3; scalar cross-check output definitions remain distinct",
        "source_license": "No redistribution grant found in official ZIP; documentation is copyright Altair, All Rights Reserved. Original decks remain local.",
        "scope": "Source mesh/material/BC retained at unit scales; first10ms is an exploratory prefix of the original40ms.",
    }
    parameters["prepared_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob("*.rad")}
    (run / "parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    print(json.dumps({"run": str(run), "parameters": parameters}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--case", default="pilot_base")
    parser.add_argument("--strength-scale", type=float, default=1.0)
    parser.add_argument("--velocity-scale", type=float, default=1.0)
    parser.add_argument("--end-ms", type=float, default=10.0)
    parser.add_argument("--interval-ms", type=float, default=1.0)
    args = parser.parse_args()
    prepare(args.work, args.case, args.strength_scale, args.velocity_scale, args.end_ms, args.interval_ms)
