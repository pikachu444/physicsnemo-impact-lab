# Research execution

This repository investigates PhysicsNeMo surrogates for appliance drop and impact FEM.
Use only public or explicitly authorized research data. Do not incur additional charges.
Keep installation, smoke tests, trained models and held-out accuracy validation distinct.
Record exact upstream revisions, dataset file hashes, commands, failures, elapsed time,
peak memory and split membership. Split by independent simulation, never by nearby time frames.
Do not silently replace a requested GPU experiment with a CPU completion claim.
Do not commit downloaded datasets or third-party binaries. Preserve their provenance and licenses.

For Cloud environment setup, first run:

    python scripts/probe_environment.py --label codex-cloud-setup --network --output results/cloud/setup-probe.json

Report measured CPU, RAM, disk, NVIDIA driver, CUDA availability and network results.
The probe does not need PhysicsNeMo installed. Missing torch is a recorded result, not a reason to abort the probe.
Do not assume the VM has a GPU. No paid GPU provisioning, subscription changes or purchases.

Prepare Python dependencies in a virtual environment. Select dependency versions only after
checking the pinned upstream example; do not mix old PhysicsNeMo Sym examples with current APIs.
Preserve setup logs in results/cloud. Do not run long training during setup.
