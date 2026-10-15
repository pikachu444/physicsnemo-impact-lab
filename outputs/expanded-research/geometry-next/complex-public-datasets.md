# Complex structural geometry datasets: bounded feasibility check

Checked 2026-10-16. This is a primary-source discovery result, **not data acquisition, model training, FEM execution, or an accuracy result**. Scope: variable *initial* 3D structural geometry with displacement/stress/strain labels, preferably complex parts or assemblies, and a staged footprint below 60 GB. No architecture was selected.

## Decision

**No immediately accessible candidate among these four meets the complex, controlled structural shape-response requirement.** SHIFT-Crash is the best conditional candidate by physical scale and genuine initial-shape variation, but its protected files require the human user to log in, review the contact-sharing/access conditions, request access, and receive approval. No consent was submitted and no credentials were sought. SFEM is accessible and genuinely varies CAD, but its coarse stochastic-median labels do not satisfy the requested complex assembly benchmark.

| Candidate | Actual initial geometry / response | Public artifact evidence | Decision |
|---|---|---|---|
| [SHIFT-Crash](https://huggingface.co/datasets/luminary-shift/SHIFT-Crash) | Morphed whole-vehicle Yaris family; four geometric offsets and two rail properties; card reports 376,593 surface nodes, 13 frames, displacement/stress/plastic strain. | Public API lists **5,183 VTP + 5,183 STL + 5,183 params JSON** files. Case 00001 is **293,277,531 bytes** total. API says `gated: manual`; protected params HEAD returned **401**. | Best fit, **access blocked**. Fixed baseline topology is not independently remeshed geometry. |
| [SFEM](https://huggingface.co/datasets/cmudrc/SFEM) | BrepGen CAD shapes; tetrahedral linear elasticity. Raw H5 includes nodal `u`, `VonMises`, deviatoric `StressTensor`, connectivity and support masks. | Ungated MIT. Actual raw archives total **3,777,988,660 bytes**. A **262,144-byte HTTP range** exposed three complete H5 records in memory. | Accessible, **rejected for requested scale/controlled-response scope**: sampled meshes have 382/544/712 nodes; producer main caps meshes at 2,000 nodes; targets are medians over stochastic loads. |
| [LatticeGraphNet](https://arxiv.org/html/2402.01045v1) | 116 nonlinear lattice simulations; 108 training / 8 test; changing cell types, strut diameters and cell sizes. Displacement increments and reduced stress invariants. | Paper verified; no downloadable dataset archive, file manifest, license or size verified in this bounded check. | Relevant physics and variable initial lattice geometry; **public acquisition unverified**. |
| [PhysicsNeMo sintering](https://github.com/NVIDIA/physicsnemo/tree/main/examples/additive_manufacturing/sintering_physics) | Paper uses seven training geometries and distinct held-out parts including USB, busbar, screw and pushing grip. Position history implies distortion/displacement. | Current TFRecord reader exposes positions, particle type, edges and temperature context; stress/strain absent. Current example has local paths without a dataset download link. | **Public acquisition unverified**, and manufacturing distortion has a narrower physical task. |

## SHIFT-Crash: next action and budget

- Metadata sources: [repository API](https://huggingface.co/api/datasets/luminary-shift/SHIFT-Crash), [one-case public file tree](https://huggingface.co/api/datasets/luminary-shift/SHIFT-Crash/tree/main/yaris_x00001?recursive=false&expand=false). Revision: `6faa776f7e363ce39d43eb8dc8b0d9c7d0c83049`; storage API: 1,515,784,528,128 bytes; license: CC BY-NC 4.0.
- `yaris_x00001/merged_surfaces.vtp`: **257,016,764 B**; STL: **36,260,484 B**; params: **283 B**. Protected [params URL](https://huggingface.co/datasets/luminary-shift/SHIFT-Crash/resolve/main/yaris_x00001/params.json) returned HTTP 401. Only public metadata was retrieved.
- A **32-case** subset at this observed case size is **9,384,880,992 B** (8.74 GiB), before derived caches. This is a budget estimate, not a verified selection. Full download exceeds 60 GB.
- After user access approval, inspect the small `params.json` files first to select geometric changes with **identical `rail_thickness` and `rail_yield_strength`**. Matched groups were **not verified** because params are protected. Split by geometric parameter tuple/case, keeping all frames of a case together. No official split was established here.
- The released representation is a morphed surface mesh. Same node count or topology does not mean identical initial shape. It supports shape-conditioned response tests; it does **not** supply a paired remeshing experiment. Save topology/coordinate hashes and part IDs before claims about correspondences.

## SFEM: verified file/header details and why it is insufficient

Sources: [actual repository files](https://huggingface.co/api/datasets/cmudrc/SFEM/tree/main?recursive=false&expand=false), [repository metadata](https://huggingface.co/api/datasets/cmudrc/SFEM), [producer source](https://github.com/cmudrc/SFEM), [loader](https://github.com/cmudrc/SFEM/blob/main/dataloader.py), [simulation](https://github.com/cmudrc/SFEM/blob/main/simulation.py), [mesher](https://github.com/cmudrc/SFEM/blob/main/mesh_generation.py).

Revision `ab8239a2d9c2c2125402ea68c557c2c8b022ac86`, `gated: false`, 39,894 STEP files in the sibling inventory. README counts conflict: about 16,000 CADs vs 39,894 STEP files; do not infer distinct simulated geometry count from either.

| File | Bytes | Direct URL |
|---|---:|---|
| Raw training H5 archive | 3,208,130,929 | [raw_train_data.tar.gz](https://huggingface.co/datasets/cmudrc/SFEM/resolve/main/Pytorch_Geometric_Files/raw_train_data.tar.gz) |
| Raw validation H5 archive | 569,857,731 | [raw_val_data.tar.gz](https://huggingface.co/datasets/cmudrc/SFEM/resolve/main/Pytorch_Geometric_Files/raw_val_data.tar.gz) |
| Processed training | 10,074,233,409 | [train_data.pt](https://huggingface.co/datasets/cmudrc/SFEM/resolve/main/Pytorch_Geometric_Files/processed/train_data.pt) |
| Processed validation | 1,788,466,635 | [val_data.pt](https://huggingface.co/datasets/cmudrc/SFEM/resolve/main/Pytorch_Geometric_Files/processed/val_data.pt) |

Anonymous HEAD on raw validation returned HTTP 200, matching content length and byte-range support. A subsequent HTTP 206 range of bytes 0–262143 was decompressed only in memory, yielding these complete H5 members:

| Member under `raw/val/` | H5 bytes | Nodes | Tetrahedra |
|---|---:|---:|---:|
| `02530_001_final_medium_Load.h5` | 105,215 | 382 | 1,037 |
| `12430_001_final_small_Load.h5` | 153,130 | 544 | 1,604 |
| `00034_003_final_large_Load.h5` | 190,138 | 712 | 1,874 |

All three have `Vertices (N,3) float64`, `u (N,3) float64`, `VonMises (N,1) float64`, `StressTensor (N,3,3) float64`, `Fixed_Facet (N,2) int64`, `Cell_Faces (F,3) int32`, `Facets (T,4) int32`, and `Load_Class (1,)` byte string. These are actual file headers, beyond a landing-page claim; the range was not retained as a local dataset.

Producer-source limits:

- `simulation.py` stores elementwise/nodewise medians over 50 stochastic realizations, not a single deterministic response. Selected load zones/realization seeds are not saved. A load class alone cannot recover the realized forcing. Within the load-zone loop the per-iteration array is overwritten, so do not describe the result as a simultaneous sum over all zones.
- The tensor written as `StressTensor` is `s = sigma - tr(sigma)/3 I`, the **deviatoric** tensor. Calling it full Cauchy stress would be wrong. No strain array is present in these headers.
- Main calls the solver with `max_nodes=2000`. The mesher can regenerate a new tetrahedral mesh from each STEP, but no paired mesh-resolution/remesh response labels are released by this evidence.
- Geometry split code groups `'_'.join(filename.split('_')[:3])`, e.g. `02530_001_final`, and assigns all load classes together to 85/15 train/validation. It has no frozen seed/test split. Verify the actual archive group disjointness before any experiment.
- Processed `.pt` loader uses only scalar Von Mises labels; displacement/tensor loading is commented out. Its per-sample target-derived normalization must not be reused for a deployable unknown-target prediction task. Raw H5 is the suitable starting representation if a coarse varied-CAD benchmark is separately desired.

## Unavailable-candidate evidence

LatticeGraphNet: [paper geometry/data sections](https://arxiv.org/html/2402.01045v1). It treats nonlinear Neo-Hookean IPC lattice compression and two graph scales; stress invariants on the reduced graph are not a verified downloadable full stress/strain field archive. A missing archive in this bounded check is an availability uncertainty, not proof that the authors will never release one.

Sintering: [paper](https://arxiv.org/html/2404.11753v1), [current README](https://github.com/NVIDIA/physicsnemo/blob/main/examples/additive_manufacturing/sintering_physics/README.md), [actual reader schema](https://github.com/NVIDIA/physicsnemo/blob/main/examples/additive_manufacturing/sintering_physics/reading_utils.py), [NGC supplemental manifest](https://api.ngc.nvidia.com/v2/resources/nvidia/physicsnemo/physicsnemo_examples_supplemental_materials/versions/0.0.1/files). The checked NGC manifest has nine files totaling 267,208,860 bytes, comprising README/weather material rather than sintering data. It is not evidence of acquired sintering labels.

Only the two new research files were authored by this task. No existing report or Git state was changed.
