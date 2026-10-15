# Deforming plate: independent source and kinematics review

The reviewed NVIDIA source snapshot is commit `11685edb3ec1fb6f25665e96cf7a8c4d11a23f16`, under `examples/structural_mechanics/deforming_plate/`. This review imported no Torch and used no GPU. It does not certify trained-model accuracy.

## Label meaning and units

- `meta.json:2–4` identifies COMSOL, `dt=0`, and collision radius `0.03`; lines 42–49 define a scalar nodal stress field with 400 frames. The metadata contains no strain field or unit declaration.
- The original [MeshGraphNets paper, section 4](https://arxiv.org/html/2010.03409v4#S4) identifies DeformingPlate as a quasi-static hyperelastic plate driven by a kinematic actuator and identifies its scalar stress output as nodal von-Mises stress. Appendix A.1 says this domain has no physical time step. Therefore the source variable called velocity is an increment between loading steps, not an established velocity in m/s.
- Neither the local NVIDIA example nor this metadata establishes coordinate or stress units. Report displacement in dataset coordinate units and stress in dataset stress units. The numerical range alone does not justify labeling stress Pa or MPa.
- The paper supports the identity of the scalar stress, not a complete six-component stress tensor, plastic strain, failure, or dynamic impact validity.

## Official feature and inference contract

- `deforming_plate_dataset.py:104–107,280–313` constructs all six tetrahedral edges, coalesces them into a bidirectional graph, and attaches reference-coordinate edge vectors and lengths. A graph stays fixed within each trajectory (`:35–40`); this loader does not implement remeshing.
- `helpers.py:23–97` builds world-space edges inside radius 0.03, excludes self and existing mesh edges, uses deformed-coordinate edge vectors and lengths, and concatenates reference and current features for mesh edges. World-edge features are repeated to match eight columns. The official implementation raises if no additional world edges exist (`:67–68`); handling that case as an empty edge set is a documented compatibility difference.
- `deforming_plate_dataset.py:144–152,191–196` concatenates three next-step position increments and next-step scalar stress into a four-channel target. `conf/config.yaml:34–36` specifies input 3, output 4, edge 8. Targets are normalized using training statistics (`dataset.py:167–183`).
- `inference.py:141–176` feeds back its own previous position prediction after the first step, then prescribes type 1 actuator and type 3 clamp positions from the known next-step boundary motion. Type 0 is the free/deformable prediction set (`dataset.py:366–370`). Report free-node errors separately; prescribed actuator motion is an input condition.
- The official inference script denormalizes only the first three channels and stores positions, with stress visualization commented (`inference.py:149–178,188–189,231–237`). Actual stress inference and stress error reporting require an explicit extension despite a stress-trained decoder.
- `inference.py:193–195` names the norm of world position `disp_mag`; this is not generally the displacement from the reference mesh. Compare displacement as `world_pos - mesh_pos`.

## Derived strain contract and actual checks

`plate_data.py:33–45` uses edge vectors as columns, `F = Ds @ inv(Dm)`, and `E = 0.5*(F.T @ F - I)`. It returns tensor components in order `xx, yy, zz, xy, yz, xz`, with tensor shear (no engineering-shear factor of two). The result is dimensionless, elementwise Green–Lagrange strain derived from positions. It is neither logarithmic/true strain nor an independently exported COMSOL strain label. Do not describe this as training on original FEM strain outputs.

The separate CPU-only script `work/expanded-plate/review/review_kinematics.py` tested an arbitrary non-orthogonal reference tetrahedron under a general affine deformation, that deformation followed by a rigid rotation, consistent node permutation, and a degenerate reference tetrahedron. All passed; maximum component error was `6.32e-9` after float32 output. This complements the worker's translation, rigid rotation, and axial-stretch checks and detects transposed deformation-gradient implementations that those simpler checks could miss.

`kinematics-review.json` also records actual geometry checks for all 12 acquired trajectories. Every initial world coordinate equals its reference coordinate, every reference determinant is positive, the maximum reference condition number is 20.85, and no final ground-truth plate tetrahedron is inverted. These checks cover initial/final geometry, not every intermediate prediction. For strain evaluation exclude tetrahedra touching type 1 actuator nodes, retain the plate's clamped nodes, use volume weighting or state the unweighted convention, and report predicted inversion counts rather than hiding invalid deformation behind a finite strain tensor.

## Reproducibility boundaries

The bounded first-trajectory selection is not a random representative sample of all 1,000/100/100 public trajectories. Keep the original independent trajectory splits and never split adjacent frames across train/test. The public data are labels from COMSOL; a reproducible COMSOL input model and material parameters were not established by these snapshots. Preserve range offsets, CRC32C checks, response provenance and SHA256 hashes; a successful partial-data run does not reproduce the full reference training regime or its published accuracy.
