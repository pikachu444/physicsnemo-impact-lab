# Official-size plate independent streaming numerical audit

PASS_SCOPED_SOURCE_AND_SAVED_FIELD_NUMERIC_AUDIT

Completed cases: 6/6; checks 297; mismatches 0.
Wall 7.627s; OS peak working set 60.30MiB; minimum host available 926.78MiB.

Source and saved prediction NPY members were accessed through guarded read-only mmap, one case and frame at a time. No neural inference or producer evaluator import.

|Case|u relative L2|source scalar stress relative L2|derived GL relative L2|predicted/reference inverted frames|
|---|---:|---:|---:|---:|
|test_0002|8.96173505|0.717597849|9374.64931|385/0|
|test_0003|2.16639171|0.682944131|57.437999|73/0|
|test_0004|2.39484793|0.66334808|9.76844469|0/0|
|test_0005|2.01097207|0.647729102|32.3338752|24/0|
|test_0006|54.6968388|0.887477714|31.4261555|119/0|
|test_0007|1.85774633|0.676662619|20.6621403|0/0|

Relative L2 values are fractions. This is numerical correspondence, not model accuracy acceptance.

Limits:
- Original source means downloaded DeepMind trajectory NPZ, not a newly solved COMSOL model or reconversion of TFRecord.
- Array/node/tetrahedron indices are not solver IDs. Boundary nodes follow prescribed next source positions; only free nodes are predicted.
- Stress is source nodal scalar von Mises in unconfirmed dataset units; no MPa conversion.
- GL strain is independently derived from source/predicted coordinates on21 sampled frames, not native COMSOL strain.
- Inversion is tetra detF<=0 relative to mesh_pos; predicted and source counts are separate. It is not a rupture/failure law.
- No new neural inference, framework loading, source editing, plot or visual review. Numeric correspondence does not qualify model accuracy.
- Streaming accumulation and independent cofactor matrix algebra may differ by floating-point rounding; declared metric and tensor tolerances are recorded.
