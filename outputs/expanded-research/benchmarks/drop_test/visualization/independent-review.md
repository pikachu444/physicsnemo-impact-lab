# Read-only VTU evidence review

Reviewer `/root/prior_context/drop_result_review`,2026-10-15. Verdict: **APPROVE; no correction requested**. No raw-file reload or repeated test was required for this review; the separate meshio parser evidence records actual reopening.

The exporter, two case sidecars, README, measured runtime, independent meshio5.3.5 reader validation and native2070x1152 preview agree. Fixed76065 NODE_ID points and130218 cells preserve source point/cell ordering, TET4/HEX8/WEDGE6 counts and PART_ID. All468wedge mappings preserve9physical edges, and the3.8895e-6 alternate-volume difference is explicitly bounded and disclosed. Points are reference+FEM_U atscale1. The prediction path requires exact NODE_ID order/shape/finiteness before adding AI/error arrays; both present VTUs are FEM-only. Meshio actually read bothVTUs and checked fields bitwise, node order, geometry, corner sets and PART_ID. The README correctly states that ParaView GUI was not launched and20us is interpolated from actual solver times. CPU1/BLAS1/noGPU,2.836s and74.52MiB agree with measured evidence. Preview titles, stressscale/maxima, corner/midside scope and no-AI state are readable.

The twoVTUs remain local-only. Public preservation is limited to code, audit/hash summaries, README andPNG; no rawVTU or mesh/cache array is approved for this visualization package.
