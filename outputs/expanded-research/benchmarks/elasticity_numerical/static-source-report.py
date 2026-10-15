"""Inspect source/package metadata without importing numerical frameworks."""
import ast
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import sys
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3]
source=ROOT/'work/expanded-elasticity/source/examples/tcad/fp_ddm'
manifest=json.loads((OUT/'source-manifest.json').read_text())
assert manifest['status']=='pinned_official_source_verified'
imports={}
for item in manifest['files']:
    file=ROOT/'work/expanded-elasticity/source'/item['path']
    assert hashlib.sha256(file.read_bytes()).hexdigest()==item['sha256']
    if file.suffix=='.py':
        tree=ast.parse(file.read_text())
        imports[str(file.relative_to(source)).replace('\\','/')]=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
versions={}
for name in ('torch','nvidia-physicsnemo','numpy','scipy','tqdm','matplotlib','imageio','opensimplex','psutil'):
    try:versions[name]=metadata.version(name)
    except metadata.PackageNotFoundError:versions[name]=None
current=ROOT/'work/expanded-elasticity/current-core-init.py'
alpha=re.search(r'__version__\s*=\s*[\"\']([^\"\']+)',current.read_text())[1]
forbidden=[k for k in sys.modules if k.split('.')[0] in ('torch','tensorflow','jax','numpy','physicsnemo')]
assert not forbidden
report={'status':'static_only_no_framework_execution','official_source_pin':manifest['commit'],'current_source_version':alpha,
 'installed_metadata_versions':versions,'python_version':sys.version.split()[0],
 'version_scope':'Current 2.3.0a0 example source is vendored at an exact SHA. Installed PhysicsNeMo 2.2.2 is a separate release. The standalone official Elasticity2DSolver imports only Torch; no wheel package is replaced.',
 'imports_by_official_file':imports,'direct_solver_imports':imports['fpddm/elasticity.py'],
 'full_ddm_transitive_imports':'elasticity_example -> schwarz -> model/solvers -> PhysicsNeMo FNO, Module, ModelMetaData, load_checkpoint. These imports do not instantiate or train an elasticity FNO.',
 'optional_requirements_scope':'opensimplex/imageio are declared by the complete fp_ddm example; domain.make_layout_fields delays thermal-data import. Bare elasticity.py requires Torch only; validation driver also uses stdlib and psutil.',
 'source_version_evidence':{'url':'https://github.com/NVIDIA/physicsnemo/blob/'+manifest['commit']+'/physicsnemo/__init__.py',
     'sha256':hashlib.sha256(current.read_bytes()).hexdigest()},
 'framework_modules_imported':forbidden,'solver_execution_performed':False,'package_installation_performed':False,
 'minimal_ddm_source_constraints':{'API':'run_elasticity(size=4,overlap=2)','patch_size':3,'global_grid_nodes':16,'Q1_elements':9,'status':'source-admissible parameter estimate, not executed'},
 'planned_solver_validation':{'grid':[9,9],'domain_lengths_y_x':[1.,2.],'young_modulus':120.,'poisson_ratio':.25,
   'uniform_strain_yx':[[.001,.0003],[.0003,.002]],'analytic_plane_stress_yx':[[.192,.0288],[.0288,.288]],
   'body_force_uniform_patch':[0.,0.],'boundary':'both displacement components prescribed on all four boundaries',
   'second_problem':'independent constant body force [f_y,f_x]=[-.01,0], clamp x=0; equilibrium/energy checks, no closed-form displacement accuracy claim',
   'units':'Official example declares no physical unit system. Use consistent model L and stress S: u[L], strain[-], E/sigma[S], body force[S/L], energy per unit thickness[S L^2].',
   'array_memory_estimate':'9x9 float64 solver fields/scratch <<1MiB; Torch import/allocator dominates process memory.',
   'wall_time_estimate':'Likely within 5 minutes for import plus two 9x9 CPU solves; not measured yet.',
   'process_memory_estimate':'Framework baseline may be hundreds of MiB to about 1.5GiB depending on loaded runtime; must measure after root grants a slot.'}}
(OUT/'static-source-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('status','current_source_version','installed_metadata_versions','framework_modules_imported','solver_execution_performed')}))
