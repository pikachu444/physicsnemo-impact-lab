# 공개 전 읽기 전용 검토

검토일: 2026-10-15 KST. 대상은 `physicsnemo-impact-lab`의 Git index 전체이며, 새 staged 파일과 기존 tracked 파일을 함께 확인했다. 대상 저장소 파일·환경·데이터는 변경하지 않았다. 이 보고서만 저장소 밖 `outputs/publication-review.md`에 작성했다.

**판정: 검토 범위에서 공개 차단 사항 없음.**

## 검토한 스냅샷

| 항목 | 결과 |
|---|---:|
| staged 변경 파일 | 171 |
| 누적 index 파일 | 209 |
| index blob 크기 합 | 8,936,717 B |
| UTF-8 텍스트 내용 검사 | 188파일 |
| outputs/results 실행 증거 바이트 검사 | 198파일 |
| 금지 파일·자격증명 후보 | 0 |
| 깨진 Markdown 상대 링크 | 0 |
| 절대 로컬 경로 Markdown 링크 | 0 |

검토한 index의 `path|Git blob id|mode|stage` 목록 SHA256은 `3f71421a35a86b7b89713ef27caa13c747db62f1a1fccd92d60ae3bbf41f0f73`이다. 이는 commit hash가 아니라 검토 스냅샷 식별자다. 이 보고서를 추가하면 목록은 달라진다.

## 파일·자격증명·자료 범위

- index 전체에서 NPZ·NPY·RAD·VTP·VTU·TFRecord·d3plot·H3D, solver EXE/DLL, ZIP·WHL, venv·캐시·third_party 경로의 포함이 없었다.
- MiniQA에서 직접 변환한 ASCII VTK 11개는 실행 결과로 포함되어 있다. 공개 학습 원본 VTP/RAD와 구분했다. 선택한 NN checkpoint와 공식 CPU 기능 검사 checkpoint는 보존되어 있다.
- 자격증명 파일명과 GitHub/OpenAI/Hugging Face/AWS/Google 키, private key, Bearer/JWT, 자격증명 포함 URL, 문자열로 직접 지정한 secret/password 계열 패턴을 검사했다. 후보가 없었고 값은 출력하지 않았다.
- `results/cloud/cloud-environment.jpg`를 직접 열어 확인했다. 저장소와 네트워크 허용 설정이 보이며 인증 토큰·개인 자격증명은 표시되지 않았다.

이는 현재 index에 대한 파일·텍스트·해당 화면 검토다. Git의 모든 과거 commit 이력이나 바이너리 checkpoint 내부를 포괄적으로 보안 분석했다는 뜻은 아니다.

## 실행 장소·자원·비용 설명

`README.md`, `docs/experiment-results.md`, `docs/resource-options.md`를 확인했다. Cloud의 장치 노출·cgroup 검사와 로컬 GPU의 실제 학습을 구분한다. Cloud 기본 사양에 대한 공식 문서와 해당 컨테이너의 실측도 구분한다. 해당 Cloud 인스턴스의 학습 성공을 주장하지 않는다.

Torch allocator 최대 할당, 프로세스 RSS의 표본 최대, 시스템 여유 RAM이 각각 별도로 적혀 있다. Torch 할당량을 전체 장치 VRAM 사용량으로 표현하지 않는다. GPU 연산 기능 검사, 실제 FEM 학습, 독립 수치 검토의 범위도 구분되어 있다.

유료 GPU 단가는 2026-10-15 공식 가격표 확인 결과이며 4 h/20 h는 GPU 점유 비용 시나리오다. 저장·전송·세금 등 제외 항목과 최종 견적 재확인을 적었다. 실제 선택 실험은 보유 GPU로 완료했고 추가 유료 자원·결제를 사용하지 않았다는 설명과 일치한다. [Runpod 공식 가격](https://www.runpod.io/pricing), [Lambda 공식 가격](https://lambda.ai/pricing), [Cloud 공식 문서](https://learn.chatgpt.com/docs/environments/cloud-environments)

## 실행 소스·checkpoint SHA256

아래 실행 소스는 각 실험 폴더의 `executed_source.py` Git blob에서 직접 계산했다. 세 값 모두 각 `metrics.json`의 `execution_source_sha256`와 일치한다.

| 실험 | 실행 소스 SHA256 |
|---|---|
| gpu1024 | `65d7156a41b8367c17d5d134d1ea1bce3055f43c4379200185405ae1e1df2241` |
| fulltraining | `179823d176347a1b350def6426845a3adde4afc75fde3c8850176b7750b38a10` |
| capacity1024 | `74a80cece908724b76103b7d91a435e8bc83180ff4eb9915cccbefbfcf1b8414` |

각 실험 폴더의 `transolver_cuda_checkpoint.pt`도 Git blob에서 해시를 계산했다. 세 값 모두 해당 roundtrip/inference 재사용 기록과 일치하고, 공개용 복사 전 원본 checkpoint 파일의 해시와도 일치한다. 원본 데이터를 새로 읽거나 checkpoint를 로드하여 추론한 검사는 아니다.

| 실험 | checkpoint SHA256 | 대조한 공개 기록 |
|---|---|---|
| gpu1024 | `217565c8a3b5c8547d5a4f0980ee154b102cb45ddf13020fed45d0f518bc021d` | `outputs/experiment/full_mesh/checkpoint_roundtrip.json` |
| fulltraining | `dc1f895100dd521afded61a9c43ced99a866e312cd2ec7629bee840ac29d124b` | `outputs/experiment/fulltraining/inference_reuse/inference_report.json` |
| capacity1024 | `380f2097a2f98d76fb5b7420e0a671adc5909df5637ff9ee77f665053bc359d4` | `outputs/experiment/capacity1024/inference_reuse/inference_report.json` |

## 실행 증거 개행·바이트 보존

`.gitattributes`에 `outputs/** -text`, `results/** -text`가 있다. 해당 두 경로의 198개 index blob을 작업 파일과 SHA256으로 대조해 모두 바이트가 일치함을 확인했다.

처음 남아 있던 기존 21개 MiniQA/로컬 환경 증거의 LF/CRLF 차이는 재스테이징 후 해소됐다. 새 ML의 세 실행 소스와 수치 파일도 바이트를 보존한다. 숫자가 같다는 검사와 정확한 파일 바이트가 같다는 검사를 구분했다.

## 재현 README와 상대 경로

모든 Markdown 상대 링크를 index의 실제 경로와 대조해 깨진 링크가 없음을 확인했다. 별도 읽기 전용 검토자가 다음 README 5개의 명령과 호출 스크립트, 다운로드 목적지, 생성 폴더를 대조했다.

- `README.md`
- `outputs/experiment/README.md`
- `outputs/gpu-experiment/README.md`
- `outputs/fem-experiment/README.md`
- `outputs/bumper-fem-rerun/README.md`

초기 발견한 범퍼 분석 Python venv 경로는 수정됐다. 현재 index에서는 ML README가 만드는 `outputs/experiment/local-run/venv/Scripts/python.exe`를 `$analysisPython`으로 선택하고, manifest·solver 다운로드가 분석 실행보다 앞에 있다. 원본 데이터·solver를 외부에서 별도로 확보하는 전제를 적용했다. 학습 NPZ를 공개 트리에 포함한다고 오해하지 않도록 한 설명도 확인했다.

이 검토는 경로와 절차의 정적 대조다. **새 clone의 완전히 비어 있는 환경에서 설치부터 모든 명령을 다시 실행한 clean-install 검증은 수행하지 않았다.** 추가 설치·데이터 다운로드·학습·FEM 실행·추론 없이 기존 실행 기록과 공개 스냅샷만 검토했다. 실행 성능·물리 정확도의 새 검증이나 전체 제품 적용 승인은 이 공개 검사 범위가 아니다.

## 검토한 index 파일 전체 목록

다음 209개 경로가 위 검토 스냅샷의 전체 파일 목록이다. 이 보고서 자체는 목록에 포함되지 않는다.

```text
.gitattributes
.gitignore
AGENTS.md
README.md
docs/experiment-results.md
docs/experiment-review.md
docs/fem-integration-review.md
docs/physicsnemo-source-review.md
docs/research-assessment.md
docs/resource-options.md
outputs/bumper-fem-rerun/.gitignore
outputs/bumper-fem-rerun/README.md
outputs/bumper-fem-rerun/attempt-1-metrics.json
outputs/bumper-fem-rerun/attempt-1-note.json
outputs/bumper-fem-rerun/compare-bumper.py
outputs/bumper-fem-rerun/compare-nn-bumper.py
outputs/bumper-fem-rerun/conversion-metrics.json
outputs/bumper-fem-rerun/download-bumper.ps1
outputs/bumper-fem-rerun/execution-metrics.json
outputs/bumper-fem-rerun/fem-rerun-summary.json
outputs/bumper-fem-rerun/fem-to-fem-run224-comparison.json
outputs/bumper-fem-rerun/fem-to-fem-run224-comparison.png
outputs/bumper-fem-rerun/nn-vs-rerun-run224.json
outputs/bumper-fem-rerun/nn-vs-rerun-run224.png
outputs/bumper-fem-rerun/postprocess-bumper.ps1
outputs/bumper-fem-rerun/rerun-Bumper_Beam_AP_meshed_0000.out
outputs/bumper-fem-rerun/rerun-Bumper_Beam_AP_meshed_0001.out
outputs/bumper-fem-rerun/rerun-engine.stderr.log
outputs/bumper-fem-rerun/rerun-engine.stdout.log
outputs/bumper-fem-rerun/rerun-starter.stderr.log
outputs/bumper-fem-rerun/rerun-starter.stdout.log
outputs/bumper-fem-rerun/run-bumper.ps1
outputs/bumper-fem-rerun/source-Bumper_Beam_AP_meshed_0000.out
outputs/bumper-fem-rerun/source-Bumper_Beam_AP_meshed_0001.out
outputs/bumper-fem-rerun/source-manifest.json
outputs/bumper-fem-rerun/summarize-bumper.py
outputs/bumper-fem-rerun/validation-review.md
outputs/experiment/.gitignore
outputs/experiment/README.md
outputs/experiment/capacity1024/console.log
outputs/experiment/capacity1024/executed_source.py
outputs/experiment/capacity1024/fem_vs_prediction.png
outputs/experiment/capacity1024/inference_reuse/console.log
outputs/experiment/capacity1024/inference_reuse/executed_inference.py
outputs/experiment/capacity1024/inference_reuse/inference_report.json
outputs/experiment/capacity1024/learning_and_error.png
outputs/experiment/capacity1024/mesh_contract.json
outputs/experiment/capacity1024/metrics.json
outputs/experiment/capacity1024/probe-console.log
outputs/experiment/capacity1024/probe/capacity_probe.json
outputs/experiment/capacity1024/probe/executed_probe.py
outputs/experiment/capacity1024/training_history.csv
outputs/experiment/capacity1024/transolver_cuda_checkpoint.pt
outputs/experiment/cpu-smoke.log
outputs/experiment/download.log
outputs/experiment/download_from_manifest.py
outputs/experiment/environment_freeze.txt
outputs/experiment/execution_status.json
outputs/experiment/fem_cpu_trial.yaml
outputs/experiment/full_mesh/checkpoint_roundtrip.json
outputs/experiment/full_mesh/console.log
outputs/experiment/full_mesh/executed_probe.py
outputs/experiment/full_mesh/full_mesh_probe.json
outputs/experiment/fulltraining/console.log
outputs/experiment/fulltraining/deformation_comparison.gif
outputs/experiment/fulltraining/deformed_geometry_comparison.png
outputs/experiment/fulltraining/executed_source.py
outputs/experiment/fulltraining/fem_vs_prediction.png
outputs/experiment/fulltraining/inference_reuse/console.log
outputs/experiment/fulltraining/inference_reuse/executed_inference.py
outputs/experiment/fulltraining/inference_reuse/inference_report.json
outputs/experiment/fulltraining/learning_and_error.png
outputs/experiment/fulltraining/mesh_contract.json
outputs/experiment/fulltraining/metrics.json
outputs/experiment/fulltraining/system_memory_samples.json
outputs/experiment/fulltraining/training_history.csv
outputs/experiment/fulltraining/transolver_cuda_checkpoint.pt
outputs/experiment/gpu1024/console.log
outputs/experiment/gpu1024/deformation_comparison.gif
outputs/experiment/gpu1024/deformed_geometry_comparison.png
outputs/experiment/gpu1024/executed_source.py
outputs/experiment/gpu1024/fem_vs_prediction.png
outputs/experiment/gpu1024/learning_and_error.png
outputs/experiment/gpu1024/mesh_contract.json
outputs/experiment/gpu1024/metrics.json
outputs/experiment/gpu1024/training_history.csv
outputs/experiment/gpu1024/transolver_cuda_checkpoint.pt
outputs/experiment/import-attempt.log
outputs/experiment/import-attempt2.log
outputs/experiment/import-attempt3.log
outputs/experiment/import-utf8.log
outputs/experiment/infer_checkpoint.py
outputs/experiment/inference_input_validation.json
outputs/experiment/inference_wrong_mesh_rejection.log
outputs/experiment/install-compact.log
outputs/experiment/make_deformation_visuals.py
outputs/experiment/make_mesh_contract.py
outputs/experiment/metric_definitions.json
outputs/experiment/official-attempt1.log
outputs/experiment/official-attempt2.log
outputs/experiment/official-attempt3.log
outputs/experiment/official-run-console.log
outputs/experiment/official_artifacts.json
outputs/experiment/official_cpu/.hydra/config.yaml
outputs/experiment/official_cpu/.hydra/hydra.yaml
outputs/experiment/official_cpu/.hydra/overrides.yaml
outputs/experiment/official_cpu/launch.log
outputs/experiment/official_cpu/stats/feature_stats.json
outputs/experiment/official_cpu/stats/node_stats.json
outputs/experiment/official_cpu/train.log
outputs/experiment/official_cpu_compat/.hydra/config.yaml
outputs/experiment/official_cpu_compat/.hydra/hydra.yaml
outputs/experiment/official_cpu_compat/.hydra/overrides.yaml
outputs/experiment/official_cpu_compat/checkpoints/TransolverOneShot.0.1.mdlus
outputs/experiment/official_cpu_compat/checkpoints/TransolverOneShot.0.2.mdlus
outputs/experiment/official_cpu_compat/checkpoints/checkpoint.0.1.pt
outputs/experiment/official_cpu_compat/checkpoints/checkpoint.0.2.pt
outputs/experiment/official_cpu_compat/launch.log
outputs/experiment/official_cpu_compat/stats/feature_stats.json
outputs/experiment/official_cpu_compat/stats/node_stats.json
outputs/experiment/official_cpu_compat/train.log
outputs/experiment/original_source_selfcheck.json
outputs/experiment/pip-check.log
outputs/experiment/prepare_official_trial.py
outputs/experiment/probe_capacity.py
outputs/experiment/probe_full_mesh.py
outputs/experiment/public_copy_policy.json
outputs/experiment/reader-selfcheck.log
outputs/experiment/requirements-direct.txt
outputs/experiment/requirements-lock-windows-cu126.txt
outputs/experiment/run_commands.json
outputs/experiment/run_fem_trial.py
outputs/experiment/runtime_dependency_closure.json
outputs/experiment/source_manifest.json
outputs/experiment/trial-console.log
outputs/fem-experiment/.gitignore
outputs/fem-experiment/README.md
outputs/fem-experiment/analyze_fem.py
outputs/fem-experiment/download-fem.ps1
outputs/fem-experiment/execution-metrics.json
outputs/fem-experiment/fem-results.png
outputs/fem-experiment/postprocess-fem.ps1
outputs/fem-experiment/prepare-publication.ps1
outputs/fem-experiment/run-fem.ps1
outputs/fem-experiment/run/BOX1BLA001.vtk
outputs/fem-experiment/run/BOX1BLA001.vtk.stderr.log
outputs/fem-experiment/run/BOX1BLA002.vtk
outputs/fem-experiment/run/BOX1BLA002.vtk.stderr.log
outputs/fem-experiment/run/BOX1BLA003.vtk
outputs/fem-experiment/run/BOX1BLA003.vtk.stderr.log
outputs/fem-experiment/run/BOX1BLA004.vtk
outputs/fem-experiment/run/BOX1BLA004.vtk.stderr.log
outputs/fem-experiment/run/BOX1BLA005.vtk
outputs/fem-experiment/run/BOX1BLA005.vtk.stderr.log
outputs/fem-experiment/run/BOX1BLA006.vtk
outputs/fem-experiment/run/BOX1BLA006.vtk.stderr.log
outputs/fem-experiment/run/BOX1BLA007.vtk
outputs/fem-experiment/run/BOX1BLA007.vtk.stderr.log
outputs/fem-experiment/run/BOX1BLA008.vtk
outputs/fem-experiment/run/BOX1BLA008.vtk.stderr.log
outputs/fem-experiment/run/BOX1BLA009.vtk
outputs/fem-experiment/run/BOX1BLA009.vtk.stderr.log
outputs/fem-experiment/run/BOX1BLA010.vtk
outputs/fem-experiment/run/BOX1BLA010.vtk.stderr.log
outputs/fem-experiment/run/BOX1BLA011.vtk
outputs/fem-experiment/run/BOX1BLA011.vtk.stderr.log
outputs/fem-experiment/run/BOX1BLT01.csv
outputs/fem-experiment/run/BOX1BL_0000.out
outputs/fem-experiment/run/BOX1BL_0001.out
outputs/fem-experiment/run/engine.stderr.log
outputs/fem-experiment/run/engine.stdout.log
outputs/fem-experiment/run/motcle
outputs/fem-experiment/run/readme.txt
outputs/fem-experiment/run/ref.extract
outputs/fem-experiment/run/starter.stderr.log
outputs/fem-experiment/run/starter.stdout.log
outputs/fem-experiment/run/th-converter.stderr.log
outputs/fem-experiment/run/th-converter.stdout.log
outputs/fem-experiment/run/th-probe.stderr.log
outputs/fem-experiment/run/th-probe.stdout.log
outputs/fem-experiment/validation-summary.json
outputs/gpu-experiment/README.md
outputs/gpu-experiment/disk-and-global-preservation.json
outputs/gpu-experiment/download-accounting.json
outputs/gpu-experiment/download.log
outputs/gpu-experiment/gpu-probe.json
outputs/gpu-experiment/install-dependencies.log
outputs/gpu-experiment/install-physicsnemo.log
outputs/gpu-experiment/install-torch-timing.json
outputs/gpu-experiment/install-torch.log
outputs/gpu-experiment/official-cu126-index.html
outputs/gpu-experiment/physicsnemo-import.log
outputs/gpu-experiment/probe.log
outputs/gpu-experiment/probe_gpu.py
outputs/gpu-experiment/requirements-local.txt
outputs/gpu-experiment/requirements-with-inherited.txt
outputs/gpu-experiment/setup-context.json
outputs/gpu-experiment/wheel-head.json
outputs/gpu-experiment/wheel-verified.json
results/cloud/README.md
results/cloud/access-gpu-command-evidence.json
results/cloud/access-gpu-recheck.json
results/cloud/access-gpu-recheck.py
results/cloud/cgroup-limits.json
results/cloud/cloud-environment.jpg
results/cloud/command-evidence.json
results/cloud/setup-probe.json
results/local/environment.json
scripts/probe_environment.py
```
