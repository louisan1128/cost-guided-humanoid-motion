# 실행 환경과 재현 범위

이 저장소는 실험 코드와 결과를 공유하는 형태다. 데이터 다운로드부터 학습까지 한 명령으로 끝나는 배포 패키지는 아니다.

## 저장소만으로 볼 수 있는 것

- README, 최종 비교 보고서, episode별 CSV, 그림과 대표 영상
- 모델·환경·비용·평가 구현과 단위 검사
- 모델과 입력 파일의 hash, 실험 조건과 선택 기록

다음은 포함하지 않는다: reference motion 원본/파생본, 전체 rollout, checkpoint와 부모 checkpoint, 외부 MimicKit/Isaac Lab, 전체 렌더링 영상. 재배포 권한이 확인되지 않은 모션을 추가하지 않았다. 저장소 코드에 별도 배포 라이선스를 지정하지 않았으므로 외부 코드까지 일괄 MIT 등으로 표시하지 않는다.

## CPU 검사

최종 실험의 Python은 3.11, PyTorch는 2.7.0+cu128이다. 아래 검사는 GPU나 simulator를 시작하지 않으며 PyTorch와 pytest가 있는 환경에서 실행한다. 새 환경 설치를 검증한 lock 파일은 제공하지 않는다.

```bash
python3 scripts/check_publish.py
python -m pytest -q \
  tests/test_feasible_root_optimization.py \
  tests/test_guidance_only.py \
  tests/test_obstacle_geometry.py \
  tests/test_arrival_success.py \
  tests/test_future_conditioned_action.py \
  tests/test_state_action_sequence.py
```

`check_publish.py`는 Git 추가 대상의 대용량 파일, 모델·데이터 혼입, 일부 자격증명 패턴, Python 구문, 주요 안내 링크를 확인한다. staging·commit·push는 하지 않는다. 정규식 검사가 모든 비밀정보를 찾는다는 보장은 없다. 기록에 남아 있는 로컬 경로는 개수로 보고한다.

## 기존 환경에서 물리 평가

최종 기록의 환경: Ubuntu, Python 3.11.14, PyTorch 2.7.0+cu128, NumPy 1.26.4, Isaac Sim 5.1, Isaac Lab Python package 0.53.1, RTX 4060 Ti 8GB. Isaac Lab의 checkout 버전 번호와 Python package 번호는 다르다. 과거 환경 점검과 중간 실험 기록은 로컬에 보존한다.

아래 명령은 **기존 데이터·외부 환경·checkpoint가 준비된 로컬 환경용**이다. `python`은 해당 Isaac 환경의 Python을 뜻한다. 프로젝트 root에서 실행하고 출력 폴더는 새 이름을 사용한다.

```bash
python scripts/run_guidance_comparison_v2.py \
  --arm D2 --variant legacy \
  --manifest "$PWD/configs/releases/guidance_comparison_v2_manifest.json" \
  --scene-file reports/assets/guidance_comparison_v2/scenes.json \
  --source-shift 15 --seed 4504 --guide-scale 15 \
  --out-dir outputs/my_comparison_v2_D2

python scripts/run_guidance_comparison_v2.py \
  --arm M1 --variant legacy \
  --manifest "$PWD/configs/releases/guidance_comparison_v2_manifest.json" \
  --scene-file reports/assets/guidance_comparison_v2/scenes.json \
  --source-shift 15 --seed 4504 --lr 0.05 \
  --out-dir outputs/my_comparison_v2_M1
```

외부 MimicKit가 작업 디렉터리를 바꾸므로 manifest는 위처럼 절대 경로로 넘긴다.

전체 DP 집계는 seed 4505도 별도 출력 폴더로 실행한 결과를 포함한다. 한 번의 seed 결과를 33/48과 직접 비교하지 않는다. `--indices 0`을 추가하면 한 장면만 실행하지만 전체 평가의 재현은 아니다. 이 runner는 headless 평가이고 저장 영상은 별도 renderer가 만든다.

## 다른 컴퓨터로 옮길 때 남은 작업

1. 외부 MimicKit와 Isaac Lab을 준비한다. 로컬 MimicKit-main은 commit 식별이 안 된 snapshot이므로 최신 upstream과 같다고 가정하지 않는다.
2. 모델뿐 아니라 부모 prior, normalization 통계, 초기 source rollout, asset·motion, manifest의 frozen 입력을 모두 확보한다.
3. 코드·설정에 남은 `/home/an/...` 경로를 이식한다. 원본 manifest는 검증 근거로 보존하고 새 실행 manifest를 따로 만든다. Hash 검사를 끄고 실행하지 않는다.
4. CPU 검사와 작은 물리 smoke 후 재평가한다. 현재 다른 컴퓨터에서의 clean install은 미검증이다.

최종 H60 학습 코드는 `scripts/train_future_motion_v10.py`, dataset 구성은 `scripts/build_motion_h60_v10.py`다. 과거 reference 생성기용 `train_motion_mlp.py`와 혼동하지 않는다. 원본 학습 자료와 부모 모델이 없어도 README의 저장 결과는 확인할 수 있다.

## 공개 범위

`publication_manifest.json`은 공개 파일의 명시적 목록이다. 일부 과거 이름의 코드는 최종 진입점의 import 또는 frozen 의존성이므로 포함한다. 전체 개발 일지·중간 버전·실험 PDF 원본은 로컬에만 보존한다. 원래 결과 manifest는 로컬 전체 실험의 hash 기록이므로 공개하지 않은 이전 보고서와 모델 경로도 담고 있다. 공개용 `reports/final_results.md`는 기존 결과를 유지하고 링크만 정리한 사본이다.
