# 환경 점검 기록

> 아래는 2026-09-14 G1 환경 점검의 역사적 기록이다.
> 현재 목표는 [최종 계획](plan.md)의 generic humanoid이며,
> 아래 GPU·설치·미완료 상태는 당시 관측이다. 현재 기능 지원 여부는 새 audit로 확인한다.

점검일: 2026-09-14. 시스템 조회, 설치 metadata, Python import와 `pip check`만 실행했다.
설치, dependency 변경, simulator 실행, 학습은 수행하지 않았다.

**판정: 의존성 충돌은 남아 있으나, 기존 환경에서 G1 headless PPO smoke test는 통과했다.**

당시 계획: 과제는 속도 조건부 보행이었다. 아래 G1 단일 걷기 설정과 motion은
초기 파이프라인 검증용이며, 여러 속도와 전환을 지원하는 task/reward를 검증한 기록은 아니다.

## 하드웨어

| 항목 | 확인 결과 |
|---|---|
| OS | Ubuntu 24.04.3 LTS, x86_64 |
| CPU | AMD Ryzen 5 7500F, 6 cores / 12 threads |
| RAM | `free -h` 기준 31 GiB, available 약 26 GiB |
| GPU | NVIDIA GeForce RTX 4060 Ti |
| VRAM | 총 8188 MiB, 조회 당시 free 7282 MiB |
| Driver | 580.159.03 |
| 프로젝트 파일시스템 | 457G 중 415G 사용, 가용 약 19G (`df -h`) |

Sandbox 안의 첫 `nvidia-smi`는 driver 통신에 실패했으나, 승인된 외부 조회는 정상 동작했다.
이 오류를 실제 driver 고장으로 해석하지 않는다. CUDA tensor 연산은 아직 검증하지 않았다.

## 기존 설치

기본 Conda는 `/home/an/miniforge3`지만 Isaac Lab 환경은 별도 Miniconda 아래에 있다.
환경은 절대 prefix로 구분한다.

| 항목 | 관측값 |
|---|---|
| 기존 Isaac 환경 | `/home/an/miniconda3/envs/issaclab` — 이름의 철자 그대로 |
| Python | 3.11.14 (직접 실행 확인) |
| PyTorch | 2.7.0+cu128 (import 확인) |
| PyTorch CUDA build | 12.8 — 시스템 CUDA toolkit 버전과 구분 |
| NumPy | 1.26.4 (import 확인) |
| Isaac Sim | 5.1.0.0 (설치 metadata 확인, 앱 실행 안 함) |
| Isaac Lab checkout VERSION | 2.3.0 |
| Isaac Lab Python package | 0.53.1 — checkout VERSION과 별도 번호 |
| Isaac Lab HEAD | `4f3e808ca75fd8ea07c20d188f067980eeab411f` |
| Isaac Lab 경로 | `/home/an/Desktop/mimickit/MimicKit/IsaacLab` |
| Isaac Lab 설치 방식 | 해당 checkout의 `source/isaaclab`을 editable install |
| 대체 환경 관측 | `/home/an/miniforge3/envs/mimickit_newton`: Python 3.10.19, Newton 0.2.0 metadata |

Isaac Lab tracked 변경은 없었다. 기존 Newton 환경은 현재 MimicKit README가 명시한
Newton 1.0.0과 다르므로 즉시 사용 가능한 대안이라고 간주하지 않는다.

## 기존 MimicKit 소스와 데이터

| 경로 | 상태 | 사용 판단 |
|---|---|---|
| `/home/an/Desktop/MimicKit-main` | `.git` 없음; G1 asset/motion 및 기본 설정 존재 | 자료 확인용. 소스 commit 식별 불가 |
| `/home/an/Desktop/mimickit/MimicKit` | HEAD `b2d3cfa69ee7f407aac23b7ceadc41f6631cdd68`, tracked 수정 다수 | 기존 작업 보존. 그대로 기준 구현으로 사용하지 않음 |

수정된 파일에는 PPO, DeepMimic env, base env, run.py, engine/env 설정 등이 포함된다.
추후 공식 commit의 별도 checkout을 사용하며 기존 저장소를 reset하거나 덮어쓰지 않는다.

`MimicKit-main`에서 확인한 파일 식별값:

| 파일 | SHA-256 |
|---|---|
| `data/assets/g1/g1.xml` | `ecca454fc889d92febcbd615ff02da95f863b4e1022a84ee7209af85098d7434` |
| `data/assets/g1/g1.usd` | `4c6c929308a804e1c63c8bf37ac5f5d620c5b0b794d35a757b2c8a1688dc5cea` |
| `data/motions/g1/g1_walk.pkl` | `ffaf84e89de6b7388c98195b459accde666b1ca1edc43a5dd697964be9dfc7e2` |

이 hash는 현재 로컬 파일의 식별값이다. 공식 배포본과의 동일성이나 재배포 권한을 뜻하지 않는다.
구현 시작 후 로컬 MimicKit loader로 `g1_walk.pkl`을 읽어 reference 속도를 측정했다.
Checkpoint는 아직 불러오지 않았다.

## Reference 걷기 속도 측정

`g1_walk.pkl`은 120 FPS, 125 frames, 약 1.033초이며 한 주기 동안 x축으로 약 1.036m 이동한다.
평균 x축 속도는 약 1.002m/s였다. 이 값은 reference trajectory의 수치이며 PPO rollout 결과가 아니다.
따라서 첫 curriculum의 목표 속도 범위를 0.8~1.2m/s로 잡았다.
실제 expert가 이 범위를 안정적으로 추종하지 못하면 원본 속도 주변으로 더 좁힌다.

## 기존 설정에서 확인한 초기 실행 후보

- G1 XML에서 hinge joint 29개 확인. Runtime observation/action 차원은 미확정.
- G1 env: phase observation 비활성화, 미래 target observation 활성화.
  Student에도 같은 reference 정보를 제공하며 실제 필드·시간 정렬은 계약 작성 시 확인한다.
- Engine/env 설정 후보: position control, policy 30 Hz / physics 120 Hz, episode 10초.
- 원본 G1 args는 Isaac Gym과 환경 4096개를 지정한다. 첫 실행에서는 Isaac Lab engine과
  환경 1개를 명시하고, 본 학습 환경 수는 메모리 측정 이후 결정한다.

이는 소스 설정 확인 결과이며 실행으로 검증한 값은 아니다.

## 발견한 의존성 충돌

`/home/an/miniconda3/envs/issaclab/bin/python -B -m pip check`가 exit code 1을 반환했다.

| 요구 패키지 | 요구 조건 | 현재 설치 |
|---|---|---|
| isaacsim-core | packaging ==23.0 | 26.0 |
| isaacsim-core | rtree ==1.3.0 | 1.2.0 |
| fastapi | starlette >=0.40.0,<0.46.0 | 0.49.1 |
| isaacsim-kernel | sentry-sdk ==2.29.1 | 2.52.0 |
| isaaclab-rl | packaging <24 | 26.0 |
| isaaclab-rl / isaaclab-tasks | protobuf >=4.25.8,!=5.26.0 | 4.25.3 |

추가로 `Rtree 1.2.0 is not supported on this platform`이 출력됐다.
NumPy dist-info가 1.26.0과 1.26.4 둘 다 있으나 실제 import는 1.26.4였다.
따라서 이 환경을 그대로 export해서 프로젝트 lock 파일로 사용하지 않는다.

## 공식 요구사항과 비교

[Isaac Sim 5.1 요구사항](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)은
Ubuntu 22.04/24.04, RAM 32GB, SSD 50GB, RTX 4080 / VRAM 16GB를 최소 표에 제시한다.
현재 GPU/VRAM은 이 표에 미달하고, 여유 디스크 19G로 새 환경과 데이터를 중복 설치하기도 어렵다.
소규모 headless scene의 실행 가능 여부는 별도 측정해야 하며 학습 가능성을 보장하지 않는다.
동 문서는 Isaac Sim 5.1을 지원 종료 릴리스로 표시한다. 여기서는 기존 설치 및 재현 후보로만 다룬다.

## 설치·검증 후보와 결정

- 기본 backend 후보: Isaac Lab / PhysX.
- 재현 후보: Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.0.
- MimicKit README가 검증했다고 명시한 Isaac Lab commit:
  `2ed331acfcbb1b96c47b190564476511836c3754` (VERSION 2.3.0).
- 기존 Isaac Lab HEAD는 위 commit과 다르므로 같은 버전 문자열만으로 호환 판정하지 않는다.
- PyTorch 2.7.0+cu128은 기존 설치 관측값이다. 최종 lock은 의존성 정리 및 실행 검증 이후 작성한다.
- MimicKit 소스 commit: 별도 checkout 준비 시 확정. 이번 점검에서는 다운로드하지 않았다.
- 별도 환경 예정 위치: 프로젝트의 `.venv/` (Conda prefix로 사용 가능, Git 제외).
- 별도 소스 예정 위치: 프로젝트의 `external/MimicKit/`, `external/IsaacLab/` (아직 생성하지 않음).

즉시 전체 재설치하거나 기존 Conda 환경을 수정하지 않는다. 저장 공간을 먼저 마련하고,
위 후보의 의존성을 별도 환경에서 해결한 다음 최소 실행으로 검증한다.
8GB에서 최소 실행이 불가능하면 MimicKit의 다른 backend를 별도 검토한다.
Backend를 바꾸게 되면 본 학습 전에 문서와 설정을 갱신하고 이후 하나만 사용한다.

## 다음 순서

1. 사용할 저장 공간 확보: 새 환경 설치 시 최소 50GB 이상 여유를 계획하고 데이터 공간은 별도 산정.
2. 별도 source checkout과 환경 생성, 출처·commit 기록.
3. 위 충돌이 없는 dependency 구성 확인 (`pip check`).
4. [x] CUDA/Isaac Lab에서 G1 한 환경과 64-sample PPO smoke 실행.
5. [ ] GUI 또는 영상으로 G1 걷기 motion 확인.
6. [x] 작은 PPO 실행, checkpoint 저장 및 별도 test process에서 재실행.

이번 단계 완료: 과제/모델/motion 선택, 설치 현황과 충돌 확인, 환경 후보 및 경로 결정.
미완료: 충돌 없는 설치 확정, 영상 확인, baseline 재현, 병렬 환경 수와 학습 처리량 측정.

## 근거

- [MimicKit 설치 안내](https://github.com/xbpeng/MimicKit#installation)
- [검증 대상 Isaac Lab VERSION](https://github.com/isaac-sim/IsaacLab/blob/2ed331acfcbb1b96c47b190564476511836c3754/VERSION)
- [G1 기본 환경](https://github.com/xbpeng/MimicKit/blob/main/data/envs/deepmimic_g1_env.yaml)
