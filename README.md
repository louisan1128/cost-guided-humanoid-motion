# Humanoid Motion Learning

휴머노이드가 목표에 맞게 움직이는 모션을 생성하고, 실제 물리 시뮬레이션에서도 실행할 수 있는지 실험하는 개인 프로젝트입니다.
MimicKit의 generic humanoid와 Isaac Lab을 사용합니다.

최종적으로는 같은 데이터로 검색·연결, MLP, Motion Diffusion을 비교합니다.
Diffusion이 만드는 것은 미래 전신 모션 reference이고, 관절을 제어하는 것은 PPO tracker입니다.
아직 Diffusion 학습 단계까지 진행한 것은 아닙니다.

## 지금 하는 일

- 출발, 걷기, 좌우 회전, 정지 모션을 준비하고 PPO tracker를 학습했습니다.
- 개별 모션은 재생·추종을 확인했습니다. 모션 연결에서는 발 위상과 정지 유지 문제가 있어 보완했습니다.
- 긴 연결 동작은 여전히 만족스럽지 않아, 명령이 바뀌는 상황을 RL로 학습하는 실험을 추가했습니다.
- 새 정책은 `(vx, wz)`를 받아 움직입니다. PPO로 학습하고 AMP로 기존 모션의 스타일을 참고합니다. 현재는 실행 점검 단계입니다.

[지금 실행하는 방법](configs/command/README.md) · [진행 기록](docs/progress.md) · [전체 계획](docs/plan.md)

## 실행 전 준비

Ubuntu에서 Isaac Lab·MimicKit이 동작하는 Python 환경과 MimicKit generic humanoid asset이 필요합니다. 이 저장소에는 외부 시뮬레이터와 모션 파일이 포함되지 않습니다. [필요한 모션 파일](data/README.md)을 준비한 뒤 [명령 조건 PPO 실행 안내](configs/command/README.md)의 경로를 설정하면 됩니다. 현재 검증한 실행 환경은 [환경 기록](docs/environment.md)에 적었습니다.

지금 확인된 것은 1024개 환경에서 한 iteration이 정상 완료된 것까지입니다. 1000 iteration 결과와 연속 명령 추종 성능은 아직 평가하지 않았습니다. 과거 실험용 설정 중에는 이 컴퓨터의 절대 경로가 남아 있어 그대로 재현하려면 경로를 바꿔야 합니다.

## 폴더

| 경로 | 내용 |
|---|---|
| `src/humanoid_imitation/` | 프로젝트 환경 adapter |
| `configs/command/` | 현재 명령 조건 PPO 실험 |
| `configs/envs/`, `configs/datasets/` | 기존 tracker 환경과 reference 목록 |
| `scripts/` | 학습 실행, 모션 변환, 연결과 평가 도구 |
| `data/`, `outputs/` | 로컬 모션·모델·로그. Git 추적 제외 |
| `docs/experiments/` | 개별 실험 기록 |
| `reports/` | 보고서와 개발 글 |

예전 G1 속도 실험과 실패한 결과도 남겨뒀습니다. 실행 스크립트의 경로를 깨뜨리지 않도록 대규모 이동은 하지 않았습니다.
외부 코드와 프로젝트 작업 범위는 [출처 기록](docs/provenance.md)에 정리했습니다.
