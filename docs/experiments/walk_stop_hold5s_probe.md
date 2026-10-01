# 걷기→정지 5초 유지 진단

기존 2초 정지 유지 reference의 마지막 자세를 3초 더 복제해, 정지 시작 약 3.25초부터 총 5초 동안 같은 자세를 요구했다. 원본 reference와 PPO checkpoint는 변경하지 않았다. 생성 스크립트는 `scripts/extend_stop_hold_probe.py`, 새 reference는 `data/reference_motions/inertial_walk_stop_500_v1_hold5s_probe.pkl`, 환경 설정은 `configs/envs/transition_inertial_walk_stop_hold5s_probe.yaml`이다.

Isaac Lab 단일 환경, seed 0, 첫 프레임 고정 시작으로 기존 선택 checkpoint `outputs/ppo_transition_inertial_v2_add200_seed0/int_models/model_0000000050.pt`를 물리 실행했다. 기록은 `outputs/transition_diagnostics/walk_stop_hold5s_selected150_seed0.csv`와 같은 이름의 JSON이다.

| 정지 후 구간 | 몸체 XY 평균 속도 | 최고 속도 | 평균 몸체 기울기 |
|---|---:|---:|---:|
| 3.25–4.25초 | 0.026 m/s | 0.116 m/s | 5.7° |
| 4.25–5.25초 | 0.065 m/s | 0.150 m/s | 7.0° |
| 5.25–6.25초 | 0.688 m/s | 1.443 m/s | 15.7° |

6.23초에 `FAIL`로 종료했다. 이때 몸체 높이는 0.527 m, 기울기는 33.6°, root XY 속도는 1.443 m/s였다. 즉 기존의 2초 유지 구간에서는 거의 정지했지만, 그 뒤에는 속도와 기울기가 증가해 5초를 유지하지 못했다. 기존 평가의 clip 종료 `SUCC`는 지속적인 정지 능력을 입증하지 않는다. 정지 시작부터 약 3초 후 실패했다.

이것은 동일 초기 상태의 단일 rollout이며 실패 원인을 단정하지 않는다. 현재 PPO의 학습 자료에는 이 5초 정지 구간이 없었으므로 긴 정지 유지가 학습됐다고 볼 수 없다. 다음에는 정지 자세의 발 접촉·중심 배치를 확인하고, 긴 정지 구간을 공통 tracker 학습 reference에 포함할지 결정한 뒤 모든 모션을 같은 tracker로 재평가한다.

## 500 iteration 추가 학습 결과

기존 선택 checkpoint `outputs/ppo_transition_inertial_v2_add200_seed0/int_models/model_0000000050.pt`의 **가중치**에서 새 PPO 학습을 시작했다. optimizer와 iteration을 이어받는 완전 resume는 아니다. 원래 11개 reference 중 걷기→정지 하나만 5초 유지 버전으로 교체했고 가중치, 보상, 관측, PPO 설정은 유지했다. `configs/datasets/humanoid_transition_stop_hold5s_pilot.yaml`과 `configs/envs/deepmimic_humanoid_transition_stop_hold5s_pilot.yaml`을 사용했다. 4096환경 × 32 step × 500 iteration = 65,536,000환경 step이며 seed 0이다. 출력은 `outputs/ppo_transition_stop_hold5s_500_seed0/`에 별도 보존했다.

동일한 첫 프레임·5초 정지 reference의 물리 rollout 비교:

| 모델 | 종료 | 종료 시각 | 마지막 1초 평균 몸체 XY 속도 |
|---|---|---:|---:|
| 기존 선택 checkpoint | FAIL | 6.23초 | 측정 불가 |
| 100 iteration pilot 마지막 checkpoint | FAIL | 5.70초 | 측정 불가 |
| 500 iteration 마지막 checkpoint | SUCC | 8.27초 | 0.022 m/s |

500 iteration 모델의 정지 구간 3.25–8.25초 평균 몸체 XY 속도는 0.037 m/s였고, 마지막 속도는 0.025 m/s였다. 마지막 몸체 기울기는 4.7°였다. 원자료는 `outputs/transition_diagnostics/walk_stop_hold5s_500final_seed0.csv`와 JSON이다.

같은 checkpoint를 기존 여섯 전환과 원본 다섯 모션에 적용했을 때, 고정 첫 프레임의 여섯 전환은 모두 clip 끝까지 도달했다. 원본 start·좌회전·우회전·stop은 각 1회 성공, looping walk는 시간 제한까지 진행했으며 실패하지 않았다. 다만 좌회전 관련 일부 끝 자세 기울기는 기존 모델보다 커졌다. 원자료는 `outputs/transition_diagnostics/stop_hold5s_500_full_suite_seed0/`이다.

이것은 학습에 포함된 reference의 고정 시작, 단일 seed 물리 시험이다. 다양한 초기 상태·외란·하나의 긴 연속 전환·독립 test motion·시각적 자연스러움은 아직 검증하지 않았다. Diffusion 단계의 범용 tracker로 채택하기 전에 연속 전환과 목표 수행 평가가 필요하다.

직접 보기:

```bash
cd /home/an/Desktop/project_alpha
/home/an/miniconda3/envs/issaclab/bin/python scripts/view_transition_v2.py walk_stop_hold5s --episodes 1
```
