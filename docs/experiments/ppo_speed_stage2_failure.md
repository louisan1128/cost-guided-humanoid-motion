# PPO 속도 조건부 정책 실패 기록

기록일: 2026-09-21

## 대상

- Checkpoint: `outputs/ppo_speed_curriculum_seed0/stage2_range_0p8_1p2/model.pt`
- 학습 명령 범위: episode별 `0.8~1.2 m/s`
- 환경 설정: `configs/envs/g1_speed_range_0p8_1p2.yaml`
- 속도 보상 비중: `0.3`
- DeepMimic root velocity 보상 비중: `0.1`

## 판정

고정된 `0.8`, `1.0`, `1.2 m/s` 명령에서 실제 보행 속도가 충분히 구분되지 않았다.
따라서 이 checkpoint는 **속도 명령 추종 실패**로 분류하며 이후 teacher 또는 dataset 생성에
사용하지 않는다. 기존 파일은 실패 분석과 재현을 위해 보존한다.

이번 육안 확인에서는 episode별 실제 평균 속도를 별도로 저장하지 않았으므로 정량적인
속도 오차는 확정하지 않는다. 실패 원인을 하나로 단정하지 않고, 원래 걷기 속도를 유도하는
모방 보상과 상대적으로 작은 속도 보상 사이의 충돌을 우선 가설로 둔다.

## 다음 실험

기존 걷기 checkpoint에 속도 입력을 추가한 `baseline_speed_init.pt`에서 다시 시작한다.
수정 설정은 원래 root velocity 모방 항을 제거하고 속도 추종 보상의 비중을 높인다.

- 설정: `configs/envs/g1_speed_range_0p8_1p2_reward_v2.yaml`
- `reward_root_vel_w: 0.0`
- `speed_reward_weight: 0.7`
- `track_global_root: false`
- 범위: `0.8~1.2 m/s`
- 환경 수: 4096
- PPO iteration: 500

재학습 결과도 고정된 `0.8`, `1.0`, `1.2 m/s`에서 각각 평가한 뒤 다음 범위로 확장한다.
