# 걷기 → 좌회전 자연 진입 구간 진단

기존 phase-26 지지 발 정렬은 좌회전 시작 직후의 발 접촉을 개선했지만, 좌회전 후반에 PPO tracker가 기준 자세에서 이탈했다. 클립 종료 때 실제 몸통 기울기 약 40°와 발 위치 오차 약 0.6~0.7m가 관찰됐다. 종료 시 `SUCC`는 아직 발 외 접촉이 발생하지 않았고 reference 길이에 도달했다는 뜻으로, 안정적인 완료를 보장하지 않는다.

## 변경한 reference

원본 MimicKit `long_walk1`에서 5.0~8.0초 선회 구간 대신 **4.5~8.0초**를 별도로 추출했다. 0.5초의 자연 진입 구간을 포함한다. 원본 파일은 변경하지 않았다. 같은 `scripts/build_contact_aligned_turn_probe.py`로 걷기 종료 phase를 다시 선택했다. 회전 시작에서는 왼발이 낮고 거의 정지해 있어, 걷기 phase 105/124에서 왼발을 딛고 연결한다. 양발의 최대 3D 기준 위치 차이는 약 0.035m다. 관절 blend나 새 PPO 학습은 없었다.

새 reference: `data/reference_motions/humanoid_probe_walk_turn_left_leadin_v1.pkl`. source 및 생성 조건은 동명의 `.json`에 기록했다. 평가 설정은 `configs/envs/deepmimic_humanoid_probe_walk_turn_left_leadin_v1.yaml`이다.

| 물리 평가, 같은 PPO 체크포인트·seed 0 | 기존 phase-26 연결 | 자연 진입 구간 연결 |
|---|---:|---:|
| 연결 시점 | 1.250초 | 1.908초 |
| 전체 reference 길이 | 4.250초 | 5.408초 |
| 종료 판정 | SUCC | SUCC |
| 종료 직전 실제 몸통 기울기 | 40.5° | 12.9° |
| 종료 직전 기준 몸통 기울기 | 약 8.2° | 약 8.2° |
| 종료 시 root 위치 오차 | 0.385m | 0.240m |
| 종료 시 왼발/오른발 body 위치 오차 | 0.748m / 0.595m | 0.279m / 0.235m |
| 종료 시 발 외 지면 접촉 | 없음 | 없음 |

실제 접촉 힘과 시계열 추종 오차는 `outputs/transition_diagnostics/walk_turn_left_leadin_seed0.csv` 및 `.json`에 있다. 이는 **첫 프레임에서 시작한 단일 물리 rollout**의 진단이다. 걷기 구간 길이, 지지 발 phase, 선회 reference의 진입 구간이 함께 달라졌으므로 어느 요소의 효과인지 분리되지 않는다. 목표 위치·방향 성공과 다양한 초기 상태에서의 안정성은 아직 평가하지 않았다. 마지막 root·발 위치 오차가 0은 아니므로 완벽한 추종도 아니다.

## 화면에서 한 번만 재생

```bash
cd /home/an/Desktop/MimicKit-main
/home/an/miniconda3/envs/issaclab/bin/python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --mode test --num_envs 1 --visualize true --test_episodes 1 --rand_seed 0 \
  --env_config /home/an/Desktop/project_alpha/configs/envs/deepmimic_humanoid_probe_walk_turn_left_leadin_v1.yaml \
  --model_file /home/an/Desktop/project_alpha/outputs/ppo_humanoid_five_motion_continue_2500_seed0/model.pt \
  --out_dir /tmp/ppo_walk_turn_left_leadin_view_v1
```

이 명령은 에피소드 종료 후 시작 자세로 반복 리셋하지 않는다. 사용자가 화면에서 회전 후반의 자세를 직접 확인해야 한다.
