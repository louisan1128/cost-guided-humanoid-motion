# 걷기 → 좌회전 지지 발 정렬 실험

문제: 기존 연결은 걷기 한 주기(1.033초) 뒤 좌회전 clip을 시작했다. 걷기 마지막 reference는 왼발이 낮고, 좌회전 첫 reference는 오른발이 낮고 거의 정지해 있다. 원본의 발바닥 추정 위치는 연결 한 프레임에서 오른발 약 0.36m 이동을 요구한다. 개별 모션 추종 성능과 별도로, 이 연결 불연속을 시험했다.

## 변경

`scripts/build_contact_aligned_turn_probe.py`는 걷기 시작 프레임은 그대로 유지하고, 한 주기 이후의 종료 phase만 고른다. `humanoid.xml`의 발 형상과 정기구학으로 발바닥 추정 위치를 계산한다. 좌회전 첫 부분에서 낮고 정지한 발은 오른발이었다. 후보 걷기 phase에서 오른발이 다른 발보다 2cm 이상 낮고 수평 속도가 0.1m/s 미만인 곳만 남긴다. 그중 연결 시 양발의 최대 3D 위치 차이가 가장 작은 phase 26/124를 선택했다. 좌회전 root 위치·heading은 걷기 끝점에 맞춘다. 관절 blend, 새 PPO 학습, Diffusion은 사용하지 않았다.

새 reference: `data/reference_motions/humanoid_probe_walk_turn_left_contact_v1.pkl`. source hash, 선택 기준, 경계 수치는 동명의 `.json`에 있다. 기존 reference와 설정은 보존했다. 새 평가 설정은 `configs/envs/deepmimic_humanoid_probe_walk_turn_left_contact_v1.yaml`이다.

| 지표 | 기존 연결 | 지지 발 정렬 연결 |
|---|---:|---:|
| 좌회전 시작 시점 | 1.033초 | 1.250초 |
| 오른발 reference 위치 차이 | 약 0.36m | 약 0.064m |
| 오른발 지지 상태 | 걷기 끝에서 비지지에 가까움 | 걷기 끝에서 낮고 거의 정지 |
| PPO의 종료 | FAIL, 2.167초 | SUCC, 4.267초(클립 종료) |
| 전환 이후 PPO 생존 시간 | 1.133초 | 3.017초 |
| 전환 전 0.25초 평균 왼발/오른발 실제 지면 반력 크기 | 426N / 0N | 228N / 270N |
| 전환 후 0.25초 평균 왼발/오른발 실제 지면 반력 크기 | 139N / 342N | 0N / 410N |
| 전환 후 0.25초 평균 관절 추종 오차 | 0.123rad | 0.077rad |

같은 `outputs/ppo_humanoid_five_motion_continue_2500_seed0/model.pt`, Isaac Lab, 단일 환경, 첫 프레임 시작, seed 0으로 한 episode씩 측정했다. 접촉 반력은 Isaac Lab의 접촉 센서에서 읽었고, 시간별 실제 발·root·관절 오차는 `outputs/transition_diagnostics/walk_turn_left_{raw,contact}_seed0.csv`에 있다. 요약 JSON도 같은 경로에 있다.

## 후속 검증: 클립 끝에서 추종 이탈

사용자가 화면에서 좌회전 후반의 넘어짐을 확인해 시간별 추적을 다시 분석했다. 접촉 정렬 reference의 마지막 약 0.3초에서 실제 몸통 기울기는 약 40°까지 커지지만, 같은 시점 reference의 기울기는 약 8°다. 마지막 3~4초 구간에는 root 위치 오차 약 0.27m, 발 body 위치 오차 약 0.32~0.35m가 누적되고, 마지막 순간 발 오차는 약 0.6~0.7m다. MimicKit는 클립 끝에서 아직 실패 접촉이 감지되지 않아 `SUCC`를 기록했다. 따라서 위 표의 `SUCC`는 **클립을 끝까지 진행했다**는 뜻이지 안정적인 회전 완료의 증거가 아니다. 재실행의 마지막 root 높이는 약 0.83m, 몸통 기울기는 약 40°였고 발 외 접촉은 아직 없었다.

같은 체크포인트로 좌회전 clip만 첫 프레임에서 단독 추종하면 3초 종료 시 몸통 기울기가 약 11°였다. 연결 직후 지지 발을 맞춘 것만으로는 이후의 전신 속도·자세 차이가 사라지지 않는다. 새 reference의 경계에서 오른발 목표 속도는 약 0.04m/s에서 7.69m/s로, root 각속도 크기는 약 0.68rad/s에서 12.28rad/s로 뛰는 120Hz 기준 불연속이 남는다. 실제 접촉을 유지하는 짧은 전환 구간이나 원본 회전 모션의 자연스러운 진입 구간이 다음 시험 대상이다.

이 결과는 **한 시작 상태와 한 좌회전 clip에서 지지 발 정렬이 효과적이었다**는 증거다. 시작 phase, 전환 시간, 좌회전 시작 시 실제 로봇 상태도 함께 달라졌으므로 각 요소의 독립 기여는 아직 분리되지 않았다. 정기구학의 발바닥 좌표는 실제 접촉 여부가 아니며, 실제 접촉은 별도 센서로 확인했다. 다른 속도·초기 상태·우회전·정지 연결의 성공을 의미하지 않는다.

## 화면 확인

```bash
cd /home/an/Desktop/MimicKit-main
/home/an/miniconda3/envs/issaclab/bin/python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --mode test --num_envs 1 --visualize true --test_episodes 1 --rand_seed 0 \
  --env_config /home/an/Desktop/project_alpha/configs/envs/deepmimic_humanoid_probe_walk_turn_left_contact_v1.yaml \
  --model_file /home/an/Desktop/project_alpha/outputs/ppo_humanoid_five_motion_continue_2500_seed0/model.pt \
  --out_dir /tmp/ppo_walk_turn_left_contact_view_v1
```

화면에서 1.25초 전후의 오른발 접촉과 좌회전 전체 3초 구간을 확인한다. 한 에피소드만 실행해 클립 종료 후 시작 자세로 리셋되는 장면과 낙상을 혼동하지 않도록 한다. 원래 연결과 같은 장면을 비교하려면 `--env_config`를 `deepmimic_humanoid_probe_walk_turn_left_v1.yaml`로 바꾼다.
