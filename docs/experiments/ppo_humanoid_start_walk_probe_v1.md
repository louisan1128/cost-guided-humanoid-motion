# 출발 → 걷기 전환 진단 v1

목적: 기존 five-motion PPO tracker가 한 물리 에피소드 안에서 출발 reference 다음 걷기 reference를 받을 때 즉시 실패하는지 확인한다. 새 정책을 학습하지 않는다.

## 평가 reference

`scripts/build_start_walk_probe.py`가 `mixamo_start_xyfix.pkl`을 걷기 클립의 120 fps로 재표본화하고, 출발 마지막 관절 자세에 가장 가까운 걷기 phase를 선택한다. 걷기 root 위치와 heading을 출발 끝점에 맞춘 뒤, 세 걷기 주기를 연결한다. 프레임 사이에 blend나 물리 최적화는 적용하지 않았다. 같은 에피소드 안에서 약 2.933초에 출발에서 걷기로 넘어가며 시뮬레이터는 리셋되지 않는다. 이 파일은 전환 진단용 reference이지 학습 데이터가 아니다.

생성 파일: `data/reference_motions/humanoid_start_to_walk_probe_v1.pkl`. 선택한 걷기 phase와 경계 차이는 동명의 `.json`에 기록했다. 선택된 walk frame은 25/124이고, 연결 순간의 joint pose distance는 약 0.859 rad, root rotation 차이는 약 0.208 rad이다. 경계가 정확히 연속인 모션은 아니다.

## 1차 물리 평가

`configs/envs/deepmimic_humanoid_start_to_walk_probe_v1.yaml`과 `outputs/ppo_humanoid_five_motion_continue_2500_seed0/model.pt`로 Isaac Lab headless test를 3 episode 실행했다. Mean Episode Length는 181 step이다. 제어 주기 30 Hz와 클립 길이 약 6.033초를 고려하면 3회 모두 약 2.933초의 전환 시점을 넘어 클립 종료까지 진행한 것이다. Mean Return은 약 169.15였다.

이 결과는 조기 종료가 없었다는 작은 smoke test다. 자연스러움, 접촉 안정성, 전환 직전·직후 tracking error, 목표 제어 능력을 증명하지 않는다. 동일 reference 1개를 세 번 실행했으므로 다양성 평가도 아니다.

## 재현

```bash
cd /home/an/Desktop/MimicKit-main
/home/an/miniconda3/envs/issaclab/bin/python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --mode test --num_envs 1 --visualize true --test_episodes 1000000 \
  --env_config /home/an/Desktop/project_alpha/configs/envs/deepmimic_humanoid_start_to_walk_probe_v1.yaml \
  --model_file /home/an/Desktop/project_alpha/outputs/ppo_humanoid_five_motion_continue_2500_seed0/model.pt \
  --out_dir /tmp/ppo_start_walk_probe_view_v1
```

화면에서 첫 2.93초는 출발, 이후 약 3.10초는 걷기다. 다음 검증은 전환 전후의 실제 root 속도·관절·발 접촉 오차를 episode별로 기록하는 것이다.
