# 다섯 모션 연결 진단 v1

목적: 각각 추종되는 start, walk, left turn, right turn, stop reference를 같은 물리 에피소드에서 연결할 때 고정된 PPO tracker가 어디까지 버티는지 확인한다. 새 정책 학습은 하지 않았다.

## 방법

`scripts/build_five_motion_chain_probe.py`가 `start → walk → turn_left → walk → turn_right → walk → stop` 순서의 평가용 단일 reference를 만든다. 각 walk의 시작 phase는 바로 앞 모션의 마지막 관절 자세와 가장 가까운 위치로 선택한다. 각 연결에서 새 모션의 root 위치와 heading만 앞 모션의 마지막 root에 맞춘다. 프레임 주기는 모두 120 Hz로 맞췄고, 관절 blend나 물리 최적화는 하지 않았다. 따라서 파일 자체가 자연스럽거나 물리적으로 가능한 전환임을 보장하지 않는다. 이는 런타임 모션 ID 전환 구현이 아닌 하나의 연결 reference를 추종하는 진단이다.

전체 체인은 `data/reference_motions/humanoid_five_motion_chain_probe_v1.pkl`이고, 출처 해시·각 연결 시간·경계 자세 및 속도 차이는 동명의 `.json`에 있다. 각각의 연결을 분리한 `humanoid_probe_*.pkl`과 평가 설정 `configs/envs/deepmimic_humanoid_probe_*.yaml`도 생성했다.

평가에는 같은 `outputs/ppo_humanoid_five_motion_continue_2500_seed0/model.pt`, Isaac Lab, 1개 환경, 첫 프레임 시작, 3개 episode, seed 0을 사용했다. 제어 주기는 30 Hz다. 아래 길이는 평균값이며, 접촉 안정성·자연스러움이나 경계 전후 추종 오차의 직접 측정은 아니다.

| 평가 reference | 연결 시점 | 클립 끝 | 평균 episode 길이 | 관찰 |
|---|---:|---:|---:|---|
| 전체 체인 | 2.93초, 3.97초, 6.97초, 8.00초, 11.00초, 12.03초 | 13.97초 | 161 step ≈ 5.37초 | 좌회전 중 조기 종료; 이후 전환은 미도달 |
| walk → turn_left | 1.03초 | 4.03초 | 65 step ≈ 2.17초 | 회전 중 조기 종료 |
| turn_left → walk | 3.00초 | 4.03초 | 115 step ≈ 3.83초 | 걷기 구간 후반에 조기 종료 |
| walk → turn_right | 1.03초 | 4.03초 | 60 step = 2.00초 | 회전 중 조기 종료 |
| turn_right → walk | 3.00초 | 4.03초 | 121 step ≈ 4.03초 | 클립 끝까지 진행 |
| walk → stop | 1.03초 | 2.97초 | 50 step ≈ 1.67초 | 정지 구간에서 조기 종료 |

이 실험에서 `walk → stop`의 시작 관절 자세 차이는 약 1.75 rad의 합성 거리이고 root 속도 점프도 크다. 단순 연결 실패를 곧바로 PPO tracker 자체의 한계로 결론내릴 수 없다. 연결 reference의 불연속, 발 접촉, tracker 추종 능력을 분리해 확인해야 한다. `stop` 클립의 종료는 정지 유지 시험이 아니다.

## 전체 체인 화면 확인

```bash
cd /home/an/Desktop/MimicKit-main
/home/an/miniconda3/envs/issaclab/bin/python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --mode test --num_envs 1 --visualize true --test_episodes 1000000 --rand_seed 0 \
  --env_config /home/an/Desktop/project_alpha/configs/envs/deepmimic_humanoid_five_motion_chain_probe_v1.yaml \
  --model_file /home/an/Desktop/project_alpha/outputs/ppo_humanoid_five_motion_continue_2500_seed0/model.pt \
  --out_dir /tmp/ppo_five_motion_chain_view_v1
```

다른 쌍을 보려면 `--env_config`의 `chain` 부분을 `probe_walk_stop`, `probe_walk_turn_left` 등으로 바꾼다. 다음 단계에서는 연결 전후의 실제 root·관절·접촉 오차를 시간별로 기록하고, 동일한 경계에서 reference interpolation이나 학습 데이터 보강의 효과를 비교한다.
