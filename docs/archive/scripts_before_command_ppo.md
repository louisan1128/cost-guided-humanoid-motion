# 실행 진입점

> 아래 실행 안내는 기존 G1 속도 PPO 실험 이력이다.
> 최종 목표 조건부 generic humanoid 프로젝트는 [최종 계획](../docs/plan.md)을 따른다.
> 출발·정지 모션별 generic humanoid PPO 추종 실험은 구현·학습했다.
> 최종 계획의 공통 tracker·goal dataset·모션 생성기는 아직 구현하지 않았다.
> 실험 결과와 PPO 물리 시각화 명령은 [출발·정지 PPO 실험](../docs/experiments/ppo_mixamo_start_stop_xyfix.md)을 따른다.

## Generic humanoid 출발·정지 후보 확인

원본 DeepMimic의 출발·정지 후보를 MimicKit generic humanoid 형식으로 변환했다.
아래 화면은 저장된 pose를 재생하는 것이며 PPO 물리 추종 결과가 아니다.

출발(`ready → walk`) 후보:

```bash
cd /home/an/Desktop/MimicKit-main
/home/an/miniconda3/envs/issaclab/bin/python mimickit/run.py \
  --arg_file args/view_motion_humanoid_args.txt \
  --env_config /home/an/Desktop/project_alpha/configs/envs/view_motion_humanoid_start.yaml \
  --num_envs 1 \
  --visualize true
```

정지(`walk → ready`) 후보:

```bash
cd /home/an/Desktop/MimicKit-main
/home/an/miniconda3/envs/issaclab/bin/python mimickit/run.py \
  --arg_file args/view_motion_humanoid_args.txt \
  --env_config /home/an/Desktop/project_alpha/configs/envs/view_motion_humanoid_stop.yaml \
  --num_envs 1 \
  --visualize true
```

변환을 다시 수행할 때는 원본 파일을 받은 뒤 다음 형식으로 실행한다.

```bash
python scripts/convert_deepmimic_humanoid_motion.py INPUT.txt OUTPUT.pkl
```

## 눈으로 확인하기

원본 G1 걷기 reference motion을 연다.

```bash
/home/an/miniconda3/envs/issaclab/bin/python scripts/view_reference_motion.py \
  --mimickit-root /home/an/Desktop/MimicKit-main
```

이 화면은 저장된 motion pose를 재생한 것이며 PPO 제어 결과가 아니다.
Isaac Lab 창을 닫으면 실행도 종료된다.

현재 smoke checkpoint의 물리 rollout은 아래처럼 연다.

```bash
/home/an/miniconda3/envs/issaclab/bin/python scripts/train_speed_expert.py \
  --mimickit-root /home/an/Desktop/MimicKit-main \
  --agent-config configs/agents/ppo_g1_speed_smoke.yaml \
  --mode test \
  --model-file outputs/ppo_speed_smoke/model.pt \
  --test-episodes 1000000 \
  --num-envs 1 \
  --visualize
```

Smoke checkpoint는 64 samples만 학습했기 때문에 넘어지는 것이 정상이다.
이 화면에서는 PPO action이 PhysX에 적용되는 실행 경로만 확인한다.

첫 PPO expert 실행 진입점과 환경 adapter를 구현했다. Simulator 호환성은 아직 검증 중이다.

## 첫 smoke test

```bash
/home/an/miniconda3/envs/issaclab/bin/python scripts/train_speed_expert.py \
  --mimickit-root /path/to/clean/MimicKit \
  --agent-config configs/agents/ppo_g1_speed_smoke.yaml \
  --num-envs 1 \
  --max-samples 64
```

Smoke 설정은 연결 확인 전용이며 학습 결과로 사용하지 않는다. 이후 환경 수와 처리량을 측정하고
`ppo_g1_speed_stage1.yaml`로 본 학습을 시작한다. 현재 로컬 MimicKit 작업본에는 기존 수정이 있으므로
재현 가능한 clean checkout을 준비한 뒤 경로를 지정하는 것이 원칙이다.

Checkpoint 재실행은 같은 command에 다음 인자를 사용한다.

```bash
--mode test --model-file outputs/ppo_speed_smoke/model.pt --test-episodes 2
```

## 속도 curriculum 이어서 학습

기존 500-iteration DeepMimic checkpoint에서 다음 세 단계를 자동으로 이어서 학습한다.

1. 목표 속도 `1.0 m/s` 고정
2. episode마다 `0.8-1.2 m/s`에서 목표 속도 선택
3. episode마다 `0.6-1.4 m/s`에서 목표 속도 선택

각 단계는 `4096 environments`, `500 PPO iterations`를 사용한다. 앞 단계의 최종
`model.pt`를 다음 단계가 자동으로 불러온다.

```bash
cd /home/an/Desktop/project_alpha
./scripts/train_speed_curriculum.sh /home/an/Desktop/MimicKit-main
```

기존 정책은 관측이 849차원이고 속도 조건부 정책은 850차원이다. 스크립트는 먼저
`convert_baseline_checkpoint.py`를 호출해 목표 속도 입력 하나를 추가한다. 새 입력의 actor와
critic 가중치는 0으로 초기화하므로 변환 직후 행동은 기존 정책과 같다.

결과는 `outputs/ppo_speed_curriculum_seed0/` 아래에 단계별로 저장한다. TensorBoard는 다음과
같이 연다.

```bash
tensorboard --logdir outputs/ppo_speed_curriculum_seed0
```

출력 폴더가 이미 있으면 서로 다른 실험이 섞이지 않도록 실행을 중단한다. 환경 수나 단계별
iteration 수를 바꿀 때는 다음처럼 실행한다.

```bash
NUM_ENVS=1024 ITERATIONS=100 ./scripts/train_speed_curriculum.sh /home/an/Desktop/MimicKit-main
```

## 진입점

- `train_speed_expert.py`: MimicKit PPO와 프로젝트의 속도 조건부 환경 연동
- `train_speed_curriculum.sh`: 세 속도 단계의 PPO 학습을 순서대로 실행
- `convert_baseline_checkpoint.py`: 기존 849차원 정책을 속도 입력이 있는 850차원 정책으로 변환
- 이전 계획의 `collect_demonstrations.py`, `train_bc.py`, `train_diffusion.py`,
  `evaluate.py`, `summarize_results.py`는 현재 존재하는 실행 진입점이 아니다.
  최종 계획에 맞는 CLI는 구현 시 검증하여 안내한다.
