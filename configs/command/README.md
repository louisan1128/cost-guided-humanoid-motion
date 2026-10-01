# 명령 조건 PPO 실험

현재 상태와 `(vx, wz)`를 보고 humanoid의 관절 action을 출력한다.
명령만 바꿀 때는 위치나 관절을 초기화하지 않는다. 넘어지거나 에피소드가 끝나면 reset한다.

- `vx`: 몸의 heading 기준 전진 속도, m/s
- `wz`: world Z축 몸체 각속도, rad/s. 기울어진 상태의 정확한 Euler yaw 미분과는 다르다.
- 2~4초마다 새 명령: 전진 0.7~1.3 m/s, 회전 -0.5~0.5 rad/s.
- 명령을 뽑을 때 10% 확률로 둘 다 0. 제자리 회전은 현재 범위 밖이다.
- control 30 Hz, physics 120 Hz, 에피소드 최대 20초.

PPO는 MimicKit 구현을 사용한다. AMP는 motion reference와 실제 움직임을 구분하는 discriminator로 스타일 보상을 만든다.
고정 phase의 reference를 따라가는 DeepMimic 보상은 사용하지 않는다.

```
r_linear = exp(-4*(vx_actual-vx_command)^2 - 4*vy_actual^2)
r_angular = exp(-4*(wz_actual-wz_command)^2)
r_task = 0.7*r_linear + 0.3*r_angular
r_train = 0.5*r_task + 0.5*r_AMP
```

AMP 보상은 agent에서 더한다. 환경에서 중복해서 더하지 않는다.
넘어짐 종료와 물리 제약은 기존 환경을 사용한다. 정지 보상은 저속·저회전을 요구하지만 발 미끄러짐이나 에너지 효율까지 보장하지 않는다.

`motions.yaml`은 기존 다섯 모션 중 stop을 정지 유지 5초가 포함된 보완 reference로 바꾼 목록이다.
이는 새로운 mocap이 아니며 기존 stop과 같은 source group이다. 파일은 로컬 `data/reference_motions`에 있어야 한다.

## 학습

Isaac Lab과 MimicKit을 설치한 Python 환경에서 프로젝트 루트 기준으로 실행한다.
`MIMICKIT_ROOT`는 MimicKit checkout 경로로 바꾼다. 필요한 모션 파일은 [데이터 목록](../../data/README.md)에 있다.
저장소의 설정에는 프로젝트 상대 경로를 사용하며, launcher가 출력 폴더 안에 실행용 절대 경로 설정을 만든다.

```bash
cd /path/to/project_alpha
export MIMICKIT_ROOT=/path/to/MimicKit-main
python scripts/train_command_policy.py \
  --mimickit-root "$MIMICKIT_ROOT" \
  --num-envs 1024 --iterations 1000 --seed 0 \
  --out-dir outputs/command_amp_1000_seed0 --save-int-models
```

1024 env × 32 step = iteration당 32,768개 학습 samples다. AMP replay buffer는 첫 rollout 전체를 담아야 해서 65,536개로 설정했다. 이 설정으로 1024 env 한 iteration 실행을 확인했다. 4096 env는 현재 버퍼에서 사전 차단한다. 메모리와 속도는 환경마다 다르다.

출력 폴더에 기존 파일이 있으면 중단한다. 재시작할 때는 새 `--out-dir`를 사용한다.
`--model-file`은 같은 command policy의 가중치만 불러오며 optimizer 상태와 iteration을 복원하지 않는다. 기존 tracker 가중치는 구조가 달라 로드하지 않는다.

## 확인

```bash
python scripts/train_command_policy.py \
  --mimickit-root "$MIMICKIT_ROOT" \
  --mode test --num-envs 1 --visualize --test-episodes 3 \
  --env-config configs/command/eval.yaml \
  --model-file outputs/command_amp_1000_seed0/model.pt \
  --out-dir outputs/command_amp_eval_seed0
```

에피소드 시간 기준 고정 순서: 0~4초 직진 → 4~8초 느린 좌회전 → 8~12초 빠른 우회전 → 12~16초 정지 → 16~20초 출발.
넘어져 reset하면 처음부터 시작한다. 순서 끝까지 버텼는지 확인해야 한다.
이 스크립트는 기본 return/episode length를 출력한다. 학습 로그의 `Command_Vx_Mae`, `Command_Wz_Mae`는 기록 시점의 환경 평균이며 전체 평가 구간 평균이 아니다.
정식 수집 전에 구간별 추종 오차·넘어짐·정지 유지율 평가를 추가해야 한다.

## 이후

command 정책이 안정적이면 실제 rollout을 수집한다. 그 모션이 기존 고정 tracker에서도 실행되는지 먼저 확인한다.
이 관문을 통과한 데이터를 검색/MLP/Diffusion에 동일하게 제공한다. command 정책을 Diffusion 실행용 tracker로 부르지 않는다.

참고: [AMP 논문](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf), [MimicKit AMP 안내](https://github.com/xbpeng/MimicKit/blob/main/docs/README_AMP.md).
