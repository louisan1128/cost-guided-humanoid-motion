# PPO로 목표 속도를 따르는 Humanoid 보행 정책 만들기: 설계와 첫 Smoke Test

## 프로젝트를 시작한 이유

이 프로젝트의 최종 목표는 RL expert가 학습한 humanoid 보행 능력을 demonstration으로 수집하고,
MLP Behavior Cloning과 Diffusion Policy가 그 능력을 얼마나 잘 모방하는지 확인하는 것이다.

단순히 정해진 걷기 동작 하나를 재현하는 데서 끝내지 않고 다음 질문을 다루고 싶었다.

> 목표 속도를 입력받아 걷는 RL expert의 제어 능력을 Diffusion Policy가 얼마나 정확하고
> 안정적으로 전달받을 수 있을까?

최종적으로는 일정 속도 유지, 가속과 감속, 학습 데이터에서 제외한 중간 속도,
외란 이후 속도 회복을 비교할 계획이다. Diffusion이 MLP보다 반드시 좋아야 하는 것은 아니다.
동일한 조건에서 성능과 계산 비용을 비교하고 실패 원인을 분석하는 것이 프로젝트의 핵심이다.

전체 파이프라인은 다음과 같다.

```text
속도 조건부 PPO expert
→ 여러 속도와 속도 전환 demonstration 수집
→ MLP BC / Diffusion Policy 학습
→ closed-loop 속도 제어
→ robustness 평가와 실패 분석
→ 선택적 residual RL
```

이번 글에서는 첫 단계인 속도 조건부 PPO 환경과 smoke test까지 정리한다.

## 모션 재생 속도를 높이는 것과 목표 속도 제어의 차이

걷기 모션을 빠르게 재생하는 방법도 있다. Reference phase를 빠르게 진행하면 같은 관절 동작을
짧은 시간에 수행하게 만들 수 있다. 구현이 간단해서 기존 motion-tracking pipeline을 확인하는
초기 실험으로는 유용하다.

하지만 reference 재생 배율과 실제 로봇의 전진 속도는 같은 개념이 아니다. 보폭은 거의 그대로인데
발만 빠르게 움직이거나, 접촉 타이밍이 깨져 발이 미끄러질 수 있다.

이번 프로젝트에서는 정책에 원하는 전진 속도를 직접 입력한다.

```text
현재 로봇 상태 + 목표 전진 속도
                ↓
             PPO Actor
                ↓
       29개 관절의 목표 위치
```

정책은 실제 root velocity가 목표 속도에 가까워지도록 학습된다. 이 방식이면 이후에 걷는 도중
목표 속도를 바꾸고 가속·감속 반응도 평가할 수 있다.

## 사용한 기존 구현과 직접 추가한 부분

첫 구현은 MimicKit과 Isaac Lab을 기반으로 만들었다.

MimicKit에서 사용한 부분은 다음과 같다.

- PPO actor/critic, rollout buffer와 update 구현
- DeepMimic observation과 imitation reward
- Isaac Lab engine adapter
- 29-DoF G1 asset
- `g1_walk.pkl` reference motion
- G1 PPO baseline 설정

직접 추가한 부분은 다음과 같다.

- Episode별 목표 속도 sampling
- PPO observation에 목표 속도 추가
- 실제 전진 속도와 목표 속도를 비교하는 reward
- DeepMimic reward와 속도 reward의 결합
- 목표 속도, 실제 속도와 speed MAE 진단값
- 프로젝트 환경을 MimicKit 학습기에 연결하는 실행 adapter

따라서 PPO 알고리즘 전체를 새로 구현한 프로젝트는 아니다. 검증된 PPO와 simulator 기반을 활용하고,
속도 조건부 학습 환경과 이후의 데이터·student·평가 pipeline을 직접 만드는 구조다.

현재 smoke test에 사용한 `/home/an/Desktop/MimicKit-main` 사본에는 Git metadata가 없어서 정확한
commit을 식별하지 못했다. 본 학습 전에는 공식 MimicKit clean checkout과 commit을 고정해야 한다.

## RL에서 transition이 만들어지는 과정

정책 학습의 가장 작은 경험 단위는 transition이다.

```text
(현재 observation, action, reward, 다음 observation, 종료 여부)
```

수식으로는 다음과 같이 쓸 수 있다.

\[
(o_t, a_t, r_t, o_{t+1}, done_t)
\]

G1 환경에서는 transition 하나가 다음 순서로 만들어진다.

```text
1. 현재 G1 상태와 목표 속도로 observation을 만든다.
2. Actor가 29개 관절 target action을 출력한다.
3. Isaac Lab이 action을 적용하고 물리 상태를 갱신한다.
4. 새로운 상태에서 imitation reward와 속도 reward를 계산한다.
5. 다음 observation과 넘어짐 여부를 기록한다.
```

현재 policy control frequency는 30Hz이고 physics simulation은 120Hz다. 따라서 Actor가 action을
한 번 출력하면 PhysX가 그 사이에 네 번의 physics step을 진행한다.

```text
Observation
→ Action 1회
→ Physics 4 step
→ Reward와 다음 observation
```

PPO는 transition 하나마다 곧바로 parameter를 수정하지 않는다. 현재 baseline은 환경별로 32 step을
수집한 뒤 rollout 전체를 사용해 Actor와 Critic을 업데이트한다. 여러 simulation 환경을 사용하면
동일한 G1 복사본들이 서로 다른 상태와 목표 속도를 동시에 경험하며, 모두 하나의 정책을 학습시킨다.

## Actor, Critic과 advantage

Actor와 Critic은 같은 observation을 보지만 역할이 다르다.

```text
Actor(observation)  → 29개 관절 action
Critic(observation) → 예상되는 미래 reward를 나타내는 value 하나
```

Rollout 이후 PPO는 실제 결과와 Critic의 예상을 비교해 advantage를 계산한다.

```text
Advantage > 0 → 예상보다 좋았던 action
Advantage < 0 → 예상보다 나빴던 action
```

Actor는 advantage가 양수인 action의 확률을 높이고 음수인 action의 확률을 낮춘다. 이때 바로 다음
reward만 보는 것이 아니라 할인된 미래 reward도 고려한다. 순간적으로 목표 속도에 도달하고 바로
넘어지는 행동보다, 약간 느리더라도 오랫동안 안정적으로 걷는 행동이 더 좋은 결과를 얻을 수 있다.

PPO는 정책이 한 번에 지나치게 바뀌는 것을 막기 위해 probability ratio를 clipping한다.
현재 설정의 `ppo_clip_ratio`는 `0.2`다. 좋은 방향으로 학습하면서도 이전 정책에서 너무 멀리
이동하지 않도록 제한하는 장치다.

## 목표 속도를 observation에 넣는 이유

목표 속도는 reward 계산에만 사용해서는 안 된다. 정책도 현재 요구받은 속도를 알아야 한다.

```text
기존 DeepMimic observation
+ 목표 전진 속도 한 값
= 속도 조건부 observation
```

코드에서는 다음과 같이 기존 observation 뒤에 목표 속도를 연결했다.

```python
obs = super()._compute_obs(env_ids)
return torch.cat([obs, target_speed.unsqueeze(-1)], dim=-1)
```

목표 속도가 observation에 없다면 동일한 로봇 상태에서 `0.8m/s`로 걸어야 하는지,
`1.2m/s`로 걸어야 하는지 Actor가 구분할 수 없다.

## 실제 전진 속도 계산

Isaac Lab에서 얻는 root velocity는 world 좌표계 기준이다. 로봇이 회전하면 world x축과 로봇의
앞쪽 방향이 일치하지 않을 수 있다. 따라서 root velocity를 로봇의 heading 기준으로 변환한다.

```python
root_vel = self._engine.get_root_vel(char_id)
heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)
local_root_vel = torch_util.quat_rotate(heading_inv_rot, root_vel)
```

변환 후 각 성분은 다음 의미를 갖는다.

```text
local_root_vel[..., 0] → 로봇 기준 전진 속도
local_root_vel[..., 1] → 로봇 기준 좌우 속도
local_root_vel[..., 2] → 위아래 속도
```

## 속도 추종 reward

현재 속도 reward는 다음과 같다.

\[
r_{speed} = \exp\left(-4(v_x-v_{cmd})^2-2v_y^2\right)
\]

- \(v_x\): 실제 전진 속도
- \(v_{cmd}\): 목표 전진 속도
- \(v_y\): 좌우 속도

목표 속도와 정확히 일치하고 옆으로 움직이지 않으면 reward는 1이다. 목표가 `1.0m/s`일 때
실제 전진 속도에 따른 reward는 다음과 같다.

| 실제 속도 | 속도 오차 | 좌우 속도 0일 때 reward |
|---:|---:|---:|
| 0.0 | -1.0 | 0.018 |
| 0.5 | -0.5 | 0.368 |
| 0.8 | -0.2 | 0.852 |
| 1.0 | 0.0 | 1.000 |
| 1.2 | 0.2 | 0.852 |
| 1.5 | 0.5 | 0.368 |

전진 속도가 정확해도 옆으로 미끄러지면 reward가 감소한다. 예를 들어 전진 속도는 정확하지만
좌우 속도가 `0.5m/s`라면 reward는 약 `0.607`이다.

## DeepMimic reward를 함께 사용하는 이유

속도 reward만 사용하면 정책이 걷지 않고도 목표 속도를 달성하는 편법을 찾을 수 있다. 몸을 앞으로
던지거나, 발을 끌거나, 불안정하게 점프하면서 순간적인 root velocity만 맞출 수도 있다.

DeepMimic imitation reward는 실제 G1과 reference motion을 다음 항목으로 비교한다.

| 항목 | 가중치 | 역할 |
|---|---:|---|
| Joint pose | 0.50 | 관절 자세를 reference와 맞춤 |
| Joint velocity | 0.10 | 관절 움직임의 시간 흐름을 맞춤 |
| Root pose | 0.15 | 몸통 높이와 방향을 유지 |
| Root velocity | 0.10 | Reference 몸통 속도를 추종 |
| Key-body position | 0.15 | 양발, 머리와 양손 위치를 맞춤 |

현재 전체 reward는 pilot 설정으로 다음과 같이 구성했다.

\[
r_{total}=0.7r_{DeepMimic}+0.3r_{speed}
\]

DeepMimic reward는 걷기다운 움직임을 유지하고, speed reward는 명령한 실제 속도를 맞추게 한다.

하지만 두 목표는 충돌할 수 있다. Reference motion은 평균 약 `1.0m/s`로 이동한다. 목표 속도가
`0.8m/s`라면 DeepMimic의 root velocity 항목은 `1.0m/s`를, 추가한 speed reward는 `0.8m/s`를
요구한다. Reward 비율과 목표 속도 범위를 실험으로 검증해야 하는 이유다.

## Curriculum 계획

처음부터 넓은 속도 범위와 보행 중 속도 전환을 모두 학습시키지 않는다.

| 단계 | 목표 속도 | 보행 중 명령 변경 |
|---|---|---|
| Baseline | 기존 reference 약 1.0m/s | 없음 |
| Stage 1 | 1.0m/s 고정 | 없음 |
| Stage 2 | 0.9~1.1m/s | 없음 |
| Stage 3 | 0.8~1.2m/s | 없음 |
| Stage 4 | 0.8~1.2m/s | 3~5초마다 변경 |
| Stage 5 | 같은 범위 | 외란 추가 또는 외란 평가 |

한 단계에서 안정적인 보행과 낮은 속도 오차를 확인한 뒤 이전 checkpoint를 다음 단계의 초기값으로
사용할 계획이다.

현재 설정 파일은 smoke test를 위해 `0.8~1.2m/s`로 작성했지만, 본 학습은 `1.0m/s` 고정부터
시작하는 것이 맞다. Smoke test는 학습 난이도를 검증한 실험이 아니기 때문이다.

## 첫 Smoke Test

첫 실행은 다음 조건으로 진행했다.

```text
Isaac Lab headless
G1 environment 1개
PPO samples 64개
32 step씩 2 iteration
```

확인한 결과는 다음과 같다.

- Isaac Lab과 G1 29-DoF model 초기화 성공
- Reference motion 로드 성공
- 속도 command가 추가된 observation으로 PPO model 생성 성공
- Rollout과 PPO update 2회 완료
- Log와 약 11MB의 `model.pt` 저장 성공
- 별도 process에서 checkpoint를 다시 불러와 test episode 실행 성공
- 목표 속도, 실제 속도와 speed MAE 진단값 기록 성공

이 결과는 학습 성능을 의미하지 않는다. 64 samples는 wiring을 확인하기 위한 매우 작은 양이며,
현재 checkpoint는 1초도 안정적으로 걷지 못한다. 짧은 episode에서 기록된 낮은 speed MAE 역시
초기 reference 상태의 영향을 받을 수 있어서 속도 제어 성공의 근거로 사용할 수 없다.

눈으로 확인할 때도 두 장면을 구분해야 한다.

```text
Reference viewer:
저장된 pose를 G1에 직접 넣어 보여줌 → PPO 제어가 아님

PPO rollout:
Actor action을 PhysX에 적용함 → 현재 smoke checkpoint는 금방 넘어지는 것이 정상
```

## 현재까지 배운 점

첫 구현을 통해 속도 조건부 PPO의 데이터 흐름을 구체적으로 정리할 수 있었다.

```text
현재 G1 상태와 목표 속도
→ Actor의 29차원 joint target
→ PhysX simulation
→ 다음 상태
→ DeepMimic reward와 speed reward
→ transition 저장
→ rollout
→ advantage와 PPO clipping을 이용한 Actor/Critic update
```

또한 simulator에서 코드가 실행된다는 사실과 학습된 정책이 과제를 잘 수행한다는 사실은 완전히
다른 검증 단계라는 점도 확인했다.

## 다음 작업

다음 목표는 속도 조건부 학습을 길게 실행하는 것이 아니다. 먼저 다음 순서로 기반을 확인한다.

1. 정확한 MimicKit commit을 고정한다.
2. 기존 G1 DeepMimic PPO baseline을 재현한다.
3. Baseline이 걷는 모습을 직접 확인하고 생존 시간과 tracking error를 측정한다.
4. 목표 속도를 `1.0m/s`로 고정한 Stage 1을 학습한다.
5. Baseline과 Stage 1의 보행 및 속도 지표를 비교한다.
6. 안정화된 checkpoint로 속도 범위를 `0.9~1.1m/s`부터 점차 넓힌다.

이 과정을 통과한 뒤 demonstration 수집과 MLP/Diffusion student 학습으로 넘어갈 예정이다.

## 관련 파일

- `src/humanoid_imitation/envs/speed_tracking_deepmimic_env.py`
- `configs/envs/g1_speed_tracking_stage1.yaml`
- `configs/agents/ppo_g1_speed_stage1.yaml`
- `scripts/train_speed_expert.py`
- `docs/experiments/ppo_speed_smoke.md`
- `docs/provenance.md`
