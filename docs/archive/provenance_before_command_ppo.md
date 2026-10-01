# 코드·모델·데이터 출처와 직접 구현 범위

> 아래 G1 표와 64-sample checkpoint 설명은 초기 smoke 시점의 기록이다.
> 이후 모든 학습 결과가 64 samples라는 뜻이 아니다. 현재 학습 상태는 각 output의 실제 로그로 확인한다.
> 2026-09-22 최종 설계는 [목표 조건부 generic humanoid 계획](plan.md)이다.
> 새 tracker·dataset·diffusion은 아직 구현하지 않았으며 출처 항목도 실제 사용 시 갱신한다.

## 최종 설계의 출처·검증 예정 항목

- Generic humanoid: MimicKit의 `data/assets/humanoid/humanoid.xml`.
- Walk/long walk: 현재 MimicKit PKL. 원본 source, 변환, 좌표·접촉을 manifest로 검증할 예정.
- Start/stop 후보: [DeepMimic 원본 모션 폴더](https://github.com/xbpeng/DeepMimic/tree/master/data/motions/amass).
  파일 존재는 확인했지만 내용·호환성·사용 조건 검증과 변환은 미완료.
- Tracker: MimicKit DeepMimic/PPO를 기반으로 직접 학습할 예정. 원본 알고리즘 구현과 프로젝트 adapter를 구분.
- 생성 모델: 기존 diffusion 구현의 출처·commit·라이선스를 선정 시 기록할 예정.
- 직접 구현 예정: reference 주입·목표 label·공통 adapter·검색/MLP baseline·평가·실패 분석.
- 논문 역할과 채택 여부: [연구 근거](related_work.md).

SIE `run_left/right` 데이터는 기본 걷기 dataset에 포함하지 않는다. 사용하게 되면 동봉한 별도 라이선스를 확인한다.
프레임 수나 파일명 대응만으로 원본과 변환본의 정확한 동등성·호환성을 주장하지 않는다.

## 현재 실제로 사용하는 구성

| 구성 | 출처 | 현재 프로젝트에서 하는 역할 |
|---|---|---|
| MimicKit | `/home/an/Desktop/MimicKit-main` | 환경 API, DeepMimic reward, PPO 학습기, G1 설정 제공 |
| Isaac Lab | `/home/an/Desktop/mimickit/MimicKit/IsaacLab`, HEAD `4f3e808ca75fd8ea07c20d188f067980eeab411f` | PhysX 시뮬레이션 backend |
| Isaac Sim | 기존 Conda 환경의 5.1.0.0 | Isaac Lab 실행 runtime |
| G1 model | MimicKit의 `data/assets/g1/g1.xml`, `g1.usd`, mesh files | 29-DoF humanoid simulation model |
| 걷기 motion | MimicKit의 `data/motions/g1/g1_walk.pkl` | 초기 자세, 걷기 reference와 imitation reward 목표 |
| PPO baseline 설정 | MimicKit의 `data/agents/deepmimic_g1_ppo_agent.yaml` | 첫 PPO optimizer/network 설정의 기준 |

`/home/an/Desktop/MimicKit-main`에는 Git metadata가 없다. 따라서 이번 smoke test에 사용한
MimicKit snapshot의 정확한 commit은 아직 식별하지 못했다. 본 학습 전에 공식
[xbpeng/MimicKit](https://github.com/xbpeng/MimicKit)의 clean checkout과 commit을 고정해야 한다.

## 가져온 부분

현재 프로젝트가 처음부터 직접 구현한 것은 다음 항목들이 아니다.

- PPO 알고리즘, actor/critic network 생성, rollout buffer와 optimizer update
- Isaac Lab engine adapter와 PhysX simulation step
- DeepMimic의 pose·velocity·root·key-body imitation reward
- G1의 XML/USD/mesh asset과 joint/actuator 설정
- 약 1초 길이의 G1 걷기 reference motion

즉, 현재 생성된 `model.pt`는 **MimicKit의 PPO 구현으로 학습한 checkpoint**다.
64 samples밖에 학습하지 않았으므로 완성된 expert가 아니며, 연구 결과로 사용할 수 없다.

## 직접 구현한 부분

| 파일 | 직접 추가한 기능 |
|---|---|
| `src/humanoid_imitation/envs/speed_tracking_deepmimic_env.py` | 목표 속도 sampling, observation 추가, 속도 reward, 혼합 reward, 속도 진단값 |
| `scripts/train_speed_expert.py` | 프로젝트 환경을 MimicKit에 등록하고 train/test를 실행하는 adapter |
| `configs/envs/g1_speed_tracking_stage1.yaml` | 속도 조건부 curriculum의 첫 환경 설정 |
| `configs/agents/ppo_g1_speed_smoke.yaml` | 32~64 sample 배선 확인용 축소 설정 |
| `configs/agents/ppo_g1_speed_stage1.yaml` | 이후 baseline 학습에 사용할 PPO 후보 설정 |

현재 reward는 다음 형태다.

```text
0.7 × DeepMimic imitation reward + 0.3 × speed tracking reward
```

속도 reward는 몸체 heading 기준 실제 전진 속도와 목표 속도의 오차, 측면 속도를 사용한다.
목표 속도는 reference 평균 속도 약 1.0m/s 주변의 0.8~1.2m/s에서 episode마다 뽑는다.
이 값과 혼합 비율은 pilot 설정이며 본 학습 전에 비교·검증한다.

## 눈으로 볼 때 구분할 대상

1. **Reference motion:** 저장된 pose를 시간에 따라 G1에 직접 넣어 재생한다. 물리적으로 PPO가 제어하는 장면이 아니다.
2. **PPO rollout:** PPO가 매 control step마다 29차원 joint target action을 출력하고 PhysX가 다음 상태를 계산한다.
3. **속도 조건부 expert:** 충분히 학습된 뒤 목표 속도별 PPO rollout을 비교해야 한다. 현재 checkpoint는 여기에 해당하지 않는다.

향후 외부 코드나 asset을 추가할 때 URL, commit, 라이선스, 재사용 파일과 수정 범위를 이 문서에 갱신한다.
