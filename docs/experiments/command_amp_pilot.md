# Command PPO + AMP 초기 실행 점검 — 2026-10-01

목적: 수동 연결 reference 대신 에피소드 내 `(vx, wz)` 변화에 반응하는 정책의 학습 경로를 추가한다.
기존 tracker는 보존한다. 새 정책은 임의 reference를 추종하는 tracker가 아니다.

## 변경

- `src/humanoid_imitation/envs/command_amp_env.py`: 명령 관측, 구간별 재샘플링, 선속도/각속도 task reward.
- `configs/command/`: 학습 환경, AMP agent, reference 목록, 고정 명령 평가.
- `scripts/train_command_policy.py`: 외부 MimicKit 연결, 덮어쓰기 방지, iteration 예산, manifest 저장.
- `tests/test_command_env.py`: 보상 부호, 일부 환경 resampling, 명령 변경 경계의 보상 정합성, 고정 schedule 검사.

기본 PPO와 AMP 코드는 MimicKit에서 사용한다. 프로젝트 코드는 AI 도움을 받아 작성·검증했다.

## 실제 확인

- CPU 테스트 4개 통과.
- 최종 reference 목록으로 Isaac Lab 16 env × 2 iteration = 1,024 samples 학습 완료.
- `outputs/command_amp_smoke_v2/`: model, env/agent/engine 설정, log 저장.
- PPO loss와 AMP discriminator loss가 유한한 값으로 출력됨.
- 저장 모델을 고정 명령 평가 환경에서 불러오는 headless test 1 episode 완료.
- GUI는 이번 변경에서 실행하지 않았다.

초기 무작위 정책의 짧은 실행이다. 정상 보행, 명령 변화 시 회복, 20초 schedule 완주를 검증한 결과가 아니다.
학습 도중 기본 평가 때문에 실제 simulator interaction은 train Samples보다 많다.
새 launcher의 manifest/launch metadata 저장은 smoke 이후 추가했으므로 위 smoke 폴더에는 없다.

## 다음

1024 env에서 1000 iteration을 첫 본 학습 예산으로 실행한다. AMP가 추가되어 4096 env 메모리 사용량은 미검증이다.
평가에서는 좌우 명령별 오차를 분리하고 정지 구간의 실제 속도와 넘어짐을 확인해야 한다.
좋은 command rollout이 생겨도 고정 tracker 재실행 확인 전에는 Diffusion 데이터로 확정하지 않는다.

## 1024 env에서 발생한 버퍼 오류 수정

초기 disc_buffer_size=20000은 16 env smoke에서는 충분했지만 1024 × 32 = 32768개의 첫 rollout을 담지 못했다. MimicKit AMP는 버퍼가 차기 전에는 전체 rollout을 push하므로 assertion이 발생했다. 버퍼를 65536으로 수정하고 launcher에서 num_envs × steps_per_iter보다 작은 버퍼를 simulator 실행 전에 차단한다.

수정 후 실제 Isaac Lab 1024 env × 1 iteration 학습 종료를 확인했다. 결과는 `outputs/command_amp_buffer_fix_smoke/`. 4096 env는 131072 용량이 필요하여 현재 설정에서는 사전 차단됨을 확인했다. 이는 실행 검증이며 보행 성능 검증은 아니다.
