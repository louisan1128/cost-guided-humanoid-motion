# 진행 기록

## 여기까지

1. G1 걷기로 속도 보상과 PPO 학습 흐름을 확인했다. generic humanoid와는 별도 실험이다.
2. Generic humanoid의 출발·걷기·좌우 회전·정지 reference를 준비하고 공통 tracker를 학습했다.
3. 전환에서 발 위상과 reference 연결 문제가 생겨 연결 방식을 보완했다.
4. 정지 유지 reference로 500 iteration 추가 학습했다. 짧은 고정 시나리오에서는 정지 유지가 개선됐다. [기록](experiments/walk_stop_hold5s_probe.md)
5. 긴 연결 장면은 사용자 관찰상 만족스럽지 않았다. 개별 모션 성공을 연속 제어 성공으로 볼 수 없다.

## 2026-10-01: 명령 조건 PPO 추가

수동 연결만 계속 조정하는 대신, 에피소드 안에서 속도·회전 명령이 바뀌는 상황을 학습한다.
PPO + AMP를 사용하며 기존 tracker와 별도 모델이다. 현재 추가한 범위는 환경·설정·launcher·고정 명령 평가 시나리오다.

다음 순서:
- 짧은 실행 검증 → 본 학습 → 고정 명령 순서 평가.
- 넘어지지 않는 것뿐 아니라 좌우 회전 부호, 속도 오차, 정지 유지 확인.
- 성공한 teacher rollout을 고정 reference tracker로 재실행해 데이터로 쓸 수 있는지 확인.
- 실제 rollout과 goal label을 만들고 검색/MLP부터 비교.

Diffusion은 아직 학습하지 않았다. 명령 정책의 성능과 데이터 수집 성공도 아직 확정하지 않았다.
