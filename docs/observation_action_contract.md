# 입출력 계약: Goal-Conditioned Motion Diffusion

> 2026-10-01 추가: 명령 조건 PPO + AMP는 별도 데이터 수집 후보다. 기존 reference tracker를 대체하지 않는다. Teacher rollout은 명령과 실제 상태를 별도로 저장하고, 공통 tracker 재실행 검증 후 채택한다. 생성기 비교에는 동일 데이터를 사용한다. 현재 teacher 성능은 미검증이다.


> 최종 설계 v1.0 · 2026-09-22. [계획](plan.md)의 설계 계약이며 runtime 검증 완료를 의미하지 않는다.

## 구성 요소

| 모듈 | 입력 | 출력 |
|---|---|---|
| Task specification | 사용자/평가 시나리오가 지정한 목표 | World 목표 xy·yaw, deadline, 목표 속도, stop/continue |
| 검색·MLP·Diffusion | 실제 현재 상태·최근 전신 이력 + 현재 좌표로 변환한 목표 조건 | 미래 전신 reference |
| 공통 adapter | 미래 reference·생성 시점 좌표·시간 | Tracker가 요구하는 pose·velocity target |
| PPO tracker | 실제 proprioception + 외부 reference의 현재/미래 target | 정규화된 관절 action |
| Simulator | action 변환·clipping·PD 처리 결과 | 다음 상태·접촉·종료 |

Tracker는 원본 reference의 데이터 수집과 생성 reference 실행에 같은 API를 사용한다. 최종 tracker는 하나로 고정한다. 별도의 명령 teacher, path follower, 상위 diffusion은 기본 구조에 없다.

## 좌표와 표현

- Simulator up-axis·전방축·yaw 부호와 각 body/joint 순서를 asset에서 확인해 기록한다.
- World 목표는 episode 동안 고정하고 매 재계획 때 현재 root heading 좌표계로 변환한다. 재계획마다 목표를 앞으로 옮기지 않는다.
- Plan 한 개의 미래 xy는 생성 시점 root 위치·heading 기준이며, z는 지면 기준 높이로 정의한다. Root 회전과 관절 표현은 artifact schema에 명시한다.
- 기본 표현은 joint DOF pose + root translation/orientation. Root 회전은 연속 표현(예: 6D)을 사용하고 정규화·퇴화 출력 처리를 검증한다. Joint 각도 wrap을 처리한다.
- Root yaw 목표는 sin/cos 또는 동등한 연속 표현을 사용한다. 목표 선속도와 각속도의 단위·좌표를 고정한다.
- dq·root velocity는 필요한 경우 공통 차분 규칙으로 구성하며 회전 차분은 quaternion/회전군 규약을 따른다. 접촉은 우선 평가 정보이며 독립 생성 target에 넣지 않는다.
- 실제 q와 PD action은 같은 값이 아니다. Runtime DOF 수, action 순서·scale·offset·clipping·gain·torque 제한을 별도 문서화한다.

기존 G1 observation/action 차원과 checkpoint는 재사용하지 않는다. Generic humanoid source frame의 34차원을 action 차원으로 가정하지 않는다.

## 시간과 정지

Physics dt, tracker control dt, motion sample dt, history, prediction horizon H, execution horizon E를 구분한다. E ≤ H이며 초기에는 짧은 E를 사용한다. 설정은 frame 수와 초를 함께 저장한다.

World deadline은 episode 시작 시 고정하고 남은 시간은 실제 경과 시간만큼 감소한다. Stop 목표 도달 후에는 같은 위치·방향과 0속도를 유지한다. Deadline 이후 모델에는 학습된 0-time hold 조건을 제공하되 평가기는 사전 설정된 grace/timeout으로 실패를 판정한다. 매번 deadline을 연장해 실패를 숨기지 않는다.

기본 task는 episode당 목표 하나다. 목표 변경 sequence는 기본 파이프라인 이후에만 추가한다. Velocity command steering은 디버그/후속 인터페이스이며 현재 목표 입력과 혼용하지 않는다.

## 외부 reference 주입

Tracker가 raw 미래 target을 관측하도록 만든다. Clip ID·고정 phase만으로 동작을 외우는 정책을 arbitrary reference tracker로 취급하지 않는다. Reference만 바꿔 simulator나 policy를 reset하지 않고 실행할 수 있어야 한다.

Chunk를 공통 world frame으로 복원하고 현재 실행 시점에 맞는 미래 target을 제공한다. 첫 target의 시간 오프셋과 t/t+1 정렬을 검증한다. Reference를 매 tracker step마다 현재 자세에 무조건 재정렬해 목표 이동 오차를 없애지 않는다. 실제 pelvis height·joint pose를 강제 설정하는 것은 replay이며 제어가 아니다.

경계 blending이나 output limit가 필요하면 세 방법 모두 같은 adapter를 사용한다. 원래 생성 결과와 adapter 적용 결과를 모두 기록한다.

## Feedback·비정상 출력

기본 history는 실제 실행 상태다. Ablation에서만 지정한 history 채널을 이전 예측으로 대체한다. 현재 상태 입력과 tracker의 실제 proprioception은 유지한다. 초기 history와 reset 처리는 공통으로 고정한다.

NaN·회전 퇴화·크기 초과·연속성 위반을 검사하고 공통으로 실패를 기록한다. 검증된 hold fallback이 있으면 사용하되 fallback을 성공으로 숨기거나 안전 보장으로 설명하지 않는다.

동기식 simulator가 inference를 기다려 주는 실행과 실제 시간 제한을 반영한 실행을 구분한다. 생성·좌표 변환·adapter까지 포함한 지연을 같은 GPU 조건에서 측정한다.
