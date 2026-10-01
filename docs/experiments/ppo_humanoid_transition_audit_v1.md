# 다섯 모션의 실제 사용 전환 점검 v1

목적: 다섯 reference를 개별적으로 추종하는 PPO tracker가 프로젝트에서 사용할 연결도 물리적으로 수행하는지 확인한다. 평가한 연결은 `start→walk`, `walk→turn_left`, `turn_left→walk`, `walk→turn_right`, `turn_right→walk`, `walk→stop`이다. 모든 5×5 순열이나 정지 유지 능력을 시험한 것은 아니다.

## 같은 조건

- 고정 정책: `outputs/ppo_humanoid_five_motion_continue_2500_seed0/model.pt`. 추가 학습 없음.
- MimicKit generic humanoid, Isaac Lab, 환경 1개, 첫 프레임 고정 시작, seed 0, 제어 주기 30 Hz. 연결마다 **물리 rollout 1회**.
- 기존 probe는 새 모션의 root 위치·방향을 이어 붙이고, walk로 진입할 때만 관절 자세가 가장 가까운 walk phase를 고른다. 관절·접촉 blend나 물리 최적화는 없다. 이는 생성 모델의 전환 결과가 아닌 평가용 단순 연결이다.
- 각 경계 전후 0.25초의 관절 오차·지면 접촉력, 종료 시 몸통 기울기·root/발 위치 오차를 `scripts/trace_transition_tracker.py`로 기록했다. `SUCC`는 클립 길이에 도달했다는 종료 코드이며 안정적인 착지나 이후 보행을 뜻하지 않는다.

| 연결 | 경계 / 클립 끝 | 종료 / 실제 종료 | 끝 몸통 기울기 | 끝 root / 양발 위치 오차 | 판정 |
|---|---:|---|---:|---:|---|
| start→walk | 2.93 / 6.03초 | SUCC / 6.03초 | 5.7° | 0.06 / 0.05·0.07m | 이 시작 상태에서 통과 |
| walk→turn_left, 단순 연결 | 1.03 / 4.03초 | FAIL / 2.17초 | 64.1° | 0.53 / 0.85·0.86m | 실패 |
| walk→turn_left, 지지 발 정렬 | 1.25 / 4.25초 | SUCC / 4.27초 | 40.5° | 0.39 / 0.75·0.60m | 클립 끝 도달, 자세 불안정 |
| walk→turn_left, 4.5–8.0초 자연 진입 포함 | 1.91 / 5.41초 | SUCC / 5.43초 | 12.9° | 0.24 / 0.28·0.24m | 개선, 추가 검증 필요 |
| turn_left→walk | 3.00 / 4.03초 | FAIL / 3.83초 | 69.9° | 0.34 / 1.05·1.02m | 실패 |
| walk→turn_right | 1.03 / 4.03초 | FAIL / 2.00초 | 57.0° | 0.49 / 0.72·0.84m | 실패 |
| turn_right→walk | 3.00 / 4.03초 | SUCC / 4.03초 | 13.4° | 0.45 / 0.50·0.42m | 클립 끝 도달, 추종 오차 큼 |
| walk→stop | 1.03 / 2.97초 | FAIL / 1.67초 | 50.4° | 0.40 / 0.66·0.08m | 실패 |

`walk→turn_left`의 개선판은 source 클립 길이와 걷기 phase가 함께 바뀌었으므로 어느 변경의 단독 효과인지 알 수 없다. 평가 방법과 원자료는 [좌회전 자연 진입 진단](ppo_walk_turn_left_leadin_v1.md)에 있다.

단순 연결 reference 자체의 경계 차이도 크다. 관절 자세 거리(전체 관절의 합성 거리)는 walk→좌회전 1.73 rad, walk→우회전 1.54 rad, walk→정지 1.74 rad다. 해당 전환의 root 속도 차이는 각각 0.59, 0.58, 0.82 m/s다. 실패를 PPO의 학습 부족이나 diffusion의 필요성 하나로 단정할 수 없다. 회전→걷기는 경계 자세 차이가 더 작지만 좌회전→걷기는 후반에 넘어지고 우회전→걷기는 끝까지 가도 root 오차가 크다. 지지 발과 실제 상태를 고려한 진입·이탈 reference, tracker의 연결 상태 일반화를 따로 점검해야 한다.

`stop`은 정지로 *전환하는* 클립이다. 정지한 채 유지하는 stand reference와 물리적 hold 평가는 아직 없다. 그러므로 현재 결과는 정지 목표의 성공을 보여주지 않는다.

## 원자료와 다음 관문

각 연결의 시계열 CSV 및 요약 JSON은 `outputs/transition_diagnostics/`에 있다. 이번 신규 결과는 `start_to_walk_audit_seed0`, `walk_turn_left_audit_seed0`, `turn_left_walk_audit_seed0`, `walk_turn_right_audit_seed0`, `turn_right_walk_audit_seed0`, `walk_stop_audit_seed0` 접두어로 저장했다. 기존 좌회전 개선 대조군은 `walk_turn_left_contact_endcheck_seed0`, `walk_turn_left_leadin_seed0`이다. 경계 reference 수치와 source는 `data/reference_motions/humanoid_*probe*.json`에 있다.

다음에는 실패한 연결의 **reference 경계와 지지 발**을 먼저 개선하고, 동일 checkpoint·시작 조건으로 재평가한다. 한 시작 상태의 통과만으로 전환 데이터 수집을 확정하지 않는다. 적어도 클립 끝의 자세·발 접촉·정지 유지까지 확인한 뒤, 다양한 시작 phase와 반복 rollout에서 성공률을 측정한다.

후속 지지 발 정렬·원본 진입/이탈 구간 실험은 [자연 연결 시도](ppo_transition_alignment_attempts_v1.md)에 기록했다.
