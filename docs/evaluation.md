# 평가 프로토콜: 목표 조건과 생성–실행 오차

> 2026-10-01 추가: 명령 조건 PPO + AMP는 별도 데이터 수집 후보다. 기존 reference tracker를 대체하지 않는다. Teacher rollout은 명령과 실제 상태를 별도로 저장하고, 공통 tracker 재실행 검증 후 채택한다. 생성기 비교에는 동일 데이터를 사용한다. 현재 teacher 성능은 미검증이다.


> 최종 설계 v1.0 · 2026-09-22. 기준은 [계획](plan.md). 임계값·목표 범위는 validation pilot 후 본 평가 전에 고정한다.

## 1. 무엇을 검증하는가

주 질문은 고정 tracker에서 검색·MLP·Diffusion의 목표 달성 능력과 실패 특성이다.
H1은 offline/kinematic 품질과 실제 성공의 관계, H2는 history feedback, H3은 실행 길이 효과를 검증한다.
Diffusion의 필수성, task PPO 대비 우월성, 사람의 개발 노력 절감은 이번 비교의 결론이 아니다.

## 2. 선행 관문

1. 원본 reference를 강제 재생해 skeleton·좌표·접촉·동작 의미를 확인한다.
2. PPO tracker로 실제 추종하고 walk·좌우 선회·출발·정지·stand별 성공률과 오차를 측정한다.
3. 실제 rollout을 reference로 다시 실행해 action 의미·시간·target 변환의 일관성을 확인한다.
4. 외부 reference를 실행 중 교체하고 chunk 경계에서 reset 없이 추종하는지 확인한다.

Source가 정지해도 실제 tracker가 멈춘다는 보장은 없다. 정지와 선회가 검증된 범위만 생성기 과제로 채택한다. 기반 관문 실패 시 생성 모델 학습량을 늘리는 것으로 대체하지 않는다.

## 3. 비교 방법과 공정성

| 방법 | 구성 |
|---|---|
| 검색·연결 | Train library에서 실제 이력·목표·남은 시간에 맞는 segment를 검색하고 재계획 |
| MLP | 동일 조건에서 H-step 전신 reference를 예측하는 chunk MLP |
| Diffusion | 동일 조건에서 H-step 전신 reference를 생성, 기본 후보 1개 |
| Ground-truth reference tracking | 별도 episode의 실제 미래 모션을 제공하는 진단용 기준 |

세 방법에 같은 dataset, 조건 정보, tracker checkpoint, adapter, H/E, 초기 상태를 제공한다. 검색 비용 가중치는 validation으로 고정한다. MLP와 diffusion은 train-only 통계와 출력 표현을 공유하고 parameter 수·최적화 예산·latency를 공개한다. 무조건 같은 parameter 수가 공정성을 보장한다고 주장하지 않는다.

검색에는 성공을 알고 고른 test segment를 제공하지 않는다. Ground-truth reference tracking은 미래 정보를 보므로 생성기 순위와 분리한다. 단순 보간·고정 reference replay를 강한 검색 기준선 대신 내세우지 않는다.

목표 조건부 task PPO는 이번 필수 구현에 추가하지 않는다. 기존 G1 checkpoint나 AMP steering checkpoint는 다른 asset/정보/과제이므로 직접 성능 순위에 넣지 않는다.

## 4. 시나리오와 성공 판정

| 구분 | 내용 |
|---|---|
| Pilot | 가까운 직진 목표, 완만한 좌우 목표, 도착 후 정지 |
| ID test | 학습 운동 범위의 새 episode·초기 상태·목표 sample |
| Held-out combinations | 학습에서 제외한 상대 위치–yaw–도착 시간 조합. 주변 개별 조건은 학습 범위 안 |
| 선택적 진단 | 약한 수평 impulse 한 종류. 방향·위치·시점·지속시간 고정 |

Primary task는 목표 위치·yaw에 도달하고 낮은 선속도·yaw rate를 일정 시간 유지하는 것이다. 동일 episode 내 지속 조건을 모두 만족해야 성공이며, 순간적으로 지나간 것을 성공으로 세지 않는다. 움직이며 통과하는 continue task는 별도 보조 결과다.

아래는 pilot 시작값이지 검증된 능력이 아니다.
- 목표 반경 오차 0.20 m 이내, 방향 오차 15도 이내.
- 실제 수평 속도 0.15 m/s 이하, yaw rate 0.20 rad/s 이하로 0.5초 유지.
- Deadline과 grace window는 시나리오에 명시하고 모든 방법에 공유.
- 넘어짐 판정은 asset에 맞는 몸체 높이·기울기·비발 접촉 기준을 검증해 설정.

값을 바꾸면 validation 근거와 변경 이력을 남기고 test 전에 확정한다. 목표 지원 범위 밖의 불가능한 조건을 주 비교에 섞지 않는다. Test 결과를 보고 성공 임계값이나 목표를 바꾸지 않는다.

## 5. 필수 실험 행렬

- **E1:** 검색 / MLP / Diffusion, 실제 history, 공통 기본 E. ID와 held-out 목표 비교.
- **E2:** 같은 diffusion checkpoint에서 실제 history / 이전 예측 history. 현재 상태·tracker 관측·초기 이력·나머지 조건 유지.
- **E3:** 같은 diffusion checkpoint와 H에서 실행 길이 E를 세 수준으로 변경. 남은 goal time과 simulator 시간은 실제 경과대로 처리.
- **E4:** E1 실패 사례를 동일 상태에서 replay. 정답 reference 추종과 생성 reference 추종을 구분해 원인 확인.

Ablation을 모두 교차해 거대한 grid로 만들지 않는다. E2/E3는 대표 조건에서 먼저 측정하고 공통 seed로 확인한다. E2는 전체 open-loop 제어 실험이 아니라 generator history channel ablation임을 명시한다.

## 6. 지표

- 목표 도달·정지 성공률, 넘어짐·timeout·출력 무효·fallback 비율.
- 위치·yaw 오차, 잔여 선속도·yaw rate, 목표 도달 시간과 정지 유지.
- Reference의 목표 오차, 실제 관절·root tracking 오차, chunk 연결 불연속.
- 발 미끄러짐: 접촉 판정과 지면 상대 접선 속도 정의를 함께 기록.
- Offline pose/root 오차와 조건 만족도. Diffusion은 여러 유효 미래가 있을 수 있어 정답 한 개와의 MSE만으로 순위를 결론내리지 않음.
- 생성·변환·adapter의 latency p50/p95, 호출 수, 제어 시간 예산 초과.
- RL interaction 수, 학습·수집 GPU 시간, 모델 크기·데이터 크기·주요 수동 튜닝 횟수.

매 생성 시점에서 reference 목표 오차를 계산할 수 있는 horizon인지 확인한다. Goal이 H 밖에 있으면 chunk 끝 오차를 최종 도달 오차라고 부르지 않는다. 계획–실행 차이는 동일 timestamp의 reference와 actual로 비교한다.

성공 episode만의 이동 시간은 선택 편향을 밝히고 성공률과 함께 제시한다. 조기 넘어짐 때문에 짧아진 구간의 작은 평균 오차를 우수한 결과로 해석하지 않는다. 모션 자연스러움은 slip·불연속 지표와 영상을 함께 쓰고 인간 지각을 완전히 측정했다고 주장하지 않는다.

## 7. 반복·통계·실패 분석

초기 예산: pilot 20개 시나리오, 본 평가 ID 100개 + held-out 100개. MLP·diffusion은 각각 학습 seed 3개를 목표로 하며, 자원 부족 시 실제 반복 수와 제한을 공개한다. Pilot 후 본 학습 전에 최종 예산을 동결한다. 하나의 tracker를 쓰므로 tracker 학습 seed 일반화는 검증하지 않는다.

Diffusion sampling seed는 미리 고정하고 모델 학습 seed와 구분한다. Sampling 반복을 독립 학습 반복으로 세지 않는다. 모든 방법은 같은 시나리오를 사용하고 성공률 차이를 paired 방식으로 보고한다. 신뢰구간은 frame/window가 아니라 독립 시나리오·source group을 기준으로 계산하고 학습 seed 간 변동도 별도 보고한다.

실패는 다음 가설로 분류하고 replay 근거를 붙인다.
1. Dataset/goal 범위: 정지·회전 경험이나 목표 조합이 부족한가?
2. 생성: reference 자체가 목표를 놓치거나 비연속·비물리적인가?
3. 추종: 유효한 reference도 tracker가 실행하지 못하는가?
4. 연결·시간: 좌표, chunk 경계, latency, deadline 처리 문제인가?

실패 예시만 선별하지 않고 전체 유형별 빈도와 대표 성공·실패 영상을 남긴다. 특정 원인은 통제 실험 없이 확정하지 않는다.

## 8. 실시간·실용성 주장

Simulator가 inference를 기다린 상태의 성공은 기능적 closed-loop 검증이다. 실시간 주장은 동일 하드웨어에서 전체 계산 경로의 deadline을 검증한 뒤 한다. Tracker/data 준비 비용을 제외하고 사람의 노력이나 전체 데이터 효율이 개선됐다고 주장하지 않는다. 필수 결과가 끝나기 전에는 run·장애물·terrain 평가를 추가하지 않는다.
