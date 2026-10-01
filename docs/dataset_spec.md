# Dataset 명세: 목표 조건부 전신 모션

> 2026-10-01 추가: 명령 조건 PPO + AMP는 별도 데이터 수집 후보다. 기존 reference tracker를 대체하지 않는다. Teacher rollout은 명령과 실제 상태를 별도로 저장하고, 공통 tracker 재실행 검증 후 채택한다. 생성기 비교에는 동일 데이터를 사용한다. 현재 teacher 성능은 미검증이다.


> 최종 설계 v1.0 · 2026-09-22. 기준은 [계획](plan.md). 저장 포맷·차원·시간 수치는 pilot 후 고정한다.

## 1. 자료와 학습 역할

| 자료 | 용도 |
|---|---|
| 원본 reference | Generic humanoid walk·좌우 선회·출발·감속·정지·stand. PPO tracker 학습 |
| 실제 tracker rollout | 생성 모델과 검색 library의 운동 데이터 |
| 목표 label | 실제 미래 root 위치·방향·속도와 도착까지 시간 |
| 독립 평가 시나리오 | 초기 상태, 목표, deadline, 종료 조건을 고정한 실제 실행 평가 |

별도 command teacher는 필수가 아니다. 같은 reference-conditioned tracker가 공개 모션을 실행한 실제 상태를 수집한다. 강제 pose 재생과 물리 rollout은 구분한다. 원본 모션과 실행 모션을 모두 남겨 오차를 진단한다.

## 2. 원본 데이터 관문

- 각 source의 URL·revision·파일 hash·라이선스·skeleton·관절 순서·시간 규약을 manifest에 남긴다.
- Walk와 turn의 내용은 root 궤적·yaw와 영상으로 확인한다. 좌표축 확인 전 yaw-rate 부호만으로 좌우를 확정하지 않는다.
- Run-left/right를 walk-turn으로 취급하지 않는다. 임의 감속 재생의 동역학적 유효성을 가정하지 않는다.
- DeepMimic ready_to_walk / walk_to_ready는 출발·정지 후보이며 검증된 데이터가 아니다.
- Stand 자세의 반복은 움직임에서 감속하여 멈추는 transition을 대신하지 않는다.
- 같은 원본의 crop·mirror·속도 변형은 같은 source group이다. 변환본과 원본을 다른 split으로 나누지 않는다.

## 3. 저장 필드

| 그룹 | 최소 내용 |
|---|---|
| 식별·시간 | source_group, clip_id, episode_id, seed, timestep, timestamp, control_dt |
| 실제 상태 | q, dq, root position/orientation, root linear/angular velocity |
| 진단 | 발 위치·접촉력, 접촉 판정, 넘어짐·종료 사유 |
| 실행 입력 | 원본 reference의 ID·시간·상태, 적용한 action |
| 목표 | 목표 world 위치·yaw, 도착 속도, stop/continue 구분, 목표 timestamp |
| 종료 | terminated, truncated, 실제 terminal observation, 성공 선별 사유 |
| 버전 | simulator·asset·tracker checkpoint·adapter·수집 코드 hash |

관절 자세, PD target, 정규화 action을 섞지 않는다. 34차원 source frame을 observation이나 action 차원으로 기록하지 않는다. Reset 이전 마지막 상태와 auto-reset 이후 첫 상태를 구분한다.

## 4. Hindsight 목표 구성

학습 sample은 다음과 같다.

```text
입력 = 현재 실제 상태 + 최근 실제 모션
     + 목표 상대 위치 + 목표 상대 yaw + 남은 시간
     + 목표 도착 속도 + stop/continue 조건

정답 = 같은 episode에서 이어지는 미래 전신 상태 sequence
```

현재 시점 t에서 미래 시점 t_g를 골라, 실제 root 위치·yaw·속도를 목표로 붙인다. 도착 시간도 t_g - t로 명시한다. 이는 추종 중 주어진 명령의 성공 label이 아니라 실제로 도달한 endpoint의 재라벨링이다. 의도한 reference와 실제 결과의 오차는 별도로 남긴다.

- Stop label은 t_g 이후에도 저속·낮은 yaw rate가 정해진 시간 유지되는 경우에만 생성한다.
- Continue label은 이동 중 endpoint를 사용한다. 이를 정지 과제로 평가하지 않는다.
- Stop 직전 감속과 stop 이후 유지 window를 모두 수집한다. 목표 시간이 0인 stand 유지 sample도 포함한다.
- 미래 목표는 외부 task 입력으로 허용하지만, 미래의 실제 관절·접촉·전체 궤적은 입력하지 않는다.
- History와 target이 episode 경계를 넘지 않는다. Clip 종료를 임의로 성공적인 stop으로 바꾸지 않는다.
- Goal timestamp와 prediction horizon은 별개다. 예측 구간보다 먼 목표를 사용하려면 해당 시간 간격의 학습 sample이 있어야 한다.
- 지원 범위를 벗어난 목표·시간 조건은 평가의 외삽 진단으로 분리한다.

초기 pilot은 작은 goal-time 범위에서 시작하고, 상대 목표 분포와 유형별 sample 수를 그린 뒤 확대한다. Stop sample만 복제해 개수를 늘린 것을 새로운 경험으로 세지 않는다.

## 5. Split과 누수 방지

1. Source group을 train/validation/test로 나누고, 각 그룹에서 독립 rollout을 수집한다. Tracker 학습에도 test source를 사용하지 않는 구성을 우선한다.
2. 생성기용 목표 위치·방향·시간 조합의 보류 구간을 window 생성 전에 정한다. Train window의 label이 test 보류 조합에 속하면 train에서 제외한다.
3. 같은 episode의 모든 history/target window는 같은 split이다. 정규화 통계와 검색 library는 train만 사용한다.
4. Tracker가 본 source·전환 범위와 생성기가 본 목표 label 범위를 각각 공개한다. 생성기의 보류 목표가 tracker가 전혀 보지 않은 기술임을 뜻하지 않는다.
5. Source가 적어 독립 source test가 불가능하면 source 공유를 명시하고, 독립 episode·초기 상태·목표 조합 test만 주장한다. 새 모션 일반화라고 부르지 않는다.
6. 같은 endpoint에서 여러 window를 만들더라도 독립 평가 표본 수를 부풀리지 않는다.

목표 holdout만 제거해도 동일 공간 위치로의 평행이동이 새로운 기술이 되지는 않는다. Body-local 상대 위치·yaw·시간 조합을 기준으로 분포를 기록한다.

## 6. 수집 품질과 예산

Nominal 성공 rollout을 먼저 수집하고 실패는 분석용으로 보존한다. 선별 기준·성공 비율·제외 이유를 공개한다. 선택적 recovery 데이터는 별도 버전으로 구성한다. 불안정한 미래를 성공 target처럼 섞거나, 실행하지 않은 clean action에 대응하는 미래 모션을 실제 정답으로 취급하지 않는다.

Pilot에서 저장·재생·tracker 재실행·목표 label을 확인한 후 확대한다. Manifest에는 clip/episode/window 수, 전환·stop 비율, 추종 성공률, 수집 interaction 수, GPU 시간, 저장 크기를 기록한다. 초기 8GB GPU·디스크 기록은 과거 관측이므로 현재 자원을 다시 확인하고 수집 예산을 정한다.
