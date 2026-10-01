# 프로젝트 작업 지침

기준: [최종 계획](docs/plan.md), **Goal-Conditioned Motion Diffusion for Closed-Loop Humanoid Control**.
최종 설계 v1.0 · 2026-09-22. 사용자 최신 지시를 우선한다.

## 목적과 고정 방향

학부 개인 Robot Learning 포트폴리오다. 사용자가 설계·구현·실험을 직접 설명할 수 있어야 한다.
MimicKit generic humanoid 하나가 평지에서 가까운 목표 위치·도착 방향·정지 조건을 수행한다.
새 논문을 이유로 주 연구 질문이나 구조를 반복 변경하지 않는다.

- RL: 공개 reference를 추종하는 PPO tracker를 직접 학습·검증한다.
- 데이터: tracker의 실제 rollout과 그 미래 endpoint에서 목표 label을 만든다.
- 검색·연결 / MLP / Motion Diffusion: 실제 현재 상태·최근 이력·목표로 미래 전신 reference를 생성한다.
- 실행: 공통 adapter와 하나의 고정 tracker로 물리 실행하고 실제 상태를 피드백한다.

2026-10-01 사용자 요청: 명령 조건 PPO + AMP를 데이터 수집 후보로 추가한다. 기존 reference tracker와 별도 모델이며 가중치를 혼용하지 않는다. Command rollout은 공통 tracker의 재실행 검증 후 생성기 데이터로 채택한다. 무제한 명령 학습을 완료 조건으로 늘리지 않는다.
이전 G1 속도 expert는 개발 이력이며 새 tracker나 generic humanoid checkpoint로 취급하지 않는다.
생성한 reference와 PD action을 구분한다. BeyondMimic의 state-action diffusion, SMP의 reward model과 혼동하지 않는다.

## 읽을 문서

[계획](docs/plan.md), [Dataset](docs/dataset_spec.md), [계약](docs/observation_action_contract.md),
[평가](docs/evaluation.md), [연구 근거](docs/related_work.md), [출처](docs/provenance.md).
현재 실행 안내는 [명령 조건 PPO](configs/command/README.md), 과거 G1 안내는 docs/archive에 보존한다.
계획·파일 존재는 구현·호환·학습 성공의 증거가 아니다.

## 사용자와 작업하는 방식

- 한국어로 역할과 이유를 먼저 설명한다. “다음”은 현재 단계의 다음 작업이다.
- 개념 질문만으로 파일을 수정하거나 실험을 실행하지 않는다.
- 구현 요청은 요청 범위의 변경과 가벼운 검증을 완료한다. 이미 받은 권한을 반복 확인하지 않는다.
- 기본적으로 학습·시각화 명령은 사용자가 실행하도록 제공한다. 명시적으로 맡긴 실행만 수행한다.
- 문서 검증에 장시간 GPU 학습·simulator 실행을 포함하지 않는다.
- 사용자의 이해와 직접 수행한 부분·외부 코드·AI 도움을 함께 기록한다.

## 구현 원칙

- 단일 generic humanoid와 단일 backend를 사용한다. 기존 Isaac Lab부터 확인하고 버전은 smoke 후 고정한다.
- 외부 MimicKit는 보존하며 가능한 변경은 프로젝트 adapter/config에 둔다.
- 기존 G1 코드·checkpoint·로그를 삭제하거나 덮어쓰지 않는다. 프로젝트 결정 YAML과 runtime 설정을 구분한다.
- 실행 명령은 CLI와 실제 경로를 확인한 후 작성한다. 미구현 진입점을 실행 가능한 것처럼 안내하지 않는다.
- 가중치 로드와 optimizer·step을 포함한 resume를 구분한다.
- 프로젝트와 무관한 사용자 변경은 보존한다. 현재 학습 상태는 실제 확인한 범위만 보고한다.
- Tests는 변경에 맞게 수행하고 simulator 검증 여부를 별도로 보고한다.

## 순서와 범위

Reference·환경 audit → PPO tracker → 실제 rollout·목표 label pilot
→ 검색·MLP closed-loop → Diffusion → 비교·실패 분석 → 보고서.

첫 관문은 walk·좌우 선회·출발·정지 reference의 내용과 물리 tracking이다.
Run-left/right를 걷기 turn으로 취급하지 않는다. Static stand와 stop transition을 구분한다.
원본 DeepMimic start/stop 후보는 변환·재생 전까지 미검증이다.
Tracker가 저장 reference도 못 따라가면 diffusion을 먼저 학습하지 않는다.

필수: 평지 목표 이동·도착 방향·정지 유지, 세 생성 방식 비교, feedback와 실행 길이 ablation.
보고서 완성 이후 첫 확장은 정적 장애물 하나. Run은 별도 동작 확장, terrain은 후속 프로젝트.
자연어·vision·SLAM·실제 로봇·G1 재이식·동적 장애물·전역 navigation·residual RL·
state-action diffusion·SMP 보상·자동 실패 증강·새 생성 구조는 기본 범위에 추가하지 않는다.

## 데이터·입출력

- Source group, episode, 목표 label과 실제 상태·원본 reference를 구분한다.
- Crop·mirror·반복 rollout을 같은 split으로 묶고 train-only 정규화를 사용한다.
- Tracker가 본 source와 생성기가 본 목표 조합을 따로 기록한다. Source를 공유하면 새 모션 일반화를 주장하지 않는다.
- 목표는 실제 미래 endpoint로 relabel할 수 있지만 미래 관절·접촉·궤적을 입력에 누출하지 않는다.
- Stop label은 실제 저속·저회전 유지 구간에서만 만든다. Clip/window 끝을 stop으로 간주하지 않는다.
- World 목표와 deadline을 고정한다. Replanning마다 목표·deadline을 재설정하지 않는다.
- 관절 순서·좌표·단위·시간·회전·velocity 차분을 검증한다. 34차원 frame을 action 차원으로 가정하지 않는다.
- PPO tracker는 외부 reference target을 관측해야 한다. Clip ID만으로 외우는 정책은 범용 tracker가 아니다.
- 실제 pose를 강제 설정한 재생과 물리적 추종을 구분한다. Reset·terminal observation·history를 정확히 처리한다.

## 비교·완료

검색·MLP·Diffusion에 동일 데이터·정보·고정 tracker·adapter·H/E·초기 상태·목표를 제공한다.
기본 diffusion은 후보 1개다. 임의 K 후보 선택기를 기본 범위에 추가하지 않는다.
Feedback ablation은 generator의 history만 바꾸고 현재 상태와 tracker의 실제 관측은 유지한다.
실행 길이 ablation은 동일 checkpoint·H에서 E만 변경한다.
Tracker 보완은 train에서 공통 데이터·변형을 사용한다. 변경 후 모든 방법을 같은 버전으로 다시 평가한다.

목표 성공률·오차·넘어짐·정지·slip·reference 추종·latency·학습 비용을 함께 기록한다.
미래 정답 reference tracking은 진단용이며 공정한 생성기 기준선이나 상한이 아니다.
주요 생성기 seed 3개를 목표로 하고 pilot 이후 평가 예산·임계값을 고정한다.
실패는 데이터/생성/추종/연결·지연 가설로 분류하고 replay로 확인한다.
Diffusion 우월성, 무데이터 기술 습득, human effort 절감, 실시간, sim-to-real, 합격을 보장하지 않는다.

최종 기준은 계획의 체크리스트다. 기능 성공과 분석 완료를 구분한다.
모델 크기·horizon·목표 범위는 pilot 조정 대상이지만 연구 질문과 구조는 고정한다.
불가피한 범위 변경은 근거와 이전 결과를 기록하고 사용자와 명시적으로 결정한다.
